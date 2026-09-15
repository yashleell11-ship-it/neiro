"""The rules the persona/Hinglish/tool-calling LoRA's data path is built
on -- everything about the 437k prepared chat records that can be
decided without torch.

`training/recipes/persona_train.py` needs CUDA torch and lives in the
training venv. Everything below is importable from the runtime venv too
(it has CPU torch and `transformers`, just no GPU) so the rules that
decide what the model is trained on are testable without a GPU, exactly
the split `elizabeth.training.stt_data` uses for the Hindi STT fine-tune.

**Qwen3.5's chat template is a hybrid-thinking template, and it renders
one assistant turn differently from every other one.** Empirically, on
this model's own `chat_template.jinja` (see
`models/qwen3.5-4b-safetensors/chat_template.jinja`):

    >>> messages = [
    ...     {"role": "user", "content": "hi"},
    ...     {"role": "assistant", "content": "hello there"},
    ...     {"role": "user", "content": "how are you"},
    ...     {"role": "assistant", "content": "good, you?"},
    ... ]
    >>> tokenizer.apply_chat_template(messages, tokenize=False)
    '<|im_start|>user\\nhi<|im_end|>\\n<|im_start|>assistant\\nhello there<|im_end|>\\n
     <|im_start|>user\\nhow are you<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n
     </think>\\n\\ngood, you?<|im_end|>\\n'

"hello there" -- an assistant turn that is already-resolved HISTORY --
renders with no think tags at all. "good, you?" -- the reply to the
LAST real user turn, i.e. the one an actual generation call would be
producing -- renders wrapped in an EMPTY `<think>\\n\\n</think>\\n\\n`
scaffold. That empty scaffold is exactly what the server injects as a
literal PROMPT PREFIX before every generation via
`chat_template_kwargs: {enable_thinking: false}` (CLAUDE.md rule 4;
`elizabeth.llm.openai_compat.OpenAiCompatLlm.build_request`) -- the model
never has to predict those four tokens itself, they are always handed
to it. Two consequences:

1. Training on our data (no record ever carries a `reasoning_content`
   field) naturally produces the non-thinking path with NO extra flag:
   the template's own fallback, when `reasoning_content` is undefined,
   is `content.split('</think>')` on text that never contains
   `</think>` -- so `reasoning_content` is always `''` and the emitted
   scaffold is always the empty one, never a real thinking block. This
   is asserted, not assumed: `build_training_chat_template` is checked
   against the live vendor template's exact text at import time (a
   template rewrite that changes this shape raises loudly instead of
   silently training on whatever the new shape happens to render), and
   `training/recipes/persona_train.py --dry-run` prints a rendered
   sample and refuses to continue if it contains a non-empty
   `<think>` block.
2. **Naive incremental-prefix masking is unsafe here.** The standard
   trick for finding which tokens in one `apply_chat_template` call
   belong to which message -- render `messages[:1]`, `messages[:2]`,
   ... and diff the token counts -- silently breaks on this template,
   because whether a given assistant turn gets the think-scaffold
   depends on whether it is *the last one*, and truncating the message
   list changes which turn that is. Rendering `messages[:2]` above
   (`["hi", "hello there"]`) treats "hello there" as the final turn and
   WOULD wrap it in the empty scaffold -- a token sequence that is not
   even a prefix of the real, full-conversation rendering. Diffing
   against that would misalign every label after the first such turn,
   silently, with no error.

   The fix used here is `transformers`' own mechanism for this exact
   problem: a chat template can wrap the text it wants scored in
   `{% generation %}...{% endgeneration %}` tags, and
   `tokenizer.apply_chat_template(..., return_assistant_tokens_mask=True)`
   returns a token-aligned mask built from those tags -- with **zero**
   effect on the rendered text (the tags are transparent). The vendor
   template has no such tags, so `build_training_chat_template` returns
   a copy with tags inserted around exactly the assistant `content` /
   tool-call XML / trailing `<|im_end|>` (never around the role header
   or the think-scaffold, which are prompt context, not something the
   model predicts). Verified byte-identical rendered text against the
   unpatched vendor template across chat, system+chat and tool-call
   conversations before this was trusted for anything.

**Tool-call arguments need parsing before they reach the template.**
The prepared records store `tool_calls[].function.arguments` as an
OpenAI-shape JSON-encoded STRING (`'{"amount": 1000}'`) -- but the
template does `tool_call.arguments|items`, which requires an already
-parsed mapping and raises `TypeError: Can only get item pairs from a
mapping` on a string. `parse_tool_call_arguments` does that parse
before anything is rendered; a record whose arguments do not parse as
JSON is dropped and counted rather than guessed at.

**The labels are next-token-shifted, same as any causal LM loss.** The
mask `apply_assistant_mask` builds lines up 1:1 with `input_ids`; the
one-position shift between a hidden state and the token it predicts is
the recipe's job (see its `masked_loss`), not this module's -- but it
is called out here because getting it wrong produces no error, just a
model trained to predict its own current token instead of the next
one, and was caught only by comparing byte-for-byte against
`transformers`' own reference loss with dropout disabled.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from elizabeth.evals.latency import percentile

IGNORE_INDEX = -100

# Every reason a record can be dropped before it reaches the model. Naming
# them here rather than accepting any string means a typo becomes an
# error instead of a silent new bucket nobody reads -- the same policy
# `elizabeth.training.stt_data.SkipLog` uses.
SKIP_KINDS: tuple[str, ...] = (
    "not_publishable",  # source's weights_publishable is not "yes", --publishable-only
    "bad_tool_args",  # tool_calls[].function.arguments did not parse as JSON
    "no_assistant_tokens",  # after templating (and any truncation), nothing to score
)


@dataclass
class SkipLog:
    """Records dropped before training, counted rather than silent.

    A `--publishable-only` run that quietly drops a third of the corpus
    and a clean run must not look the same in the report.
    """

    counts: Counter[str] = field(default_factory=Counter)

    def skip(self, kind: str, n: int = 1) -> None:
        if kind not in SKIP_KINDS:
            raise ValueError(f"unknown skip kind {kind!r}; add it to SKIP_KINDS")
        self.counts[kind] += n

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def as_dict(self) -> dict[str, Any]:
        return {"total": self.total, "by_reason": dict(sorted(self.counts.items()))}


# --------------------------------------------------------------------------
# reading the prepared JSONL


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """One dict per non-blank line. Never loads the file into memory: this
    is the lazy read `scripts/prep_text.py`'s writer and this reader agree
    on, so 437k rows never sit in RAM as one list.
    """
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def index_jsonl_lines(path: str | Path) -> list[int]:
    """The byte offset of every non-blank line in `path` -- one sequential
    pass, O(n) small ints in memory (437k lines is ~3.5 MB of offsets),
    never the file's content.

    This is what makes a GLOBAL shuffle of a >1 GB file possible without
    holding it in RAM: `scripts/prep_text.py` writes one source's rows
    contiguously before moving to the next, so `train.jsonl` is NOT
    shuffled -- the first 40,266 lines are entirely GoEmotions
    (`emotion_text`, tiny records), and the last 139,223 are entirely
    IndicTalk (`chat`, `hi`+`hinglish`). This is not a small effect: an
    earlier version of this data path used a fixed-size reservoir
    -shuffle buffer (`buffer_size=20000`) on the raw line stream, and
    because every source's block is contiguous and several exceed that
    buffer size, the first tens of thousands of examples it produced were
    still effectively 100% GoEmotions -- a buffer fed nothing but one
    source can only ever emit that source, no matter how it shuffles
    internally. Shuffling the INDEX instead of streaming the content
    fixes this at the root: `iter_jsonl_at_offsets` then reads in
    whatever order the offsets list says, so the first record it yields
    can come from anywhere in the file.
    """
    offsets: list[int] = []
    with Path(path).open("rb") as fh:
        offset = fh.tell()
        for raw_line in fh:
            if raw_line.strip():
                offsets.append(offset)
            offset = fh.tell()
    return offsets


def iter_jsonl_at_offsets(path: str | Path, offsets: Sequence[int]) -> Iterator[dict[str, Any]]:
    """Read the JSONL records at `offsets`, in that exact order -- one
    open file handle, one seek+readline per offset. Pairs with
    `index_jsonl_lines`: shuffle the offsets list (cheap: a list of
    ints), then stream through this to get a globally-shuffled read of a
    file too large to shuffle by holding its rows.
    """
    with Path(path).open("rb") as fh:
        for offset in offsets:
            fh.seek(offset)
            yield json.loads(fh.readline().decode("utf-8"))


def parse_tool_call_arguments(
    messages: Sequence[Mapping[str, Any]], log: SkipLog | None = None
) -> list[dict[str, Any]] | None:
    """Parse every `tool_calls[].function.arguments` JSON string into a
    mapping, the shape Qwen3.5's chat template requires.

    Returns `None` (and, if `log` is given, counts `bad_tool_args`) when
    an arguments string does not parse -- a corrupt row is dropped, never
    guessed at. Messages with no tool calls pass through as shallow
    copies; the input is never mutated.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        calls = message.get("tool_calls")
        if message.get("role") != "assistant" or not calls:
            out.append(dict(message))
            continue
        new_calls = []
        for call in calls:
            function = dict(call.get("function") or {})
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments) if arguments.strip() else {}
                except json.JSONDecodeError:
                    if log is not None:
                        log.skip("bad_tool_args")
                    return None
            function["arguments"] = arguments
            new_calls.append({**call, "function": function})
        out.append({**message, "tool_calls": new_calls})
    return out


def find_tool_call_turn(messages: Sequence[Mapping[str, Any]]) -> int | None:
    """The index of the first assistant message that carries `tool_calls`,
    or `None`.

    Used for the tool-call behavioural eval: the PROMPT is everything
    before this turn, not everything before the record's last message --
    a `kind == "tool_call"` conversation often continues past the call
    (the tool's response, then a plain-language follow-up), and testing
    generation from `messages[:-1]` would silently evaluate chat quality
    instead of tool-calling.
    """
    for index, message in enumerate(messages):
        if message.get("role") == "assistant" and message.get("tool_calls"):
            return index
    return None


def filter_publishable(
    records: Iterable[Mapping[str, Any]],
    per_source: Mapping[str, Mapping[str, str]],
    log: SkipLog,
) -> Iterator[Mapping[str, Any]]:
    """Drop every record whose `source` is not `weights_publishable: yes`
    in `per_source` (the table `elizabeth.training.licences.licences_for`
    returns), counting what was dropped rather than skipping quietly.
    """
    for record in records:
        info = per_source.get(record.get("source", ""))
        if info is None or info.get("weights_publishable") != "yes":
            log.skip("not_publishable")
            continue
        yield record


# --------------------------------------------------------------------------
# the training-only chat template


# The vendor template's assistant block, verbatim (see the module
# docstring for why it needs patching). Matched against the live file at
# call time rather than trusted from memory: a template rewrite that
# changes this text must fail loudly, not silently patch nothing and
# train on an unmasked or misaligned target.
_ASSISTANT_BLOCK_OLD = (
    "        {%- if loop.index0 > ns.last_query_index %}\n"
    "            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content"
    " + '\\n</think>\\n\\n' + content }}\n"
    "        {%- else %}\n"
    "            {{- '<|im_start|>' + message.role + '\\n' + content }}\n"
    "        {%- endif %}\n"
    "        {%- if message.tool_calls and message.tool_calls is iterable"
    " and message.tool_calls is not mapping %}"
)
_ASSISTANT_BLOCK_NEW = (
    "        {%- if loop.index0 > ns.last_query_index %}\n"
    "            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content"
    " + '\\n</think>\\n\\n' }}\n"
    "        {%- else %}\n"
    "            {{- '<|im_start|>' + message.role + '\\n' }}\n"
    "        {%- endif %}\n"
    "        {%- generation %}\n"
    "        {{- content }}\n"
    "        {%- if message.tool_calls and message.tool_calls is iterable"
    " and message.tool_calls is not mapping %}"
)
_ASSISTANT_TAIL_OLD = (
    "                {{- '</function>\\n</tool_call>' }}\n"
    "            {%- endfor %}\n"
    "        {%- endif %}\n"
    "        {{- '<|im_end|>\\n' }}\n"
    '    {%- elif message.role == "tool" %}'
)
_ASSISTANT_TAIL_NEW = (
    "                {{- '</function>\\n</tool_call>' }}\n"
    "            {%- endfor %}\n"
    "        {%- endif %}\n"
    "        {{- '<|im_end|>\\n' }}\n"
    "        {%- endgeneration %}\n"
    '    {%- elif message.role == "tool" %}'
)


def build_training_chat_template(vendor_template: str) -> str:
    """The vendor's chat template, with `{% generation %}` markers around
    exactly what an assistant turn actually generates: its content, its
    tool-call XML, and the closing `<|im_end|>`. Never the role header or
    the `<think>` scaffold, which are always prompt context (see the
    module docstring) -- the model is never asked to predict them.

    Raises if the vendor text does not contain the exact block this
    patch targets, so a future template rewrite fails the run instead of
    silently training on an unmasked or misaligned target.
    """
    if _ASSISTANT_BLOCK_OLD not in vendor_template:
        raise ValueError(
            "vendor chat_template.jinja's assistant-block text did not match what "
            "build_training_chat_template expects to patch -- the template changed "
            "shape; re-derive _ASSISTANT_BLOCK_OLD/_NEW against the new source"
        )
    if _ASSISTANT_TAIL_OLD not in vendor_template:
        raise ValueError(
            "vendor chat_template.jinja's assistant-block tail did not match what "
            "build_training_chat_template expects to patch -- the template changed "
            "shape; re-derive _ASSISTANT_TAIL_OLD/_NEW against the new source"
        )
    patched = vendor_template.replace(_ASSISTANT_BLOCK_OLD, _ASSISTANT_BLOCK_NEW, 1)
    return patched.replace(_ASSISTANT_TAIL_OLD, _ASSISTANT_TAIL_NEW, 1)


def apply_assistant_mask(
    input_ids: Sequence[int], assistant_masks: Sequence[int], ignore_index: int = IGNORE_INDEX
) -> list[int]:
    """`input_ids` where `assistant_masks` is truthy, `ignore_index`
    elsewhere -- the label array a causal LM loss should be given.

    Pure and torch-free on purpose: the one place this repo's whole
    masking policy is expressed as a list operation, so it is testable
    without a tokenizer, a template, or a model.
    """
    if len(input_ids) != len(assistant_masks):
        raise ValueError(
            f"input_ids has {len(input_ids)} tokens but assistant_masks has "
            f"{len(assistant_masks)} -- they must come from the same render"
        )
    return [token if mask else ignore_index for token, mask in zip(input_ids, assistant_masks)]


def render_and_mask(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None,
    chat_template: str,
    max_length: int,
) -> dict[str, list[int]] | None:
    """Tokenize one conversation and build its assistant-only labels.

    Truncation drops tokens from the FRONT, not the back: the oldest
    turns are what a long conversation can best afford to lose, and the
    assistant target this record exists to train on sits at the end. If
    that leaves nothing for the loss to score (a pathological record, or
    a target longer than `max_length` on its own), returns `None`.
    """
    rendered = tokenizer.apply_chat_template(
        list(messages),
        tools=list(tools) if tools else None,
        chat_template=chat_template,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    ids = list(rendered["input_ids"])
    mask = list(rendered["assistant_masks"])
    if len(ids) > max_length:
        ids = ids[-max_length:]
        mask = mask[-max_length:]
    if not any(mask):
        return None
    return {"input_ids": ids, "labels": apply_assistant_mask(ids, mask)}


# --------------------------------------------------------------------------
# sequence length


def choose_max_length(lengths: Sequence[int], q: float, *, round_to: int = 64) -> int:
    """The smallest multiple of `round_to` at or above the `q`-th
    nearest-rank percentile of `lengths` (via `elizabeth.evals.latency.
    percentile` -- the one percentile function in this repo, never a
    mean: CLAUDE.md rule 8).

    `round_to` exists because padding a batch to an odd number like 1316
    buys nothing -- every example still pads to the batch's own longest
    member, so only the boundary's granularity matters, and a round
    multiple is easier to reason about in the report than an arbitrary
    integer measured off one sample.
    """
    if not lengths:
        raise ValueError("cannot choose a max length from zero measured examples")
    value = percentile([float(n) for n in lengths], q)
    return round_to * math.ceil(value / round_to)


# --------------------------------------------------------------------------
# the tool-call structural check


_TOOL_CALL = re.compile(
    r"<tool_call>\s*<function=([^>\n]*)>(.*?)</function>\s*</tool_call>", re.DOTALL
)
_PARAMETER = re.compile(r"<parameter=[^>\n]*>.*?</parameter>", re.DOTALL)


@dataclass(frozen=True)
class ToolCallCheck:
    """A structural verdict, never a semantic one -- whether the text
    parses as Qwen3.5's own tool-call format, not whether the call was
    the *right* one. `function_known` is `None` when no declared tool
    names were given to check against.
    """

    well_formed: bool
    function_names: tuple[str, ...]
    function_known: bool | None
    reason: str | None = None


def check_tool_call_shape(text: str, declared_names: Iterable[str] | None = None) -> ToolCallCheck:
    """Does `text` contain at least one well-formed
    `<tool_call><function=NAME>...</function></tool_call>` block, in
    exactly the shape the model was trained to emit (see the tool
    -rendering half of `chat_template.jinja`)?

    Structural, on purpose -- matching the task's ask for "a structural
    check, not semantic equality": every `<function=...>` tag names a
    non-empty function, and every `<parameter=...>` tag it contains is
    closed. Not checked: whether the arguments are well-typed JSON or
    the right values -- that is a different, harder question this check
    does not claim to answer.
    """
    calls = _TOOL_CALL.findall(text)
    if not calls:
        return ToolCallCheck(False, (), None, "no <tool_call> block found")
    names = tuple(name.strip() for name, _ in calls)
    if any(not name for name in names):
        return ToolCallCheck(False, names, None, "a <function=> tag named nothing")
    for _, body in calls:
        if body.count("<parameter=") != len(_PARAMETER.findall(body)):
            return ToolCallCheck(False, names, None, "an unclosed <parameter> tag")
    known = None
    if declared_names is not None:
        declared = set(declared_names)
        known = all(name in declared for name in names)
    return ToolCallCheck(True, names, known, None)


# --------------------------------------------------------------------------
# turning the adapter back into what the daemon can load


def gguf_convert_commands(
    merged_dir: str | Path, f16_out: str | Path, quant_out: str | Path, quant_type: str = "Q4_K_M"
) -> tuple[list[str], list[str]]:
    """The two llama.cpp commands that turn a merged safetensors model
    into what `elizabeth`'s daemon actually loads -- an f16 GGUF, then a
    quantized one, the same two-step pattern the repo's own
    `models/qwen3.5-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf` was produced by.

    A LoRA adapter, and even a merged safetensors model, is invisible to
    the daemon: it loads a GGUF through its own runtime. This is a
    command to print, not to run from here -- `convert_hf_to_gguf.py`
    and `llama-quantize` are llama.cpp tools, not a training-venv
    dependency, and are not installed on this machine as of this run
    (`pacman -Qs llama` and `which convert_hf_to_gguf.py` both came back
    empty; only `ollama` is present).
    """
    convert = [
        "python",
        "convert_hf_to_gguf.py",
        str(merged_dir),
        "--outfile",
        str(f16_out),
        "--outtype",
        "f16",
    ]
    quantize = ["llama-quantize", str(f16_out), str(quant_out), quant_type]
    return convert, quantize
