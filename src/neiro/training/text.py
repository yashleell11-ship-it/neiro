"""Turn the downloaded TEXT corpora into one chat format for the persona /
Hinglish / tool-calling LoRA.

Thirteen corpora, thirteen layouts: Dolly-Hinglish is a parquet table,
OASST2 is a forest of ranked reply trees, Glaive is a 271 MB JSON list of
`USER: / ASSISTANT: / FUNCTION RESPONSE:` transcripts, xLAM stores its
tool list as a JSON string inside JSON, IndicTalk is 780 MB of jsonl.
This module hides all of that behind one row shape:

    Record(messages, tools, source, licence, lang, kind)

where `messages` is the OpenAI chat shape (`role`/`content`, and for tool
use `tool_calls` / `tool_call_id`) and `tools` is the OpenAI function
list — so `apply_chat_template` on the 3090 Ti renders every corpus with
the same template, and adding a corpus is one reader function rather
than a change to the recipe.

**Why the licence rides on every row.** A LoRA is one artifact. One NC
corpus in the mix means the whole adapter can never leave the machine,
and that decision has to be visible in the *prepared* file — the recipe
never sees the manifest. `load()` refuses sources the manifest marks
non-publishable rather than including them quietly; `Record.licence`
and the `publishable` flag in `stats.json` carry the answer forward.

**Why the emotion corpora are not chat.** GoEmotions and BRIGHTER are
labelled sentences, not dialogue. A Reddit comment does not answer
anything, and pairing it with an invented reply would teach her to
reply to sadness with a label. They become short `(text -> label)`
examples in the vocabulary of her own `<e:LABEL:D>` tag so the recipe
can condition her register on them however it likes. No corpus here
records an *intensity*, so the `D` digit is never fabricated.

Readers are deliberately tolerant, like `corpora.py`: a row that does
not parse is dropped, never guessed, and the caller reports the count.
A count of zero is visible.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from neiro.training.manifest import NON_PUBLISHABLE_FLAGS, Dataset, Manifest

REPO = Path(__file__).resolve().parents[3]
DATASETS_DIR = REPO / "data" / "datasets"
PREPARED_DIR = REPO / "data" / "prepared" / "text"
COMPLETE_MARKER = ".neiro-complete"  # written by scripts/fetch_datasets.py

Lang = Literal["en", "hi", "hinglish"]
Kind = Literal["chat", "tool_call", "emotion_text"]

# The six labels of her tag — prompts/neiro.v4.md and llm/emotion_tag.py
# agree on exactly these. Emotion-text corpora are mapped onto this set
# and nothing else, because the point is conditioning *her* register.
TAG_LABELS: tuple[str, ...] = ("happy", "angry", "sad", "relaxed", "surprised", "neutral")

# Split: a record's side is a stable hash of its content, so re-running
# prep after adding a corpus never moves an existing row across the line.
# 5%, not corpora.py's 15%: these are 400k+ rows and val only has to be
# large enough to notice a loss curve turning, not to estimate a metric.
VAL_FRACTION = 0.05
SPLIT_BUCKETS = 1000
DEFAULT_SEED = 0

# Rows per pyarrow batch. Only a memory knob — small enough that a
# 211k-row GoEmotions table never sits in RAM twice.
PARQUET_BATCH_ROWS = 4096

_TABLE_SUFFIXES = (".parquet", ".jsonl", ".jsonl.gz", ".csv", ".json")


@dataclass
class Record:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    source: str
    licence: str
    lang: Lang
    kind: Kind

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical(self) -> str:
        """One byte string per distinct *conversation*: the messages and
        the tools, nothing else. Not the licence text — rewording a
        manifest entry must not move rows across the split — and not
        the source, so the same exchange arriving from two corpora
        (Hermes re-renders 5k of Glaive) dedupes and lands on one side.
        `sort_keys` so two readers building the same dict in different
        orders hash the same.
        """
        return json.dumps(
            {"messages": self.messages, "tools": self.tools}, sort_keys=True, ensure_ascii=False
        )


class LicenceRefused(Exception):
    """A source the manifest says cannot be trained on for release."""


# --- text hygiene -------------------------------------------------------

# Everything she writes is spoken aloud. IndicTalk and Persona-Chat are
# LLM-made and decorated with emoji; a TTS engine reads "😟" as
# "worried face" or as silence, and either way it is not how a person
# talks. Pictographs, dingbats, the variation selector and the ZWJ that
# glue them together.
_EMOJI_RANGES: tuple[tuple[int, int], ...] = (
    (0x1F000, 0x1FAFF),  # every emoji block: pictographs, emoticons, transport, flags
    (0x2600, 0x27BF),  # miscellaneous symbols and dingbats
    (0x2B50, 0x2B50),  # star
    (0x2B55, 0x2B55),  # hollow circle
    (0x231A, 0x231B),  # watch, hourglass
    (0x23E9, 0x23FA),  # media-control arrows
    (0xFE0F, 0xFE0F),  # variation selector 16
    (0x200D, 0x200D),  # zero-width joiner
)
_EMOJI = re.compile(
    "[" + "".join(f"{re.escape(chr(a))}-{re.escape(chr(b))}" for a, b in _EMOJI_RANGES) + "]"
)
_SPACES = re.compile(r"\s+")


def _clean(text: str | None) -> str:
    return _EMOJI.sub("", text or "").strip()


def _spoken(text: str | None) -> str:
    """The stronger clean for corpora that are meant to be *said*: no
    markdown bold, no line breaks. Not applied to OASST2 or the tool
    corpora, where a code block or a JSON body is the content.
    """
    return _SPACES.sub(" ", _clean(text).replace("**", "")).strip()


# --- table access -------------------------------------------------------
#
# HF snapshots arrive as parquet, jsonl, jsonl.gz, csv or one JSON list,
# and the same corpus can change container between revisions. Readers
# ask for "the tables under this directory" and never spell a suffix, so
# a test can drop a three-row `.jsonl` where the real corpus has parquet
# and exercise the whole mapping without pyarrow.


def _tables(directory: Path, prefix: str = "") -> list[Path]:
    if not directory.is_dir():
        return []
    out = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.name.startswith(prefix) and p.name.endswith(_TABLE_SUFFIXES)
    ]
    return sorted(out)


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    name = path.name
    if name.endswith(".parquet"):
        yield from _parquet_rows(path)
    elif name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif name.endswith(".jsonl"):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif name.endswith(".csv"):
        with path.open(encoding="utf-8", newline="") as f:
            yield from csv.DictReader(f)
    elif name.endswith(".json"):
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        yield from data if isinstance(data, list) else [data]


def _parquet_rows(path: Path) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # the runtime venv has no pyarrow, on purpose
        raise ImportError(
            f"{path.name} is parquet and pyarrow is not installed here — run prep from "
            "the training venv: `cd training && uv run python ../scripts/prep_text.py`"
        ) from exc
    reader = pq.ParquetFile(path)
    for batch in reader.iter_batches(batch_size=PARQUET_BATCH_ROWS):
        yield from batch.to_pylist()


# --- OpenAI shapes ------------------------------------------------------

# Corpus type spellings → JSON-schema types. Explicit, like labels.py's
# ALIASES: an unknown spelling stays as it is rather than being guessed.
_SCHEMA_TYPES: dict[str, str] = {
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "double": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "list": "array",
    "array": "array",
    "tuple": "array",
    "dict": "object",
    "object": "object",
}


def _schema_type(spelling: str) -> str:
    """`"str, optional"` → `"string"`. The `, optional` suffix is xLAM's
    (and, inherited, When2Call's) way of saying not-required; it is not
    part of the type.
    """
    key = spelling.split(",")[0].strip()
    if key.lower().startswith(("list[", "list<")):
        return "array"
    return _SCHEMA_TYPES.get(key.lower(), key)


def _openai_tool(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }


def _fix_schema_types(node: Any) -> Any:
    """Rewrite `"type": "dict"` / `"float"` spellings inside a schema
    tree in place. When2Call and xLAM write Python types where JSON
    schema wants JSON ones; a chat template does not care, but a tool
    validator on the runtime side will.
    """
    if isinstance(node, dict):
        if isinstance(node.get("type"), str):
            node["type"] = _schema_type(node["type"])
        for v in node.values():
            _fix_schema_types(v)
    elif isinstance(node, list):
        for v in node:
            _fix_schema_types(v)
    return node


class _Calls:
    """Hands out `call_N` ids within one record and matches tool results
    back to them by position — the OpenAI shape needs `tool_call_id`, and
    none of these corpora carry ids of their own.
    """

    def __init__(self) -> None:
        self.n = 0
        self.pending: list[str] = []

    def call(self, name: str, arguments: Any) -> dict[str, Any]:
        cid = f"call_{self.n}"
        self.n += 1
        self.pending.append(cid)
        args = (
            arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
        )
        return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}

    def result(self, content: Any) -> dict[str, Any] | None:
        if not self.pending:
            return None  # a result with no call to answer: the row is malformed
        body = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        return {"role": "tool", "tool_call_id": self.pending.pop(0), "content": body}


def _assistant(content: str | None, calls: list[dict[str, Any]]) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": content or None}
    if calls:
        msg["tool_calls"] = calls
    return msg


def _finish(messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """A record ends on her turn. Trailing user turns nobody answered and
    trailing tool results nobody read (Glaive has transcripts cut off
    mid-`FUNCTION RESPONSE`) are trimmed; what is left must still be an
    exchange. An assistant turn that only issued calls is a fine ending —
    xLAM is nothing but those — the results are the runtime's job.
    """
    while messages and messages[-1]["role"] != "assistant":
        messages.pop()
    return messages if len(messages) >= 2 else None


def _dialogue(turns: Iterable[tuple[str, str]]) -> list[dict[str, Any]] | None:
    """Alternating user/assistant messages from (speaker, text) turns.

    Consecutive turns by one speaker merge into one message, trailing
    user turns are trimmed, and a conversation that never reaches an
    assistant turn is None. The first speaker is the user by definition:
    these corpora are two people talking, and she has to be one of them.
    """
    out: list[dict[str, Any]] = []
    first: str | None = None
    for speaker, text in turns:
        text = _spoken(text)
        if not text:
            continue
        if first is None:
            first = speaker
        role = "user" if speaker == first else "assistant"
        if out and out[-1]["role"] == role:
            out[-1]["content"] += " " + text
        else:
            out.append({"role": role, "content": text})
    return _finish(out)


# --- per-corpus readers -------------------------------------------------


def read_dolly_hinglish(root: Path) -> Iterator[Record]:
    """aaditya/databricks-dolly-15k-Hinglish-Codemix: the Dolly instruction
    set rendered in Roman Hinglish. Only the codemix side — the English
    original is a different dataset with its own manifest entry.
    """
    for table in _tables(root / "data"):
        for row in _rows(table):
            instruction = _clean(row.get("codemix_instruction"))
            output = _clean(row.get("codemix_output"))
            if not instruction or not output:
                continue
            context = _clean(row.get("codemix_input"))
            user = f"{instruction}\n\n{context}" if context else instruction
            yield Record(
                messages=[
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": output},
                ],
                tools=None,
                source="databricks-dolly-15k-hinglish",
                licence="",
                lang="hinglish",
                kind="chat",
            )


# OASST2 tags every message with a language; these are the two she speaks.
_OASST_LANGS: dict[str, Lang] = {"en": "en", "hi": "hi"}


def _oasst_paths(node: dict[str, Any], prefix: list[dict[str, Any]]) -> Iterator[list[dict]]:
    """Every root-to-leaf conversation through the best-ranked replies.

    At a prompter turn, follow the rank-0 reply (human raters' choice);
    when nothing was ranked, the sole reply. At an assistant turn every
    prompter follow-up is a different conversation. Deleted messages end
    the path where they stand.
    """
    if node.get("deleted"):
        return
    text = _clean(node.get("text"))
    if not text:
        return
    role = "user" if node.get("role") == "prompter" else "assistant"
    path = [*prefix, {"role": role, "content": text}]
    replies = [r for r in node.get("replies") or [] if not r.get("deleted")]
    if role == "user":
        chosen = [r for r in replies if r.get("rank") == 0]
        if not chosen and replies and all(r.get("rank") is None for r in replies):
            chosen = replies[:1]
        if not chosen:
            if len(prefix) >= 2:
                yield prefix  # ends on the last assistant turn
            return
        for r in chosen:
            yield from _oasst_paths(r, path)
    else:
        if not replies:
            yield path
        for r in replies:
            yield from _oasst_paths(r, path)


def read_oasst2(root: Path) -> Iterator[Record]:
    """OpenAssistant/oasst2 `ready` trees — the general-chat anchor that
    keeps the adapter from collapsing into tool-only or persona-only
    behaviour. Trees, not the flat message table: a message's rank only
    means anything relative to its siblings.
    """
    for table in _tables(root):
        # `all.trees` is a superset holding the unfinished trees too;
        # reading both would double every ready conversation.
        if "ready" not in table.name or "trees" not in table.name:
            continue
        for tree in _rows(table):
            prompt = tree.get("prompt") or {}
            lang = _OASST_LANGS.get(prompt.get("lang", ""))
            if lang is None:
                continue
            for path in _oasst_paths(prompt, []):
                yield Record(
                    messages=path,
                    tools=None,
                    source="openassistant-oasst2",
                    licence="",
                    lang=lang,
                    kind="chat",
                )


# The two pair-only corpora (Hinglish-TOP, PHINC) have no reply to learn.
# What they do have is what he will *say* in Hinglish, so the exchange
# taught is comprehension: hear Hinglish, say plainly in English what
# was meant. Byte-identical across every row, deliberately.
COMPREHENSION_SYSTEM = "Say in plain English what he just said."


def read_hinglish_top(root: Path) -> Iterator[Record]:
    """rvv-karma/English-Hinglish-TOP, human-written rows only. The 170k
    synthetic rows are machine translations of the same assistant-domain
    requests and would outweigh every other Hinglish source twelve to
    one; the 14k human rows are the register that matters.
    """
    for table in _tables(root / "data"):
        for row in _rows(table):
            if row.get("generated_by") != "human":
                continue
            hinglish, english = _clean(row.get("hi_en")), _clean(row.get("en"))
            if not hinglish or not english:
                continue
            yield Record(
                messages=[
                    {"role": "system", "content": COMPREHENSION_SYSTEM},
                    {"role": "user", "content": hinglish},
                    {"role": "assistant", "content": english},
                ],
                tools=None,
                source="hinglish-top",
                licence="",
                lang="hinglish",
                kind="chat",
            )


PERSONA_SYSTEM_PREFIX = "Your persona:\n"
_PERSONA_LINE = re.compile(r"^User ([12]): ?(.*)$")


def read_synthetic_persona_chat(root: Path) -> Iterator[Record]:
    """google/Synthetic-Persona-Chat: two-party chats where each side stays
    in character. User 2 answers User 1, so User 2 is her; User 2's
    persona lines become the system prompt, because staying consistent
    with a stated persona is the skill being taught.
    """
    for table in _tables(root / "data"):
        for row in _rows(table):
            transcript = row.get("Best Generated Conversation") or ""
            turns: list[tuple[str, str]] = []
            for line in transcript.splitlines():
                m = _PERSONA_LINE.match(line.strip())
                if m:
                    turns.append((m.group(1), m.group(2)))
                elif turns and line.strip():
                    turns[-1] = (turns[-1][0], turns[-1][1] + " " + line.strip())
            messages = _dialogue(turns)
            if messages is None:
                continue
            persona = [_spoken(p) for p in (row.get("user 2 personas") or "").splitlines()]
            lines = "\n".join(f"- {p}" for p in persona if p)
            if lines:
                messages.insert(0, {"role": "system", "content": PERSONA_SYSTEM_PREFIX + lines})
            yield Record(
                messages=messages,
                tools=None,
                source="synthetic-persona-chat",
                licence="",
                lang="en",
                kind="chat",
            )


def _xlam_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """xLAM's `{name: {description, type, default}}` parameter map into a
    JSON-schema object. A parameter is required unless it says
    `optional` or carries a default — xLAM marks both ways.
    """
    props: dict[str, Any] = {}
    required: list[str] = []
    for pname, spec in (tool.get("parameters") or {}).items():
        spelling = str(spec.get("type", ""))
        prop: dict[str, Any] = {
            "type": _schema_type(spelling),
            "description": spec.get("description", ""),
        }
        if "default" in spec:
            prop["default"] = spec["default"]
        elif "optional" not in spelling:
            required.append(pname)
        props[pname] = prop
    return _openai_tool(
        tool["name"],
        tool.get("description", ""),
        {"type": "object", "properties": props, "required": required},
    )


def read_xlam(root: Path) -> Iterator[Record]:
    """Salesforce/xlam-function-calling-60k: one query, one or more
    execution-verified calls. The cleanest argument-typing supervision
    in the mix, and the only one with parallel calls.
    """
    for table in _tables(root):
        for row in _rows(table):
            try:
                tools = [_xlam_tool(t) for t in json.loads(row["tools"])]
                answers = json.loads(row["answers"])
                query = _clean(row["query"])
            except (KeyError, TypeError, ValueError):
                continue
            if not tools or not query or not isinstance(answers, list):
                continue
            calls = _Calls()
            tool_calls = [calls.call(a["name"], a.get("arguments", {})) for a in answers]
            yield Record(
                messages=[{"role": "user", "content": query}, _assistant(None, tool_calls)],
                tools=tools,
                source="xlam-function-calling-60k",
                licence="",
                lang="en",
                kind="tool_call",
            )


_HERMES_ROLES = {"system": "system", "human": "user", "gpt": "assistant", "tool": "tool"}
# 793 single-turn rows write the tag padding as a *literal* backslash-n
# (`<tool_call>\n{...}\n</tool_call>`); the pattern accepts it so the
# body is at least looked at. Every one of those bodies then fails —
# Python-quoted strings, and `name` nested inside `arguments` — and the
# row is dropped. What must never happen is the tag surviving into her
# text: `_hermes_messages` checks for it after the substitution.
_TOOL_CALL = re.compile(r"<tool_call>(?:\s|\\n)*(\{.*?\})(?:\s|\\n)*</tool_call>", re.DOTALL)
_TOOL_RESPONSE = re.compile(r"<tool_response>\s*(\{.*?\})\s*</tool_response>", re.DOTALL)
_TOOLS_TAG = "<tools>"


def _hermes_messages(conv: list[dict[str, Any]], has_tools: bool) -> list[dict] | None:
    calls = _Calls()
    out: list[dict[str, Any]] = []
    for turn in conv:
        role = _HERMES_ROLES.get(turn.get("from", ""))
        value = turn.get("value") or ""
        if role is None:
            return None
        if role == "system":
            # A system turn carrying a <tools> block is Hermes' own
            # boilerplate about XML tags our format does not have — the
            # chat template renders `tools` itself — so it goes, with or
            # without a tool list beside it. Any other system turn (the
            # JSON-mode rows' schema, the Glaive negatives' "no access to
            # functions") *is* the task and stays.
            if _TOOLS_TAG not in value and _clean(value):
                out.append({"role": "system", "content": _clean(value)})
            continue
        if role == "assistant":
            tool_calls = []
            for body in _TOOL_CALL.findall(value):
                try:
                    call = json.loads(body)
                    tool_calls.append(calls.call(call["name"], call.get("arguments", {})))
                except (KeyError, TypeError, ValueError):
                    return None
            if tool_calls and not has_tools:
                return None  # a call with nothing offered to call
            text = _clean(_TOOL_CALL.sub("", value))
            if "<tool_call>" in text or (not text and not tool_calls):
                return None
            out.append(_assistant(text, tool_calls))
            continue
        if role == "tool":
            bodies = _TOOL_RESPONSE.findall(value)
            if not bodies:
                return None
            for body in bodies:
                try:
                    payload = json.loads(body)
                except ValueError:
                    return None
                if not isinstance(payload, dict):
                    return None
                # Usually `{"name": ..., "content": <result>}`; the Glaive
                # subset sometimes writes the result object bare
                # (`{"name": "James"}` from generate_random_name). Then
                # the object *is* the result — the reply that follows
                # quotes it — and an empty tool turn would be the guess.
                msg = calls.result(payload.get("content", payload))
                if msg is None:
                    return None
                out.append(msg)
            continue
        text = _clean(value)
        if not text:
            return None
        out.append({"role": "user", "content": text})
    return _finish(out)


def read_hermes(root: Path) -> Iterator[Record]:
    """NousResearch/hermes-function-calling-v1: ShareGPT rows with
    <tool_call>/<tool_response> tags, plus two JSON-mode subsets whose
    `schema` lives in the system turn and which have no tools at all.
    """
    for table in _tables(root):
        for row in _rows(table):
            tools: list[dict[str, Any]] | None = None
            raw = row.get("tools")
            if raw:
                try:
                    parsed = json.loads(raw) if isinstance(raw, str) else raw
                except ValueError:
                    continue
                # 865 rows of the Glaive subset carry the string "null":
                # the "no access to external functions" negatives. Plain
                # chat, with that sentence kept as the system turn. 61
                # rows carry "[]" and then call a function anyway; those
                # fall in _hermes_messages.
                if isinstance(parsed, list):
                    tools = [t for t in parsed if isinstance(t, dict) and "function" in t] or None
            messages = _hermes_messages(row.get("conversations") or [], tools is not None)
            if messages is None:
                continue
            yield Record(
                messages=messages,
                tools=tools,
                source="hermes-function-calling-v1",
                licence="",
                lang="en",
                kind="tool_call" if tools else "chat",
            )


_GLAIVE_TURN = re.compile(r"^(USER|ASSISTANT|FUNCTION RESPONSE): ?", re.MULTILINE)
# Glaive writes the arguments as a Python-quoted string inside JSON:
#     {"name": "get_news", "arguments": '{"country": "France"}'}
# which is not JSON. Most of the corpus is like this (14,225 of the
# first 20,000 calls), so it is the normal case, not an exception.
_GLAIVE_CALL = re.compile(
    r"^\{\s*\"name\":\s*\"(?P<name>[^\"]+)\",\s*\"arguments\":\s*'(?P<args>.*)'\s*\}$", re.DOTALL
)
_ENDOFTEXT = "<|endoftext|>"


def _glaive_call(body: str) -> tuple[str, Any] | None:
    try:
        call = json.loads(body)
        args = call.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
        return call["name"], args
    except (KeyError, TypeError, ValueError, AttributeError):
        pass
    m = _GLAIVE_CALL.match(body.strip())
    if not m:
        return None
    try:
        # Inside the Python-quoted string an apostrophe is written `\'`
        # (124 rows: "It\'s raining"); JSON has no such escape.
        return m["name"], json.loads(m["args"].replace("\\'", "'"))
    except ValueError:
        return None


def _glaive_tools(system: str) -> list[dict[str, Any]] | None:
    """The function list is N JSON objects concatenated after a sentence,
    not a JSON array — `raw_decode` walks them one at a time. A row
    stating "no access to external functions" has none, and that is the
    row's point.
    """
    start = system.find("{")
    if start == -1:
        return None
    dec = json.JSONDecoder()
    tools: list[dict[str, Any]] = []
    i = start
    while i != -1:
        obj, end = dec.raw_decode(system, i)
        if "name" not in obj:
            return None
        tools.append(
            _openai_tool(obj["name"], obj.get("description", ""), obj.get("parameters") or {})
        )
        i = system.find("{", end)
    return tools or None


def _glaive_messages(chat: str) -> list[dict] | None:
    parts = _GLAIVE_TURN.split(chat)
    if len(parts) < 3:
        return None
    calls = _Calls()
    out: list[dict[str, Any]] = []
    for marker, body in zip(parts[1::2], parts[2::2], strict=True):
        body = body.replace(_ENDOFTEXT, "").strip()
        if marker == "USER":
            if not body:
                return None
            out.append({"role": "user", "content": _clean(body)})
        elif marker == "FUNCTION RESPONSE":
            msg = calls.result(body)
            if msg is None:
                return None
            out.append(msg)
        else:
            text, sep, call_body = body.partition("<functioncall>")
            tool_calls = []
            if sep:
                parsed = _glaive_call(call_body.strip())
                if parsed is None:
                    return None
                tool_calls.append(calls.call(*parsed))
            text = _clean(text)
            if not text and not tool_calls:
                return None
            out.append(_assistant(text, tool_calls))
    return _finish(out)


def read_glaive(root: Path) -> Iterator[Record]:
    """glaiveai/glaive-function-calling-v2: `{system, chat}` transcripts.
    About half the rows have tools available and make no call — the
    negatives a desktop-driving assistant needs most — so `kind` is
    `tool_call` whenever tools are *offered*, called or not.
    """
    for table in _tables(root):
        for row in _rows(table):
            try:
                tools = _glaive_tools(row.get("system") or "")
            except ValueError:
                continue
            messages = _glaive_messages(row.get("chat") or "")
            if messages is None:
                continue
            yield Record(
                messages=messages,
                tools=tools,
                source="glaive-function-calling-v2",
                licence="",
                lang="en",
                kind="tool_call" if tools else "chat",
            )


_W2C_CALL = re.compile(r"<TOOLCALL>\s*(\[.*\])\s*</TOOLCALL>", re.DOTALL)


def _when2call_assistant(msg: dict[str, Any], calls: _Calls) -> dict[str, Any] | None:
    content = msg.get("content") or ""
    m = _W2C_CALL.search(content)
    if not m:
        text = _clean(content)
        return _assistant(text, []) if text else None
    try:
        parsed = json.loads(m.group(1))
        tool_calls = [calls.call(c["name"], c.get("arguments", {})) for c in parsed]
    except (KeyError, TypeError, ValueError):
        return None
    return _assistant(_clean(_W2C_CALL.sub("", content)), tool_calls)


def read_when2call(root: Path) -> Iterator[Record]:
    """nvidia/When2Call `train/`: the SFT rows (tools offered, the right
    answer is a question, a refusal or a plain reply) and the preference
    rows' *chosen* side (which is where the actual calls are). The test
    split is left alone — it is a benchmark.
    """
    for table in _tables(root / "train"):
        for row in _rows(table):
            try:
                raw_tools = [json.loads(t) if isinstance(t, str) else t for t in row["tools"]]
                tools = [
                    _openai_tool(
                        t["name"],
                        t.get("description", ""),
                        _fix_schema_types(t.get("parameters") or {}),
                    )
                    for t in raw_tools
                ]
            except (KeyError, TypeError, ValueError):
                continue
            history = list(row.get("messages") or [])
            if "chosen_response" in row:
                history.append(row["chosen_response"])
            calls = _Calls()
            messages: list[dict[str, Any]] = []
            for msg in history:
                role = msg.get("role")
                if role == "assistant":
                    rendered = _when2call_assistant(msg, calls)
                    if rendered is None:
                        messages = []
                        break
                    messages.append(rendered)
                elif role in ("user", "system"):
                    text = _clean(msg.get("content"))
                    if text:
                        messages.append({"role": role, "content": text})
                else:
                    messages = []
                    break
            if len(messages) < 2 or messages[-1]["role"] != "assistant":
                continue
            # 3,084 of the 24,000 training rows offer no tools at all: the
            # "nothing fits, answer in words" negatives. Plain chat.
            yield Record(
                messages=messages,
                tools=tools or None,
                source="when2call",
                licence="",
                lang="en",
                kind="tool_call" if tools else "chat",
            )


# --- emotion text -------------------------------------------------------
#
# Three explicit sets per corpus, the way labels.py keeps ALIASES explicit:
# a label with a home on her tag, a label that is not an emotion for our
# purposes and is ignored beside the others, and a label that IS an
# emotion but has no home (fear, disgust) — a row carrying one of those
# is dropped rather than filed under whatever else it also carries.

GOEMOTIONS_NAMES: tuple[str, ...] = (
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral",
)  # fmt: skip
_GOEMOTIONS_TO_TAG: dict[str, str] = {
    "admiration": "happy",
    "amusement": "happy",
    "excitement": "happy",
    "gratitude": "happy",
    "joy": "happy",
    "love": "happy",
    "optimism": "happy",
    "pride": "happy",
    "anger": "angry",
    "annoyance": "angry",
    "disappointment": "sad",
    "grief": "sad",
    "remorse": "sad",
    "sadness": "sad",
    "relief": "relaxed",
    "surprise": "surprised",
    "neutral": "neutral",
}
_GOEMOTIONS_IGNORED: frozenset[str] = frozenset(
    {"approval", "disapproval", "caring", "confusion", "curiosity", "desire", "realization"}
)
_BRIGHTER_TO_TAG: dict[str, str] = {
    "joy": "happy",
    "anger": "angry",
    "sadness": "sad",
    "surprise": "surprised",
}


def _tag_label(
    names: Iterable[str], mapping: dict[str, str], ignored: frozenset[str]
) -> str | None:
    """One of TAG_LABELS, or None when the row is ambiguous or homeless.
    No labels at all is `neutral` — BRIGHTER's convention, and the only
    honest reading of "the raters marked nothing".
    """
    homes: set[str] = set()
    for n in names:
        if n in mapping:
            homes.add(mapping[n])
        elif n not in ignored:
            return None
    if not homes:
        return "neutral"
    return homes.pop() if len(homes) == 1 else None


def _emotion_record(text: str, label: str, source: str, lang: Lang) -> Record:
    return Record(
        messages=[{"role": "user", "content": text}, {"role": "assistant", "content": label}],
        tools=None,
        source=source,
        licence="",
        lang=lang,
        kind="emotion_text",
    )


def read_goemotions(root: Path) -> Iterator[Record]:
    """google-research-datasets/go_emotions, `simplified` config: 54k
    Reddit comments where at least two raters agreed. Not `raw` — those
    are per-rater votes, and one comment appearing five times with five
    different labels is noise dressed as data.
    """
    for table in _tables(root / "simplified"):
        for row in _rows(table):
            text = _clean(row.get("text"))
            ids = row.get("labels")
            if not text or not isinstance(ids, list):
                continue
            try:
                names = [GOEMOTIONS_NAMES[int(i)] for i in ids]
            except (IndexError, ValueError, TypeError):
                continue
            label = _tag_label(names, _GOEMOTIONS_TO_TAG, _GOEMOTIONS_IGNORED)
            if label is not None:
                yield _emotion_record(text, label, "goemotions", "en")


def read_brighter_hindi(root: Path) -> Iterator[Record]:
    """brighter-dataset/BRIGHTER-emotion-categories, config `hin`: real
    Devanagari sentences with gold labels. The only Hindi text tied to
    her exact label set; `emotions == []` is the corpus's own neutral.
    """
    for table in _tables(root / "hin"):
        for row in _rows(table):
            text = _clean(row.get("text"))
            emotions = row.get("emotions")
            if not text or not isinstance(emotions, list):
                continue
            label = _tag_label(emotions, _BRIGHTER_TO_TAG, frozenset())
            if label is not None:
                yield _emotion_record(text, label, "brighter-hindi-emotion-categories", "hi")


_INDICTALK_SCRIPT: dict[str, Lang] = {"Native": "hi", "Romanized": "hinglish"}


def read_indictalk_hindi(root: Path) -> Iterator[Record]:
    """LingoIITGN/IndicTalk, `hindi/` only — 146k eight-turn workplace
    chats in Devanagari (`Native`) or Roman (`Romanized`) script. Both
    are code-mixed; the script is what the corpus records, so `hi` means
    Devanagari here. Single-speaker rows exist and are not a dialogue.
    """
    for table in _tables(root / "hindi"):
        for row in _rows(table):
            lang = _INDICTALK_SCRIPT.get(row.get("Script", ""))
            turns = row.get("Conversation")
            if lang is None or not isinstance(turns, list):
                continue
            speakers = {t.get("speaker") for t in turns}
            if len(speakers) != 2:
                continue
            messages = _dialogue((t.get("speaker", ""), t.get("text", "")) for t in turns)
            if messages is None:
                continue
            yield Record(
                messages=messages,
                tools=None,
                source="indictalk-hindi-config",
                licence="",
                lang=lang,
                kind="chat",
            )


def read_cmu_hinglish_dog(root: Path) -> Iterator[Record]:
    """festvox/cmu_hinglish_dog: crowdworkers chatting about films in real
    Roman Hinglish. The HF table is one utterance per row; a conversation
    is the run of rows sharing `(date, user2_id)`, in timestamp order.
    """
    for table in _tables(root / "data"):
        group: list[dict[str, Any]] = []
        key: tuple[Any, Any] | None = None
        for row in _rows(table):
            row_key = (row.get("date"), row.get("user2_id"))
            if key is not None and row_key != key:
                if (rec := _cmu_record(group)) is not None:
                    yield rec
                group = []
            key = row_key
            group.append(row)
        if group and (rec := _cmu_record(group)) is not None:
            yield rec


def _cmu_record(group: list[dict[str, Any]]) -> Record | None:
    ordered = sorted(group, key=lambda r: r.get("utcTimestamp") or "")
    messages = _dialogue(
        (r.get("uid", ""), (r.get("translation") or {}).get("hi_en", "")) for r in ordered
    )
    if messages is None:
        return None
    return Record(
        messages=messages,
        tools=None,
        source="cmu-hinglish-dog",
        licence="",
        lang="hinglish",
        kind="chat",
    )


def read_phinc(root: Path) -> Iterator[Record]:
    """LingoIITGN/PHINC: real user-written Hinglish social-media sentences
    with human English translations — the noisy spelling the LLM-made
    Hinglish sets never have.
    """
    for table in _tables(root):
        for row in _rows(table):
            hinglish, english = _clean(row.get("Sentence")), _clean(row.get("English_Translation"))
            if not hinglish or not english:
                continue
            yield Record(
                messages=[
                    {"role": "system", "content": COMPREHENSION_SYSTEM},
                    {"role": "user", "content": hinglish},
                    {"role": "assistant", "content": english},
                ],
                tools=None,
                source="phinc",
                licence="",
                lang="hinglish",
                kind="chat",
            )


READERS: dict[str, Callable[[Path], Iterator[Record]]] = {
    "databricks-dolly-15k-hinglish": read_dolly_hinglish,
    "openassistant-oasst2": read_oasst2,
    "hinglish-top": read_hinglish_top,
    "synthetic-persona-chat": read_synthetic_persona_chat,
    "xlam-function-calling-60k": read_xlam,
    "hermes-function-calling-v1": read_hermes,
    "glaive-function-calling-v2": read_glaive,
    "when2call": read_when2call,
    "goemotions": read_goemotions,
    "brighter-hindi-emotion-categories": read_brighter_hindi,
    "indictalk-hindi-config": read_indictalk_hindi,
    "cmu-hinglish-dog": read_cmu_hinglish_dog,
    "phinc": read_phinc,
}


# --- licence, loading, splitting ------------------------------------------


def refusal_reason(ds: Dataset) -> str | None:
    """Why this source must not go into the adapter, or None.

    The manifest already encodes the decision: a flag in
    NON_PUBLISHABLE_FLAGS means weights trained on it cannot be
    released. One such corpus taints the whole LoRA, so it is refused
    here rather than included and remembered.
    """
    bad = sorted(set(ds.flags) & NON_PUBLISHABLE_FLAGS)
    if bad:
        return f"licence flags {bad} forbid training for release ({ds.license})"
    return None


def publishable(ds: Dataset) -> bool:
    return ds.weights_publishable == "yes"


def is_complete(root: Path) -> bool:
    return (root / COMPLETE_MARKER).exists()


@dataclass
class Plan:
    """Which manifest sources this run reads, and why the rest are not."""

    included: list[Dataset] = field(default_factory=list)
    refused: dict[str, str] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)  # in the manifest, no complete download
    unreadable: list[str] = field(default_factory=list)  # in the manifest, no reader yet


def plan(
    manifest: Manifest,
    *,
    datasets_dir: Path | None = None,
    only: list[str] | None = None,
    publishable_only: bool = False,
) -> Plan:
    base = datasets_dir or DATASETS_DIR
    out = Plan()
    names = {d.name for d in manifest.dataset}
    if only:
        unknown = sorted(set(only) - names)
        if unknown:
            raise KeyError(f"not in manifest: {unknown}")
    for ds in manifest.dataset:
        if only and ds.name not in only:
            continue
        if ds.name not in READERS:
            if only:
                out.unreadable.append(ds.name)
            continue
        reason = refusal_reason(ds)
        if reason is not None:
            if only:
                # Asked for by name: refusing quietly would look like a bug
                # in the reader. Refusing loudly is the point.
                raise LicenceRefused(f"{ds.name}: {reason}")
            out.refused[ds.name] = reason
            continue
        if publishable_only and not publishable(ds):
            out.refused[ds.name] = f"weights_publishable is {ds.weights_publishable!r}"
            continue
        if not is_complete(base / ds.name):
            out.missing.append(ds.name)
            continue
        out.included.append(ds)
    return out


def load(
    ds: Dataset, datasets_dir: Path | None = None, limit: int | None = None
) -> Iterator[Record]:
    """Records from one included source, licence stamped from the manifest."""
    base = datasets_dir or DATASETS_DIR
    reason = refusal_reason(ds)
    if reason is not None:
        raise LicenceRefused(f"{ds.name}: {reason}")
    for n, rec in enumerate(READERS[ds.name](base / ds.name), start=1):
        yield replace(rec, source=ds.name, licence=ds.license)
        if limit is not None and n >= limit:
            return


def bucket(canonical: str, seed: int = DEFAULT_SEED, buckets: int = SPLIT_BUCKETS) -> int:
    """Stable hash of a record's content. Python's `hash()` is salted per
    process; sha256 of the canonical JSON is the same on the laptop and
    on the box, which is what lets val stay val across runs.
    """
    h = hashlib.sha256(f"{seed}:{canonical}".encode()).hexdigest()
    return int(h[:12], 16) % buckets


def partition(canonical: str, seed: int = DEFAULT_SEED, val_fraction: float = VAL_FRACTION) -> str:
    return "val" if bucket(canonical, seed) < int(val_fraction * SPLIT_BUCKETS) else "train"
