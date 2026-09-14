# Training the persona LoRA: the text data

One adapter, three manifest targets. `persona_lora` is her character,
the `<e:LABEL:D>` tag and tool-calling; `hinglish_llm` is code-switching
in his register; `hindi_emotion_text` is emotional Hindi text. Every
corpus below is in `data/datasets.toml` with its licence, and
`scripts/prep_text.py` turns all of them into one file pair the recipe
on the 3090 Ti reads without knowing any corpus's layout. The LoRA
recipe itself is not written yet; this page is what it will be fed.

## What each target's data is

| target | source | what it teaches | lang | kind | licence | weights publishable |
|---|---|---|---|---|---|---|
| persona_lora | synthetic-persona-chat | stay in a stated persona across a chat (User 2 is her; her persona lines are the system prompt) | en | chat | CC-BY-4.0 | yes |
| persona_lora | openassistant-oasst2 | general chat — the anchor that stops the adapter collapsing into tool-only or persona-only replies; rank-0 paths through the `ready` trees, English only (the corpus has no Hindi) | en | chat | Apache-2.0 | yes |
| persona_lora | goemotions | how a feeling reads in English text: Reddit comments → one of her six labels | en | emotion_text | Apache-2.0 | yes |
| persona_lora | xlam-function-calling-60k | exact argument typing, parallel calls (one query → one or more calls) | en | tool_call | CC-BY-4.0 | yes |
| persona_lora | hermes-function-calling-v1 | multi-turn agentic loops: call → tool result → answer; plus JSON-mode rows (kind `chat`, schema in the system turn) | en | tool_call / chat | Apache-2.0 | yes |
| persona_lora | glaive-function-calling-v2 | tools offered and *not* called — the no-call negatives; `kind` is `tool_call` whenever tools are offered | en | tool_call / chat | Apache-2.0 | yes |
| persona_lora | when2call | when not to call: ask, refuse, or answer in words (SFT rows) and the chosen side of the preference rows | en | tool_call | CC-BY-4.0 | yes |
| hinglish_llm | databricks-dolly-15k-hinglish | instruction following in Roman Hinglish (the codemix side only) | hinglish | chat | CC-BY-SA-3.0 | unclear |
| hinglish_llm | hinglish-top | assistant-domain requests as he would say them (alarm, timer, weather…) → what was meant, in English; human rows only | hinglish | chat | Apache-2.0 | yes |
| hinglish_llm | cmu-hinglish-dog | real human Hinglish dialogue, rebuilt from per-utterance rows | hinglish | chat | CC-BY-SA-3.0 + GFDL | unclear |
| hinglish_llm | phinc | noisy real-world Hinglish → English, the spelling variation LLM-made Hinglish never has | hinglish | chat | CC-BY-4.0 | yes |
| hindi_emotion_text | brighter-hindi-emotion-categories | how a feeling reads in Devanagari: gold-labelled sentences → her six labels; `[]` is neutral | hi | emotion_text | CC-BY-4.0 | yes |
| hindi_emotion_text | indictalk-hindi-config | code-mixed workplace dialogue, Devanagari (`hi`) and Roman (`hinglish`) script | hi / hinglish | chat | CC-BY-4.0 | yes |

Two sources are `unclear` (ShareAlike): an adapter trained with them in
the mix is fine to *use* here and is flagged `publishable: false` in
`stats.json`. `--publishable-only` builds the set without them.

## The record

One JSON object per line, the same six keys for every corpus:

```json
{"messages": [{"role": "user", "content": "News for France?"},
              {"role": "assistant", "content": null,
               "tool_calls": [{"id": "call_0", "type": "function",
                               "function": {"name": "get_news", "arguments": "{\"country\": \"France\"}"}}]},
              {"role": "tool", "tool_call_id": "call_0", "content": "{\"headlines\": [\"...\"]}"},
              {"role": "assistant", "content": "Here are the headlines."}],
 "tools": [{"type": "function", "function": {"name": "get_news", "description": "...",
                                              "parameters": {"type": "object", "properties": {"...": {}}}}}],
 "source": "glaive-function-calling-v2", "licence": "apache-2.0", "lang": "en", "kind": "tool_call"}
```

- `messages` and `tools` are the OpenAI shapes, so `apply_chat_template`
  renders every corpus the same way. `arguments` is a JSON *string*, as
  the API has it; `tool_call_id`s are `call_N`, assigned per record.
- `tools[].function.parameters` is JSON schema. The corpora's Python
  spellings (`str`, `int, optional`, `List[int]`, `Tuple[float, float]`,
  `set`) are mapped to JSON types and an `optional` or defaulted
  parameter is not `required`; a spelling with no JSON type
  (`Callable[[float], float]`) is left as written, never truncated.
- `kind` is `chat`, `tool_call` (tools were offered, called or not), or
  `emotion_text`.
- `emotion_text` rows are `user: <the text>` → `assistant: <label>` with
  the label in her tag vocabulary (`happy angry sad relaxed surprised
  neutral`). **No corpus records an intensity, so no `D` digit is in the
  data.** The recipe renders these however it needs; it must not invent
  the digit either.
- Emoji are stripped everywhere (she is spoken aloud); markdown bold and
  line breaks are collapsed only in the corpora that are dialogue
  (IndicTalk, Persona-Chat, CMU-DoG), never in OASST2 or the tool rows.

## Running prep

The parquet corpora need pyarrow, which the runtime venv does not carry
on purpose, so prep runs from the training venv:

```sh
cd training && uv run python ../scripts/prep_text.py            # every complete download
cd training && uv run python ../scripts/prep_text.py --dry-run  # the plan only
cd training && uv run python ../scripts/prep_text.py --only goemotions --limit 200
cd training && uv run python ../scripts/prep_text.py --publishable-only
```

It reads every text corpus that has a `.neiro-complete` marker (from
`scripts/fetch_datasets.py`), writes `data/prepared/text/train.jsonl`,
`val.jsonl` and `stats.json`, and prints a table. The split is a seeded
sha256 of the conversation (5% val by default, `--seed`,
`--val-fraction`): a re-run, or a run on another machine, draws the same
line, and adding a corpus never moves an existing row across it. Exact
duplicates are written once. A source whose manifest flags forbid
training for release (`NC`, `ND`, `research-only`) is refused and listed
in `stats.json`, never included quietly; naming one with `--only` is an
error. A malformed row is dropped rather than guessed at, and a reader
that can name why it refused one counts it under that source's
`dropped` in `stats.json`, so a corpus losing rows shows up as a number
instead of as a gap in the arithmetic. `data/prepared/` is gitignored.

## What the box runs

On the 3090 Ti, from a fresh clone, in this order:

```sh
uv run scripts/fetch_datasets.py --tier 2 --target persona_lora
uv run scripts/fetch_datasets.py --tier 2 --target hinglish_llm
uv run scripts/fetch_datasets.py --tier 2 --target hindi_emotion_text
cd training && uv sync && uv run python ../scripts/prep_text.py
```

(`xlam-function-calling-60k` is gated with automatic approval: `uv run
hf auth login` once, first.) The LoRA recipe, when it exists, reads
`data/prepared/text/{train,val}.jsonl` and nothing else — and
`stats.json` is what says whether the adapter it produces may leave the
machine.

## The last real run (laptop, 2026-09-14)

`stats.json` from the run over all thirteen text corpora. 41 of the
manifest's 73 datasets have finished downloading; thirteen of those are
the ones with a reader here, and none of them is refused or missing, so
this run is the whole text side — 437,031 records, 1.19 GB of
`train.jsonl`, 62 MB of `val.jsonl`, 45.7 s of reader time summed over
sources (25.4 s of it IndicTalk's 780 MB). `dup` is exact duplicates
written once; it is where most of the gap between a corpus's advertised
size and its row count goes.

| source | n | train | val | dup | lang | kind | publishable |
|---|---:|---:|---:|---:|---|---|---|
| goemotions | 40266 | 38229 | 2037 | 215 | en 40266 | emotion_text 40266 | yes |
| synthetic-persona-chat | 21901 | 20760 | 1141 | 0 | en 21901 | chat 21901 | yes |
| glaive-function-calling-v2 | 103551 | 98345 | 5206 | 9231 | en 103551 | chat 34593, tool_call 68958 | yes |
| hermes-function-calling-v1 | 6394 | 6074 | 320 | 3432 | en 6394 | chat 3407, tool_call 2987 | yes |
| xlam-function-calling-60k | 59625 | 56579 | 3046 | 375 | en 59625 | tool_call 59625 | yes |
| when2call | 21269 | 20283 | 986 | 2731 | en 21269 | chat 2729, tool_call 18540 | yes |
| openassistant-oasst2 | 5666 | 5368 | 298 | 291 | en 5666 | chat 5666 | yes |
| cmu-hinglish-dog | 320 | 301 | 19 | 1 | hinglish 320 | chat 320 | no |
| hinglish-top | 7540 | 7182 | 358 | 6876 | hinglish 7540 | chat 7540 | yes |
| databricks-dolly-15k-hinglish | 14964 | 14226 | 738 | 44 | hinglish 14964 | chat 14964 | no |
| phinc | 13737 | 13055 | 682 | 1 | hinglish 13737 | chat 13737 | yes |
| brighter-hindi-emotion-categories | 2575 | 2449 | 126 | 801 | hi 2575 | emotion_text 2575 | yes |
| indictalk-hindi-config | 139223 | 132358 | 6865 | 0 | hi 96457, hinglish 42766 | chat 139223 | yes |
| **total** | **437031** | 415209 | 21822 | | en 258672, hi 99032, hinglish 79327 | chat 244080, emotion_text 42841, tool_call 150110 | no |

What the numbers say, so the recipe does not have to rediscover it:

- **English is 259k rows, Hindi + Hinglish 178k, and 139k of those are
  one LLM-generated corpus.** Hindi and English are equal priorities;
  the recipe must weight by `lang` (and cap IndicTalk), not sample rows
  uniformly. `by_lang` in `stats.json` is what to balance against.
- Hinglish-TOP's `test` split is its `train` split again (6,467 of
  6,513 pairs identical), and 3,382 of the 5,209 rows in Hermes' Glaive
  subset are byte-identical to Glaive-v2 rows — both land in `dup`,
  which is why the dedupe hashes conversation content and not source.
  (The other 50 Hermes dups: 41 inside `json-mode-agentic`, 7 shared by
  its two `func-calling` files, 2 inside the Glaive subset.)
- BRIGHTER keeps 2,575 of 4,776: `fear` and `disgust` (about 1,200
  rows, alone or combined) have no home on her tag and are dropped
  rather than filed as `sad`; a further ~200 carry two labels that map
  to different tags.
- GoEmotions keeps 40,266 of 54,263: 2,284 rows carry `fear`,
  `disgust`, `nervousness` or `embarrassment`, 1,791 carry labels that
  map to two different tags, and 9,707 carry *only* `approval`,
  `disapproval`, `caring`, `confusion`, `curiosity`, `desire` or
  `realization` — a feeling the raters did name and the tag cannot
  hold, so the row is dropped rather than filed as `neutral`. Every
  `neutral` GoEmotions contributes (16,798) is one the raters gave;
  BRIGHTER's 775 are its own empty-list convention.
- Dropped as malformed, never guessed. Hermes, 1,752 of 11,578 rows:
  1,586 whose `<tool_call>` body is Python-quoted behind a literal `\n`
  and never parses as a call — the same 793 prompts in each of
  `func-calling-singleturn.json` and `func-calling.json`, and the 122
  rows that offer `tools: []` are among them — plus 166 with more tool
  results than calls. Glaive, 178 of 112,960: 164 with a `FUNCTION
  RESPONSE` before any call, 9 whose call arguments do not parse, 3
  empty assistant turns, 2 empty user turns. No `<tool_call>` or
  `<tools>` tag reaches any message in the output, and no tool turn is
  empty — the 69 results the Glaive subset writes with no `content` key
  are the bare object, which is what the reply after them quotes.
- Synthetic-Persona-Chat keeps 21,901 of 21,907. 36 transcripts write
  the speaker marker in markdown (`**User 1:**`, `* * User 1: * *`,
  `* * User 1: ... * *`) and are read as the marker they are. The 6 that
  remain — 5 with no speaker marker on any line, 1 empty — are dropped,
  because which of the two people is talking is not something to
  invent, and they are counted under `dropped` in `stats.json` rather
  than lost in the arithmetic.
- Tool parameter `type`s come out as JSON-schema types except 653 left
  as written because there is none for them: 652 `Callable[[float],
  float]` (xLAM 584, When2Call 68) and one Glaive parameter whose own
  schema says `"type": {}`. `Tuple[float, float]` and `set` are arrays,
  and a spelling with no mapping is kept whole rather than truncated at
  its first comma. Glaive's schemas are passed through as the corpus
  wrote them, so one nested `"type": "any"` survives inside an `items`.
- `publishable: no` is entirely the two ShareAlike sources
  (Dolly-Hinglish, CMU-DoG); `--publishable-only` drops 15,284 rows
  and flips it.
