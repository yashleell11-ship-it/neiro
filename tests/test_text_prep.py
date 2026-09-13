"""The text-corpus readers, the seeded split, and the licence gate.

Every reader is exercised on a three-row synthetic fixture in tmp_path
written in a container the reader also accepts (jsonl / csv / json) —
never the real download, and never parquet, so the whole mapping runs
in the runtime venv without pyarrow. What is asserted is the *shape*
that comes out: roles alternate, tool calls carry ids the tool results
answer, Glaive's Python-quoted arguments become JSON, an emoji never
reaches a spoken turn. Each assertion fails on a reader that guesses.
"""

from __future__ import annotations

import csv
import json
import sys
from itertools import pairwise
from pathlib import Path

import pytest

from neiro.training import text as prep
from neiro.training.manifest import Manifest
from neiro.training.text import (
    COMPREHENSION_SYSTEM,
    PERSONA_SYSTEM_PREFIX,
    READERS,
    TAG_LABELS,
    LicenceRefused,
    Record,
    bucket,
    load,
    partition,
    plan,
    read_brighter_hindi,
    read_cmu_hinglish_dog,
    read_dolly_hinglish,
    read_glaive,
    read_goemotions,
    read_hermes,
    read_hinglish_top,
    read_indictalk_hindi,
    read_oasst2,
    read_phinc,
    read_synthetic_persona_chat,
    read_when2call,
    read_xlam,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prep_text


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return path


def _json(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False))
    return path


def _csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def _roles(rec: Record) -> list[str]:
    return [m["role"] for m in rec.messages]


def _assert_alternates(rec: Record) -> None:
    turns = [m for m in rec.messages if m["role"] in ("user", "assistant")]
    for a, b in pairwise(turns):
        assert a["role"] != b["role"], "two consecutive turns by the same side"
    assert turns[0]["role"] == "user" and turns[-1]["role"] == "assistant"


# --- plain chat corpora -------------------------------------------------


class TestDollyHinglish:
    def test_instruction_context_and_output_become_one_exchange(self, tmp_path: Path) -> None:
        _jsonl(
            tmp_path / "data" / "train.jsonl",
            [
                {
                    "codemix_instruction": "Yeh kya hai?",
                    "codemix_input": "Ek context.",
                    "codemix_output": "Yeh ek test hai.",
                },
                {"codemix_instruction": "Bina context?", "codemix_input": None,
                 "codemix_output": "Haan."},
                {"codemix_instruction": "Output nahi", "codemix_input": None,
                 "codemix_output": ""},
            ],
        )  # fmt: skip
        recs = list(read_dolly_hinglish(tmp_path))
        assert len(recs) == 2, "a row with no output is dropped, not emitted empty"
        assert recs[0].messages[0]["content"] == "Yeh kya hai?\n\nEk context."
        assert recs[1].messages[0]["content"] == "Bina context?"
        assert {r.lang for r in recs} == {"hinglish"} and {r.kind for r in recs} == {"chat"}
        for r in recs:
            _assert_alternates(r)


def _oasst_node(text: str, role: str, rank: int | None = None, **extra: object) -> dict:
    return {"text": text, "role": role, "lang": "en", "rank": rank, "replies": [], **extra}


class TestOasst2:
    def test_follows_the_rank_zero_reply_and_the_sole_unranked_one(self, tmp_path: Path) -> None:
        follow_up = _oasst_node("and then?", "prompter")
        follow_up["replies"] = [_oasst_node("then this", "assistant", rank=None)]
        best = _oasst_node("best answer", "assistant", rank=0)
        best["replies"] = [follow_up]
        root = _oasst_node("question", "prompter")
        root["replies"] = [_oasst_node("worse answer", "assistant", rank=1), best]
        spanish = _oasst_node("pregunta", "prompter", lang="es")
        spanish["replies"] = [_oasst_node("respuesta", "assistant", rank=0)]
        gone = _oasst_node("q", "prompter")
        gone["replies"] = [_oasst_node("deleted", "assistant", rank=0, deleted=True)]
        trees = [{"prompt": root}, {"prompt": spanish}, {"prompt": gone}]
        _jsonl(tmp_path / "2023_oasst2_ready.trees.jsonl", trees)
        # The superset file must be ignored or every path is emitted twice.
        _jsonl(tmp_path / "2023_oasst2_all.trees.jsonl", trees)

        recs = list(read_oasst2(tmp_path))
        assert len(recs) == 1
        assert [m["content"] for m in recs[0].messages] == [
            "question", "best answer", "and then?", "then this"
        ]  # fmt: skip
        assert recs[0].lang == "en"

    def test_a_dangling_prompter_turn_is_trimmed(self, tmp_path: Path) -> None:
        # A user turn nobody answered teaches nothing and breaks
        # alternation; the path ends on the last assistant turn.
        best = _oasst_node("answer", "assistant", rank=0)
        best["replies"] = [_oasst_node("unanswered", "prompter")]
        root = _oasst_node("question", "prompter")
        root["replies"] = [best]
        _jsonl(tmp_path / "x_ready.trees.jsonl", [{"prompt": root}])
        recs = list(read_oasst2(tmp_path))
        assert len(recs) == 1 and _roles(recs[0]) == ["user", "assistant"]


class TestHinglishTop:
    def test_only_human_rows_and_the_fixed_comprehension_frame(self, tmp_path: Path) -> None:
        _jsonl(
            tmp_path / "data" / "val.jsonl",
            [
                {"en": "Pause my timer.", "hi_en": "Mere timer ko roko.", "generated_by": "human"},
                {"en": "Set alarm.", "hi_en": "Alarm lagao.", "generated_by": "synthetic"},
                {"en": "", "hi_en": "Kuch bhi.", "generated_by": "human"},
            ],
        )
        recs = list(read_hinglish_top(tmp_path))
        assert len(recs) == 1
        assert recs[0].messages[0] == {"role": "system", "content": COMPREHENSION_SYSTEM}
        assert recs[0].messages[1]["content"] == "Mere timer ko roko."
        assert recs[0].messages[2]["content"] == "Pause my timer."
        assert recs[0].lang == "hinglish"


class TestSyntheticPersonaChat:
    def test_user_two_is_her_and_her_persona_is_the_system_prompt(self, tmp_path: Path) -> None:
        _csv(
            tmp_path / "data" / "Synthetic-Persona-Chat_valid.csv",
            [
                {
                    "user 1 personas": "I bake.\nI have a dog.",
                    "user 2 personas": "I knit. 🧶\nI run marathons.",
                    "Best Generated Conversation": (
                        "User 1: Hi! 😀\nUser 2: Hello!\nUser 1: What do you do?\n"
                        "for fun\nUser 2: I knit.\nUser 1: Nice."
                    ),
                },
                {"user 1 personas": "x", "user 2 personas": "y", "Best Generated Conversation": ""},
                {
                    "user 1 personas": "x",
                    "user 2 personas": "y",
                    "Best Generated Conversation": "User 1: Hello?\nUser 1: Anyone?",
                },
            ],
        )
        recs = list(read_synthetic_persona_chat(tmp_path))
        assert len(recs) == 1, "an empty transcript and a monologue are not conversations"
        rec = recs[0]
        assert rec.messages[0]["role"] == "system"
        assert rec.messages[0]["content"] == PERSONA_SYSTEM_PREFIX + "- I knit.\n- I run marathons."
        assert [m["content"] for m in rec.messages[1:]] == [
            "Hi!", "Hello!", "What do you do? for fun", "I knit."
        ]  # fmt: skip
        _assert_alternates(rec)
        assert "😀" not in json.dumps(rec.as_dict(), ensure_ascii=False)


class TestIndicTalk:
    def test_script_sets_lang_and_spoken_text_is_cleaned(self, tmp_path: Path) -> None:
        _jsonl(
            tmp_path / "hindi" / "hindi.jsonl",
            [
                {
                    "Persona": "colleagues",
                    "Script": "Native",
                    "Conversation": [
                        {
                            "speaker": "Mentor",
                            "text": "यह **case** देख कर 😟  \nमुझे uneasy लग रहा है।",
                        },
                        {"speaker": "Mentee", "text": "हाँ, सही कहा।"},
                        {"speaker": "Mentor", "text": "तो क्या करें?"},
                        {"speaker": "Mentee", "text": "Monitor करते हैं।"},
                    ],
                },
                {
                    "Persona": "colleagues",
                    "Script": "Romanized",
                    "Conversation": [
                        {"speaker": "A", "text": "Kya haal hai?"},
                        {"speaker": "B", "text": "Sab theek."},
                    ],
                },
                {
                    "Persona": "experts",
                    "Script": "Native",
                    "Conversation": [{"speaker": "Solo", "text": "अकेला"}] * 4,
                },
            ],
        )
        recs = list(read_indictalk_hindi(tmp_path))
        assert len(recs) == 2, "a single-speaker row is not a dialogue"
        native, roman = recs
        assert native.lang == "hi" and roman.lang == "hinglish"
        assert native.messages[0]["content"] == "यह case देख कर मुझे uneasy लग रहा है।"
        assert len(native.messages) == 4
        _assert_alternates(native)


class TestCmuHinglishDog:
    def test_rows_regroup_into_conversations_in_hinglish(self, tmp_path: Path) -> None:
        def row(date: str, uid: str, ts: str, en: str, hi_en: str) -> dict:
            return {
                "date": date,
                "user2_id": "USR1",
                "uid": uid,
                "utcTimestamp": ts,
                "translation": {"en": en, "hi_en": hi_en},
            }

        _jsonl(
            tmp_path / "data" / "validation.jsonl",
            [
                row("d1", "user2", "t2", "What is it about?", "movie kis baare me hai?"),
                row("d1", "user1", "t3", "A musical.", "musical hai."),
                row("d1", "user2", "t1", "Hi!", "Hi!"),
                row("d1", "user2", "t4", "Ok", "theek"),
                row("d2", "user1", "t1", "Hey", "Hey"),
                row("d2", "user2", "t2", "Hello", "Hello ji"),
            ],
        )
        recs = list(read_cmu_hinglish_dog(tmp_path))
        assert len(recs) == 2, "a new date is a new conversation"
        first = recs[0]
        assert first.lang == "hinglish"
        assert [m["content"] for m in first.messages] == [
            "Hi! movie kis baare me hai?", "musical hai."
        ], "timestamp order, same-speaker merge, trailing user turn trimmed"  # fmt: skip
        assert "What is it about?" not in json.dumps(first.as_dict()), (
            "the English side is not used"
        )
        _assert_alternates(recs[1])


class TestPhinc:
    def test_hinglish_sentence_to_english_translation(self, tmp_path: Path) -> None:
        _csv(
            tmp_path / "PHINC.csv",
            [
                {"Sentence": "uske liye bahot kuch karna padega", "English_Translation": "a lot"},
                {"Sentence": "khali", "English_Translation": ""},
                {"Sentence": "aaj mausam acha hai", "English_Translation": "weather is nice today"},
            ],
        )
        recs = list(read_phinc(tmp_path))
        assert [r.messages[2]["content"] for r in recs] == ["a lot", "weather is nice today"]
        assert all(r.messages[0]["content"] == COMPREHENSION_SYSTEM for r in recs)
        assert {r.lang for r in recs} == {"hinglish"}


# --- tool-calling corpora -----------------------------------------------


def _tool_names(rec: Record) -> list[str]:
    return [t["function"]["name"] for t in rec.tools or []]


class TestXlam:
    def test_calls_and_schema_arrive_in_the_openai_shape(self, tmp_path: Path) -> None:
        tools = [
            {
                "name": "giveaways",
                "description": "Live giveaways.",
                "parameters": {
                    "type": {"description": "kind", "type": "str, optional"},
                    "count": {"description": "how many", "type": "int"},
                    "platform": {"description": "where", "type": "str", "default": "pc"},
                    "ids": {"description": "ids", "type": "List[int]"},
                },
            }
        ]
        _json(
            tmp_path / "xlam_function_calling_60k.json",
            [
                {
                    "id": 0,
                    "query": "Beta and game giveaways?",
                    "answers": json.dumps(
                        [
                            {"name": "giveaways", "arguments": {"type": "beta", "count": 2}},
                            {"name": "giveaways", "arguments": {"type": "game", "count": 2}},
                        ]
                    ),
                    "tools": json.dumps(tools),
                },
                {"id": 1, "query": "broken", "answers": "[]", "tools": "not json"},
                {"id": 2, "query": "", "answers": "[]", "tools": json.dumps(tools)},
            ],
        )
        recs = list(read_xlam(tmp_path))
        assert len(recs) == 1
        rec = recs[0]
        assert rec.kind == "tool_call" and _roles(rec) == ["user", "assistant"]
        calls = rec.messages[1]["tool_calls"]
        assert [c["id"] for c in calls] == ["call_0", "call_1"]
        assert all(c["type"] == "function" for c in calls)
        assert json.loads(calls[0]["function"]["arguments"]) == {"type": "beta", "count": 2}
        assert rec.messages[1]["content"] is None
        params = rec.tools[0]["function"]["parameters"]
        assert params["type"] == "object"
        assert params["properties"]["type"]["type"] == "string"
        assert params["properties"]["count"]["type"] == "integer"
        assert params["properties"]["ids"]["type"] == "array"
        assert params["properties"]["platform"]["default"] == "pc"
        assert params["required"] == ["count", "ids"], (
            "optional and defaulted params are not required"
        )


def _hermes_system(tools: list[dict]) -> str:
    return "You are a function calling AI model. <tools>" + json.dumps(tools) + "</tools>"


class TestHermes:
    def test_tool_calls_and_responses_pair_up_by_position(self, tmp_path: Path) -> None:
        tools = [
            {"type": "function", "function": {"name": "feed", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "record", "parameters": {"type": "object"}}},
        ]
        conv = [
            {"from": "system", "value": _hermes_system(tools)},
            {"from": "human", "value": "Show the front door and record it."},
            {
                "from": "gpt",
                "value": (
                    'Sure.\n<tool_call>\n{"name": "feed", "arguments": {"camera_id": "front"}}\n'
                    '</tool_call>\n<tool_call>\n{"name": "record", "arguments": {"minutes": 30}}\n'
                    "</tool_call>\n"
                ),
            },
            {
                "from": "tool",
                "value": (
                    '<tool_response>\n{"name": "feed", "content": {"url": "https://x/1"}}\n'
                    '</tool_response>\n<tool_response>\n{"name": "record", "content": "started"}\n'
                    "</tool_response>"
                ),
            },
            {"from": "gpt", "value": "Live feed is at https://x/1 and recording started."},
        ]
        json_mode = [
            {"from": "system", "value": "Answer in JSON per <schema>{...}</schema>"},
            {"from": "human", "value": "Give me a patent record."},
            {"from": "gpt", "value": '{"patentID": "1"}'},
        ]
        orphan = [
            {"from": "human", "value": "hi"},
            {
                "from": "tool",
                "value": '<tool_response>\n{"name": "x", "content": 1}\n</tool_response>',
            },
            {"from": "gpt", "value": "ok"},
        ]
        no_access = [
            {
                "from": "system",
                "value": "You are a helpful assistant, with no access to functions.",
            },
            {"from": "human", "value": "Book me a flight."},
            {"from": "gpt", "value": "I can't book flights."},
        ]
        _json(
            tmp_path / "func-calling.json",
            [
                {"id": "a", "conversations": conv, "tools": json.dumps(tools)},
                {"id": "b", "conversations": json_mode, "schema": "{}"},
                {"id": "c", "conversations": orphan, "tools": json.dumps(tools)},
                # The Glaive subset writes the *string* "null" for no tools.
                {"id": "d", "conversations": no_access, "tools": "null"},
            ],
        )
        recs = list(read_hermes(tmp_path))
        assert len(recs) == 3, "a tool result with no call to answer is malformed and dropped"
        agentic, plain, negative = recs
        assert negative.tools is None and negative.kind == "chat"
        assert _roles(negative) == ["system", "user", "assistant"]
        assert _roles(agentic) == ["user", "assistant", "tool", "tool", "assistant"]
        assert agentic.kind == "tool_call" and _tool_names(agentic) == ["feed", "record"]
        assistant = agentic.messages[1]
        assert assistant["content"] == "Sure."
        assert [c["id"] for c in assistant["tool_calls"]] == ["call_0", "call_1"]
        assert json.loads(assistant["tool_calls"][1]["function"]["arguments"]) == {"minutes": 30}
        assert [m["tool_call_id"] for m in agentic.messages[2:4]] == ["call_0", "call_1"]
        assert json.loads(agentic.messages[2]["content"]) == {"url": "https://x/1"}
        assert agentic.messages[3]["content"] == "started"

        assert plain.kind == "chat" and plain.tools is None
        assert _roles(plain) == ["system", "user", "assistant"], "JSON-mode keeps its system turn"


class TestGlaive:
    def test_python_quoted_arguments_and_concatenated_functions(self, tmp_path: Path) -> None:
        fn1 = json.dumps(
            {
                "name": "get_news",
                "description": "Headlines",
                "parameters": {"type": "object", "properties": {"country": {"type": "string"}}},
            },
            indent=4,
        )
        fn2 = json.dumps({"name": "get_time", "description": "Clock", "parameters": {}}, indent=4)
        system = "SYSTEM: You are a helpful assistant with access to the following functions. "
        system += f"Use them if required -\n{fn1}\n\n{fn2}\n"
        chat = (
            "USER: News for France?\n\n\n"
            'ASSISTANT: <functioncall> {"name": "get_news", "arguments": '
            '\'{"country": "France"}\'} <|endoftext|>\n\n\n'
            'FUNCTION RESPONSE: {"headlines": ["a", "b"]}\n\n\n'
            "ASSISTANT: Here are the headlines: a and b. <|endoftext|>\n\n\n"
            "USER: thanks\n\n\n"
        )
        no_tools = "SYSTEM: You are a helpful assistant, with no access to external functions.\n"
        _json(
            tmp_path / "glaive-function-calling-v2.json",
            [
                {"system": system, "chat": chat},
                {"system": no_tools, "chat": "USER: hi\n\n\nASSISTANT: hello <|endoftext|>\n\n\n"},
                {
                    "system": system,
                    "chat": 'FUNCTION RESPONSE: {"x": 1}\n\n\nASSISTANT: ok <|endoftext|>',
                },
            ],
        )
        recs = list(read_glaive(tmp_path))
        assert len(recs) == 2, "a function response before any call is malformed"
        with_tools, without = recs
        assert _tool_names(with_tools) == ["get_news", "get_time"]
        assert all(t["type"] == "function" for t in with_tools.tools)
        assert with_tools.kind == "tool_call"
        assert _roles(with_tools) == ["user", "assistant", "tool", "assistant"]
        call = with_tools.messages[1]["tool_calls"][0]
        assert call["function"]["name"] == "get_news"
        assert json.loads(call["function"]["arguments"]) == {"country": "France"}
        assert with_tools.messages[1]["content"] is None
        assert with_tools.messages[2]["tool_call_id"] == call["id"]
        assert with_tools.messages[3]["content"] == "Here are the headlines: a and b."
        assert "<|endoftext|>" not in json.dumps(with_tools.as_dict())

        assert without.tools is None and without.kind == "chat"
        assert _roles(without) == ["user", "assistant"]


class TestWhen2Call:
    def test_sft_and_the_chosen_side_of_pref(self, tmp_path: Path) -> None:
        tool = json.dumps(
            {
                "name": "get_ico",
                "description": "ICO calendar",
                "parameters": {
                    "type": "dict",
                    "properties": {"tab": {"type": "str", "description": "which"}},
                    "required": ["tab"],
                },
            }
        )
        _jsonl(
            tmp_path / "train" / "when2call_train_sft.jsonl",
            [
                {
                    "tools": [tool],
                    "messages": [
                        {"role": "user", "content": "Trending topics?"},
                        {"role": "assistant", "content": "I can't fetch real-time trends."},
                    ],
                },
                {
                    "tools": [tool],
                    "messages": [
                        {"role": "user", "content": "Show ICOs"},
                        {"role": "assistant", "content": "Which tab: upcoming or completed?"},
                    ],
                },
            ],
        )
        _jsonl(
            tmp_path / "train" / "when2call_train_pref.jsonl",
            [
                {
                    "tools": [tool],
                    "messages": [{"role": "user", "content": "Completed ICOs please."}],
                    "chosen_response": {
                        "role": "assistant",
                        "content": '<TOOLCALL>[{"name": "get_ico", "arguments": {"tab": "completed"}}]</TOOLCALL>',
                    },
                    "rejected_response": {"role": "assistant", "content": "Which language?"},
                }
            ],
        )
        recs = list(read_when2call(tmp_path))
        assert len(recs) == 3
        assert all(r.kind == "tool_call" and _tool_names(r) == ["get_ico"] for r in recs)
        params = recs[0].tools[0]["function"]["parameters"]
        assert params["type"] == "object" and params["properties"]["tab"]["type"] == "string"
        # Files are read in name order (pref before sft), so pick by shape.
        in_words = [r for r in recs if "tool_calls" not in r.messages[-1]]
        called = [r for r in recs if "tool_calls" in r.messages[-1]]
        assert len(in_words) == 2 and len(called) == 1, "SFT answers in words; pref's chosen calls"
        chosen = called[0].messages[-1]
        assert json.loads(chosen["tool_calls"][0]["function"]["arguments"]) == {"tab": "completed"}
        assert "Which language?" not in json.dumps(called[0].as_dict()), (
            "the rejected side is not data"
        )


# --- emotion text -------------------------------------------------------


class TestGoEmotions:
    def test_labels_collapse_onto_her_tag_or_the_row_is_dropped(self, tmp_path: Path) -> None:
        names = prep.GOEMOTIONS_NAMES
        _jsonl(
            tmp_path / "simplified" / "train.jsonl",
            [
                {"text": "That worked!", "labels": [names.index("joy")], "id": "a"},
                {
                    "text": "Ok.",
                    "labels": [names.index("approval"), names.index("neutral")],
                    "id": "b",
                },
                {"text": "Scary.", "labels": [names.index("fear")], "id": "c"},
                {
                    "text": "Mixed.",
                    "labels": [names.index("joy"), names.index("sadness")],
                    "id": "d",
                },
            ],
        )
        recs = list(read_goemotions(tmp_path))
        assert [(r.messages[0]["content"], r.messages[1]["content"]) for r in recs] == [
            ("That worked!", "happy"),
            ("Ok.", "neutral"),
        ], "fear has no home on the tag and joy+sadness is ambiguous: both dropped"
        assert all(r.kind == "emotion_text" and r.lang == "en" for r in recs)
        assert all(r.messages[1]["content"] in TAG_LABELS for r in recs)


class TestBrighterHindi:
    def test_hindi_labels_and_the_empty_list_is_neutral(self, tmp_path: Path) -> None:
        _jsonl(
            tmp_path / "hin" / "train.jsonl",
            [
                {"text": "वह मूवी देखने गई थी।", "emotions": []},
                {"text": "मुझे तुम्हारी असलियत पता चल गयी है।", "emotions": ["anger"]},
                {"text": "डर लग रहा है।", "emotions": ["fear"]},
            ],
        )
        recs = list(read_brighter_hindi(tmp_path))
        assert [r.messages[1]["content"] for r in recs] == ["neutral", "angry"]
        assert all(r.lang == "hi" and r.kind == "emotion_text" for r in recs)


class TestRegistry:
    def test_every_manifest_text_corpus_has_a_reader(self) -> None:
        manifest = Manifest.load(Path(__file__).resolve().parents[1] / "data" / "datasets.toml")
        text_targets = {"persona_lora", "hinglish_llm", "hindi_emotion_text"}
        in_manifest = {d.name for d in manifest.dataset if d.target in text_targets}
        assert set(READERS) <= in_manifest, "a reader for a corpus the manifest does not list"
        assert len(READERS) == 13


# --- split ----------------------------------------------------------------


def _record(i: int, **over: object) -> Record:
    base: dict[str, object] = {
        "messages": [
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": f"a{i}"},
        ],
        "tools": None,
        "source": "s",
        "licence": "MIT",
        "lang": "en",
        "kind": "chat",
    }
    return Record(**(base | over))  # type: ignore[arg-type]


class TestSplit:
    def test_the_same_record_lands_on_the_same_side_every_time(self) -> None:
        canon = _record(1).canonical()
        assert bucket(canon) == bucket(canon)
        assert [partition(_record(i).canonical()) for i in range(200)] == [
            partition(_record(i).canonical()) for i in range(200)
        ]

    def test_the_seed_moves_the_line(self) -> None:
        a = [partition(_record(i).canonical(), seed=0) for i in range(500)]
        b = [partition(_record(i).canonical(), seed=1) for i in range(500)]
        assert a != b

    def test_the_fraction_is_roughly_honoured(self) -> None:
        sides = [partition(_record(i).canonical(), val_fraction=0.2) for i in range(5000)]
        share = sides.count("val") / len(sides)
        assert 0.16 < share < 0.24

    def test_licence_and_source_do_not_move_a_record(self) -> None:
        # Rewording a manifest entry must not shuffle val into train; and
        # the same exchange from two corpora must dedupe to one line.
        a = _record(7, licence="MIT", source="x").canonical()
        b = _record(7, licence="cc-by-4.0", source="y").canonical()
        assert a == b and partition(a) == partition(b)

    def test_content_does(self) -> None:
        assert _record(7).canonical() != _record(8).canonical()


# --- licence gate and the script -----------------------------------------


_MANIFEST = """
schema_version = 1

[[dataset]]
name = "phinc"
target = "hinglish_llm"
priority = 1
hf_id = "org/phinc"
license = "cc-by-4.0"
flags = ["permissive"]
access = "direct"
size_gb = 0.0
weights_publishable = "yes"

[[dataset]]
name = "goemotions"
target = "persona_lora"
priority = 1
hf_id = "org/goemotions"
license = "CC BY-NC 4.0"
flags = ["NC"]
access = "direct"
size_gb = 0.0
weights_publishable = "no"

[[dataset]]
name = "cmu-hinglish-dog"
target = "hinglish_llm"
priority = 1
hf_id = "org/dog"
license = "cc-by-sa-3.0"
flags = ["SA"]
access = "direct"
size_gb = 0.0
weights_publishable = "unclear"

[[dataset]]
name = "brighter-hindi-emotion-categories"
target = "hindi_emotion_text"
priority = 1
hf_id = "org/brighter"
license = "cc-by-4.0"
flags = ["permissive"]
access = "direct"
size_gb = 0.0
weights_publishable = "yes"

[[dataset]]
name = "crema-d"
target = "ser_lane_b"
priority = 1
hf_id = "org/crema"
license = "ODbL"
flags = ["permissive"]
access = "direct"
size_gb = 1.0
weights_publishable = "yes"
"""


def _downloads(tmp_path: Path) -> Path:
    """phinc, goemotions and cmu-hinglish-dog complete; brighter never fetched."""
    base = tmp_path / "datasets"
    _csv(
        base / "phinc" / "PHINC.csv",
        [
            {"Sentence": "kya haal", "English_Translation": "how are you"},
            {"Sentence": "kya haal", "English_Translation": "how are you"},
            {"Sentence": "theek", "English_Translation": "fine"},
        ],
    )
    (base / "phinc" / prep.COMPLETE_MARKER).write_text("{}")
    _jsonl(
        base / "goemotions" / "simplified" / "train.jsonl",
        [{"text": "yay", "labels": [prep.GOEMOTIONS_NAMES.index("joy")], "id": "a"}],
    )
    (base / "goemotions" / prep.COMPLETE_MARKER).write_text("{}")
    _jsonl(
        base / "cmu-hinglish-dog" / "data" / "train.jsonl",
        [
            {"date": "d", "user2_id": "u", "uid": "user1", "utcTimestamp": "1",
             "translation": {"en": "hi", "hi_en": "hi"}},
            {"date": "d", "user2_id": "u", "uid": "user2", "utcTimestamp": "2",
             "translation": {"en": "hello", "hi_en": "hello ji"}},
        ],
    )  # fmt: skip
    (base / "cmu-hinglish-dog" / prep.COMPLETE_MARKER).write_text("{}")
    return base


class TestLicenceGate:
    def test_a_non_publishable_source_is_refused_with_a_reason(self, tmp_path: Path) -> None:
        base = _downloads(tmp_path)
        p = plan(Manifest.loads(_MANIFEST), datasets_dir=base)
        assert [d.name for d in p.included] == ["phinc", "cmu-hinglish-dog"]
        assert "goemotions" in p.refused and "NC" in p.refused["goemotions"]
        assert p.missing == ["brighter-hindi-emotion-categories"], "in the manifest, not on disk"

    def test_asking_for_a_refused_source_by_name_raises(self, tmp_path: Path) -> None:
        base = _downloads(tmp_path)
        manifest = Manifest.loads(_MANIFEST)
        with pytest.raises(LicenceRefused):
            plan(manifest, datasets_dir=base, only=["goemotions"])
        nc = next(d for d in manifest.dataset if d.name == "goemotions")
        with pytest.raises(LicenceRefused):
            list(load(nc, base))

    def test_publishable_only_also_drops_the_unclear_ones(self, tmp_path: Path) -> None:
        base = _downloads(tmp_path)
        p = plan(Manifest.loads(_MANIFEST), datasets_dir=base, publishable_only=True)
        assert [d.name for d in p.included] == ["phinc"]
        assert "cmu-hinglish-dog" in p.refused

    def test_an_unknown_name_is_an_error_not_a_silent_skip(self, tmp_path: Path) -> None:
        with pytest.raises(KeyError):
            plan(Manifest.loads(_MANIFEST), datasets_dir=tmp_path, only=["no-such-corpus"])

    def test_load_stamps_the_manifest_licence(self, tmp_path: Path) -> None:
        base = _downloads(tmp_path)
        ds = next(d for d in Manifest.loads(_MANIFEST).dataset if d.name == "phinc")
        recs = list(load(ds, base, limit=1))
        assert len(recs) == 1 and recs[0].licence == "cc-by-4.0" and recs[0].source == "phinc"


class TestScript:
    def test_writes_split_files_and_stats(self, tmp_path: Path) -> None:
        base = _downloads(tmp_path)
        manifest = tmp_path / "datasets.toml"
        manifest.write_text(_MANIFEST)
        out = tmp_path / "prepared" / "text"
        rc = prep_text.main(
            ["--manifest", str(manifest), "--datasets-dir", str(base), "--out", str(out)]
        )
        assert rc == 0
        rows = [
            json.loads(line)
            for name in ("train.jsonl", "val.jsonl")
            for line in (out / name).read_text().splitlines()
        ]
        assert len(rows) == 3, "two phinc rows are identical and are written once"
        assert {r["source"] for r in rows} == {"phinc", "cmu-hinglish-dog"}
        assert all(
            set(r) == {"messages", "tools", "source", "licence", "lang", "kind"} for r in rows
        )

        stats = json.loads((out / "stats.json").read_text())
        assert stats["sources"]["phinc"]["n"] == 2
        assert stats["sources"]["phinc"]["duplicates"] == 1
        assert stats["sources"]["phinc"]["publishable"] is True
        assert stats["sources"]["cmu-hinglish-dog"]["publishable"] is False
        assert stats["sources"]["phinc"]["lang"] == {"hinglish": 2}
        assert "goemotions" in stats["refused"]
        assert stats["missing"] == ["brighter-hindi-emotion-categories"]
        assert stats["totals"]["n"] == 3
        assert stats["publishable"] is False, "one unclear source makes the adapter unpublishable"

    def test_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        base = _downloads(tmp_path)
        manifest = tmp_path / "datasets.toml"
        manifest.write_text(_MANIFEST)
        out = tmp_path / "prepared" / "text"
        rc = prep_text.main(
            [
                "--manifest",
                str(manifest),
                "--datasets-dir",
                str(base),
                "--out",
                str(out),
                "--dry-run",
            ]
        )
        assert rc == 0 and not out.exists()

    def test_a_refused_name_on_the_command_line_fails_loudly(self, tmp_path: Path) -> None:
        base = _downloads(tmp_path)
        manifest = tmp_path / "datasets.toml"
        manifest.write_text(_MANIFEST)
        out = tmp_path / "prepared" / "text"
        rc = prep_text.main(
            ["--manifest", str(manifest), "--datasets-dir", str(base), "--out", str(out),
             "--only", "goemotions"]
        )  # fmt: skip
        assert rc == 2 and not out.exists()
