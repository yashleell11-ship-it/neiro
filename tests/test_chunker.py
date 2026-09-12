"""Tests for the sentence chunker.

Two things must hold, and they pull against each other: the FIRST chunk
should come out as early as possible (it gates time-to-first-audio on
every turn), and no chunk should ever split mid-word or mid-number
(which would make her mispronounce things or pause in odd places).
"""

from __future__ import annotations

from neiro.llm.chunker import SentenceChunker


def stream(chunker: SentenceChunker, text: str, chunk_size: int = 3) -> list[str]:
    """Feed `text` through in small fragments, like real token streaming."""
    out = []
    for i in range(0, len(text), chunk_size):
        out.extend(chunker.feed(text[i : i + chunk_size]))
    out.extend(chunker.flush())
    return out


class TestShortFirstClause:
    def test_first_chunk_breaks_at_a_comma(self) -> None:
        chunker = SentenceChunker()
        chunks = stream(chunker, "Hey there, you sound pretty tired today.")
        assert chunks[0] == "Hey there,"
        assert chunks[1] == "you sound pretty tired today."

    def test_first_chunk_breaks_at_sentence_end_when_thats_first(self) -> None:
        chunker = SentenceChunker()
        chunks = stream(chunker, "Sure thing. Let me look that up for you.")
        assert chunks[0] == "Sure thing."

    def test_later_chunks_do_not_split_on_commas(self) -> None:
        # Only the FIRST chunk gets the clause treatment — splitting on
        # every comma throughout would make her prosody choppy.
        chunker = SentenceChunker()
        chunks = stream(chunker, "Okay. I checked, and it looks fine, mostly.")
        assert chunks[0] == "Okay."
        assert chunks[1] == "I checked, and it looks fine, mostly."

    def test_comma_free_opening_does_not_hold_the_chunk_hostage(self) -> None:
        # No clause boundary at all — must still emit at the sentence end
        # rather than waiting forever.
        text = "I think the answer you are looking for is probably somewhere in the logs. Next."
        chunker = SentenceChunker()
        chunks = stream(chunker, text)
        assert chunks[0].endswith("logs.")

    def test_very_short_first_clause_is_not_emitted_alone(self) -> None:
        # "Hi," on its own reads as clipped; wait for something speakable.
        chunker = SentenceChunker()
        chunks = stream(chunker, "Hi, how are you doing today?")
        assert chunks[0] != "Hi,"


class TestSentenceSplitting:
    def test_multiple_sentences(self) -> None:
        chunker = SentenceChunker()
        chunks = stream(chunker, "First one. Second one. Third one.")
        assert chunks == ["First one.", "Second one.", "Third one."]

    def test_question_and_exclamation(self) -> None:
        chunker = SentenceChunker()
        chunks = stream(chunker, "Really? That's great! Okay then.")
        assert chunks[0] == "Really?"
        assert "That's great!" in chunks

    def test_decimals_are_not_split(self) -> None:
        chunker = SentenceChunker()
        chunks = stream(chunker, "It is 3.14 exactly. Done.")
        joined = " ".join(chunks)
        assert "3.14" in joined
        assert not any(c.strip() == "3." for c in chunks)

    def test_version_numbers_are_not_split(self) -> None:
        chunker = SentenceChunker()
        chunks = stream(chunker, "Running 1.2 now. Fine.")
        assert "1.2" in " ".join(chunks)


class TestStreamingBehaviour:
    def test_nothing_emitted_until_a_boundary_arrives(self) -> None:
        chunker = SentenceChunker()
        assert chunker.feed("Hey there") == []
        assert chunker.feed(" you") == []
        emitted = chunker.feed(" sound tired,")
        assert emitted  # the comma completes the first clause

    def test_chunk_boundaries_are_independent_of_fragment_size(self) -> None:
        text = "Okay. That works, I think. Done now."
        by_char = stream(SentenceChunker(), text, chunk_size=1)
        by_big = stream(SentenceChunker(), text, chunk_size=100)
        assert by_char == by_big

    def test_flush_emits_unpunctuated_tail(self) -> None:
        # Models don't always end on punctuation, especially when the
        # token budget cuts them off — the tail must not be swallowed.
        chunker = SentenceChunker()
        chunker.feed("All done. And then")
        assert chunker.flush() == ["And then"]

    def test_flush_on_empty_buffer_emits_nothing(self) -> None:
        chunker = SentenceChunker()
        chunker.feed("Done.")
        assert chunker.flush() == []

    def test_whitespace_only_input_emits_nothing(self) -> None:
        chunker = SentenceChunker()
        assert chunker.feed("   \n  ") == []
        assert chunker.flush() == []

    def test_no_text_is_lost(self) -> None:
        text = "Hey, I checked the thing. It works. Mostly, anyway"
        chunks = stream(SentenceChunker(), text)
        rejoined = " ".join(chunks).split()
        assert rejoined == text.split()
