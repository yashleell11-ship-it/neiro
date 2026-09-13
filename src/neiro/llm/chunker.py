"""Split Neiro's streaming reply into speakable chunks.

Why this exists: TTS synthesises a chunk at a time, and time-to-first-
audio scales with the length of the chunk you hand it. If you wait for
her whole reply before synthesising, the user waits for the whole reply.
If you hand TTS each sentence as it completes, she starts talking while
the model is still writing — which is most of how a sub-second budget
is actually met.

The short-first-clause rule: the FIRST chunk is emitted at the earliest
clause boundary (comma, semicolon, colon, dash) rather than waiting for
the sentence to end, because Kokoro's time-to-first-audio is
proportional to the duration of what it's given. Gate G6 measured this.
A ~4-word opener starts audibly faster than a ~20-word sentence, and on
every single turn. Later chunks split on sentence boundaries only —
splitting mid-clause throughout would make her prosody choppy.

Deliberately simple and deliberately visible: this does not attempt
abbreviation handling beyond the obvious ("Dr." / "e.g." will split a
chunk early). The cost of that is an occasional slightly-short chunk,
which is inaudible. The cost of a clever-but-wrong splitter is her
pausing in the middle of a word.
"""

from __future__ import annotations

import re

# Sentence-final punctuation followed by whitespace or end-of-string.
#
# There is deliberately NO `(?<!\d)` lookbehind here. An earlier version
# had one, to keep "3.14" and "v1.2" intact — but it also blocked every
# sentence ENDING in a digit, and Neiro says numbers constantly:
# "Battery is at 96. Still charging." never split, so the whole reply
# went to the synthesiser as one chunk and paid full TTFA (Gate G6: 1088
# ms for 16 words versus 384 for three).
#
# The lookbehind was never needed: the trailing `(\s+|$)` already
# protects decimals, because the "." in "3.14" is followed by a digit
# rather than by whitespace, so it cannot match in the first place.
_SENTENCE_END = re.compile(r"([.!?])(\s+|$)")

# Clause boundaries, used only to get the FIRST chunk out fast.
_CLAUSE_END = re.compile(r"([,;:—–])(\s+|$)")

# Don't emit a first chunk shorter than this MANY WORDS — but only when
# splitting at a CLAUSE boundary. "Hey," alone reads as clipped.
# A complete SENTENCE is always fine to emit however short it is:
# "Really?" and "Okay." are natural whole utterances, and holding them
# back would delay the first audio for no reason.
MIN_FIRST_CLAUSE_WORDS = 2


def _word_count(text: str) -> int:
    return len(text.split())


class SentenceChunker:
    """Feed streaming text, get back complete chunks ready to synthesise.

    Usage::

        chunker = SentenceChunker()
        for text in stream:
            for chunk in chunker.feed(text):
                tts.synth(chunk)
        for chunk in chunker.flush():
            tts.synth(chunk)
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._emitted_first = False

    def feed(self, text: str) -> list[str]:
        """Add streamed text; return any chunks that are now complete."""
        self._buffer += text
        chunks = []
        while True:
            chunk = self._take_one()
            if chunk is None:
                break
            chunks.append(chunk)
        return chunks

    def flush(self) -> list[str]:
        """End of stream: emit whatever is left, even if unpunctuated."""
        remaining = self._buffer.strip()
        self._buffer = ""
        if remaining:
            self._emitted_first = True
            return [remaining]
        return []

    def _take_one(self) -> str | None:
        """Pull one complete chunk off the front of the buffer, or None."""
        if not self._buffer.strip():
            return None

        split_at = None

        if not self._emitted_first:
            # Short-first-clause rule: emit at whichever comes first, a
            # complete sentence or a long-enough clause.
            #
            # A sentence end is ALWAYS acceptable regardless of length —
            # "Really?" and "Okay." are whole utterances, and holding
            # them back to reach some word count would delay first audio
            # for nothing. The minimum applies only to clause fragments,
            # where "Hey," alone would read as clipped.
            sentence = _SENTENCE_END.search(self._buffer)
            sentence_end = sentence.end() if sentence else None

            clause_end = None
            for match in _CLAUSE_END.finditer(self._buffer):
                if _word_count(self._buffer[: match.end()]) >= MIN_FIRST_CLAUSE_WORDS:
                    clause_end = match.end()
                    break

            candidates = [e for e in (sentence_end, clause_end) if e is not None]
            if candidates:
                split_at = min(candidates)
        else:
            sentence = _SENTENCE_END.search(self._buffer)
            if sentence:
                split_at = sentence.end()

        if split_at is None:
            return None

        chunk = self._buffer[:split_at].strip()
        self._buffer = self._buffer[split_at:]
        if not chunk:
            return None
        self._emitted_first = True
        return chunk
