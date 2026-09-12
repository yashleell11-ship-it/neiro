"""Parse Neiro's emotional state off the front of her own reply stream.

She is prompted to open every spoken reply with exactly one tag:

    <e:happy:8> Hey, you sound tired.

The label is one of the six VRM 1.0 expression presets; the intensity is
a single digit 0-9 (a digit rather than "0.8" saves two tokens, ~20 ms).
This parser strips that tag off the stream and publishes the state, so
only the speakable remainder reaches the sentence chunker and the TTS.

Why a prompted tag and not something more rigorous — all three
alternatives were ruled out on hard constraints, not taste:

  - GBNF grammar: llama.cpp throws `Cannot specify grammar with tools`,
    and Stage 3's tool registry means `tools` is present on every turn.
    Mutually exclusive, so a grammar-enforced tag is unavailable.
  - JSON mode / response_format: same exclusion, plus ~15 scaffolding
    tokens before the first speakable character, plus it forces the
    chunker to parse a half-formed escaped string mid-stream.
  - A `set_emotion` tool call, or a two-pass classify-then-speak: a
    whole extra prefill and TTFT, 250-500 ms, on every single turn.

The tradeoff a prompted tag accepts is that the model will sometimes not
emit one. That is handled by design rather than hoped away: after
`FALLBACK_AFTER_CHARS` characters with no match, this parser gives up,
publishes neutral, and forwards everything it buffered. **Audio never
blocks on the tag.** Tag compliance is a measured gate in Stage 1
(>= 98% over 200 turns), not an assumption.
"""

from __future__ import annotations

import re

from neiro.state import EmotionLabel, NeiroState

# Anchored at the start: the tag is only meaningful as an opener. A
# `<e:...>` appearing mid-sentence is the model misbehaving, and treating
# it as authoritative would let a stray tag hijack her face mid-reply.
_TAG = re.compile(
    r"^<e:(happy|angry|sad|relaxed|surprised|neutral):([0-9])>\s*",
    re.IGNORECASE,
)

# How much text to buffer before concluding no tag is coming. Generous
# enough for the ~8-12 characters a well-formed tag needs plus a little
# leading whitespace, small enough that the fallback costs a negligible
# delay. Never raise this far: every character here is a character the
# TTS hasn't started on yet.
FALLBACK_AFTER_CHARS = 32

_NEUTRAL_FALLBACK = NeiroState.from_label(EmotionLabel.NEUTRAL, intensity=0.5)


class EmotionTagParser:
    """Streaming parser. Feed it `delta.content` fragments as they
    arrive; it returns the speakable text to forward downstream, and
    publishes the state exactly once, the moment it's known.

    Usage::

        parser = EmotionTagParser()
        for chunk in stream:
            state, text = parser.feed(chunk)
            if state is not None:
                publish(state)      # face can start blending now
            if text:
                chunker.add(text)   # TTS gets only the speakable part
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._resolved = False
        # The tag's trailing separator (`<e:happy:8> hi`) often arrives in
        # a LATER chunk than the `>` that resolves the match, so the
        # regex's own `\s*` can't consume it. Without this flag, a
        # stream chunked as ["<e:happy:8>", " hi"] forwards " hi" with a
        # leading space.
        self._eating_separator = False

    @property
    def resolved(self) -> bool:
        """True once the state has been published (by match or fallback)."""
        return self._resolved

    def feed(self, chunk: str) -> tuple[NeiroState | None, str]:
        """Returns ``(state_if_resolved_by_this_chunk, text_to_forward)``.

        The state is returned exactly once, on the chunk that resolves
        it. Every later call returns ``(None, chunk)`` and is a
        pass-through.
        """
        if self._resolved:
            if self._eating_separator:
                chunk = chunk.lstrip()
                if chunk:
                    self._eating_separator = False
            return None, chunk

        self._buffer += chunk

        match = _TAG.match(self._buffer)
        if match:
            self._resolved = True
            label = EmotionLabel(match.group(1).lower())
            intensity = int(match.group(2)) / 9.0
            remainder = self._buffer[match.end() :]
            self._buffer = ""
            # If the tag ended exactly at the chunk boundary, the
            # separator whitespace is still to come — keep eating it.
            self._eating_separator = not remainder
            return NeiroState.from_label(label, intensity), remainder

        if len(self._buffer) >= FALLBACK_AFTER_CHARS:
            # Give up and let the audio start. Forward everything
            # buffered — the model said something speakable, it just
            # didn't label it, and dropping her words would be far worse
            # than losing the expression.
            self._resolved = True
            buffered = self._buffer
            self._buffer = ""
            return _NEUTRAL_FALLBACK, buffered

        # Still might be a tag arriving one token at a time — hold.
        return None, ""

    def flush(self) -> tuple[NeiroState | None, str]:
        """End of stream. If the reply was shorter than
        FALLBACK_AFTER_CHARS and had no tag, resolve to neutral and
        release whatever was buffered — otherwise a short untagged reply
        ("Sure.") would be silently swallowed.
        """
        if self._resolved:
            return None, ""
        self._resolved = True
        buffered = self._buffer
        self._buffer = ""
        return _NEUTRAL_FALLBACK, buffered
