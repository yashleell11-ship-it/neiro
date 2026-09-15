"""Text to speech. Stage 0 is Kokoro; the expressive voice is gate G5.

Kokoro is here for one reason and it is not quality: it returns **phoneme
durations**, which is free lip-sync data. Every codec-style TTS (Qwen3-TTS,
Chatterbox) returns audio and nothing else, so the mouth has to be driven
from amplitude instead. Collecting real visemes now, while the engine
hands them over, is what makes Stage 1's face cheap.

What Kokoro cannot do is emotion. Not "less well" — the model has no
emotion control at all, which is why gate G5 exists.
"""
