"""What Neiro hears in *how* Yash speaks — the project's differentiator.

Lane A only in v1: librosa prosody (pitch, energy, pace, pausing) scored
as z-scores against a rolling baseline of his own normal voice. Not a
trained emotion classifier — that is Lane B, gated on beating Lane A on
his own recordings.

Why the modest version is the one that ships: the best published system
on natural (non-acted) speech reaches macro-F1 **0.43** on eight
classes, and there is no published number at all for Indian-accented
English. What *is* reliable is that a person who is activated speaks
louder, higher, and faster than that same person's usual — which is a
measurement against a per-speaker baseline, not a classification.

Everything here writes `Turn.user_affect` and nothing else. It must
never touch `Turn.neiro_state`: one is what we heard, the other is what
she feels, and sharing a variable is how an assistant ends up reading
its own TTS back as the user's mood.
"""
