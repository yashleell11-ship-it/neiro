"""The output half of the emotional loop: what she feels reaching her
face and her voice.

`affect/` is the input half — what she HEARS in Yash's voice. This is the
other direction, and the two must never share a variable (CLAUDE.md
rule 5). One `ElizabethState`, parsed from her own `<e:LABEL:D>` tag, fans
out to three consumers that must agree: the VRM face, the TTS voice, and
the next turn's prompt.

"One shared emotional state" is the project's differentiating claim.
This package is where that claim is either true or a lie.
"""
