"""Load Neiro's character prompt, and keep its bytes stable.

**Why the bytes matter.** llama-server and ollama both cache the KV
state of a stable prompt prefix. If the system prompt is byte-identical
every turn and the conversation is appended rather than rewritten, the
prefix cache hits and prefill costs almost nothing. If a single
character drifts — a rebuilt f-string, a re-ordered tool list, a
timestamp — the cache misses and every turn pays full prefill, which
the research put at 0.3-1.3s. That is a latency cliff with no error
message, which is exactly the kind of failure this project designs out.

So: the prompt is a file, loaded once, cached in memory, and never
interpolated with anything per-turn. Anything that varies per turn (the
`[voice: ...]` annotation, the transcript) goes on the USER message,
never in here.

`PROMPT_VERSION` is logged with every turn so a behaviour change can
always be traced back to which prompt produced it.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"
PROMPT_VERSION = "neiro.v3"


@lru_cache(maxsize=4)
def load_prompt(version: str = PROMPT_VERSION) -> str:
    """Read a character prompt from disk. Cached, so repeated turns read
    the same bytes rather than re-reading (and risking) the file.
    """
    path = PROMPTS_DIR / f"{version}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"Character prompt not found at {path}. Neiro without a character "
            "prompt is a chatbot with a face bolted on — this is not an "
            "optional file."
        )
    return path.read_text().strip()


def prompt_fingerprint(version: str = PROMPT_VERSION) -> str:
    """Short hash of the prompt bytes.

    Logged per session so that "she started behaving differently" can be
    correlated with "the prompt changed", rather than debugged from
    scratch.
    """
    return hashlib.sha256(load_prompt(version).encode()).hexdigest()[:12]


def system_message(version: str = PROMPT_VERSION) -> dict[str, str]:
    """The system message, in the shape the chat API wants.

    Always first in the messages list, always identical — see the module
    docstring on why that is load-bearing rather than tidy.
    """
    return {"role": "system", "content": load_prompt(version)}


def user_message(transcript: str, voice_annotation: str | None = None) -> dict[str, str]:
    """A user turn, with the prosody annotation inline if there is one.

    The annotation goes HERE, on the user turn, and never into the
    system prompt: mood in the system prompt bleeds across every
    subsequent turn and turns the character into a caricature of
    whatever she was told once.

    `voice_annotation` is omitted entirely below the dead-band or the
    confidence floor — not softened, omitted. Telling her "he sounds
    normal" every turn is noise she will eventually act on.
    """
    if voice_annotation:
        return {"role": "user", "content": f"[voice: {voice_annotation}] {transcript}"}
    return {"role": "user", "content": transcript}
