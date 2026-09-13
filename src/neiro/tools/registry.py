"""The typed tool registry — what she is allowed to do, enforced in code.

The plan's threat model is specific and was proven by experiment: on this
machine `hyprctl dispatch X` is a **Lua 5.5 eval with `os.execute` in
scope**. The spec's original claim — "the worst case of a misheard
command is the wrong volume" — is false here. So the design rule is not
"validate the model's strings carefully"; it is **the model never emits
a string that reaches a command at all**.

Three walls, in order, and each one alone would be enough for most of
the risk:

1. **Shape.** Every tool's arguments are a pydantic model with
   `extra="forbid"`. An unexpected key is a rejection, not a warning.
   Free-form `str` is refused *at registration time* — a tool that wants
   a window, an app or a unit takes an integer index into a list Neiro
   produced this turn, or a `Literal` enum. `str` is allowed only where
   it is declared to be spoken content, never an identifier.

2. **Tier.** GREEN runs. YELLOW needs confirmation the model cannot
   fake, because the nonce is computed here and never shown to it. RED
   is not registered at all — the model cannot express it, so there is
   nothing to gate.

3. **Budget.** Per-tool rate limits and a global side-effect budget, so
   a loop cannot do 400 small allowed things.

`to_openai_schema()` is what the LLM sees, and it is generated *from*
the same pydantic model that validates the call. They cannot drift,
which is the usual way a registry like this quietly stops matching
what it advertises.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel, ValidationError

from neiro.tools.tiers import Tier

# Types an argument may legally be. `str` is deliberately absent — see
# the module docstring. A tool that genuinely needs free text opts in
# per-field via `spoken_text_fields`.
_SAFE_SCALARS = (int, float, bool)


class ToolError(RuntimeError):
    """A tool call that must not run. The message is spoken to Yash, so
    it says what happened rather than quoting a traceback.
    """


class ToolRejected(ToolError):
    """Arguments failed validation, or the tool does not exist."""


class ToolNotConfirmed(ToolError):
    """A YELLOW tool whose confirmation was absent, ambiguous or denied."""


class RateLimited(ToolError):
    """Per-tool bucket or the global side-effect budget is exhausted."""


def _field_is_safe(annotation: Any, name: str, spoken: frozenset[str]) -> str | None:
    """Return a reason the field is unsafe, or None if it is fine."""
    if name in spoken:
        return None if annotation is str else f"{name!r} is declared spoken text but is not `str`"
    origin = get_origin(annotation)
    if origin is Literal:
        bad = [a for a in get_args(annotation) if not isinstance(a, (str, int, bool))]
        return f"{name!r} Literal has non-scalar members {bad}" if bad else None
    if annotation in _SAFE_SCALARS:
        return None
    if annotation is str:
        return (
            f"{name!r} is a bare `str`. The model must never emit a string that "
            "reaches a command — use an int index into a list Neiro produced this "
            "turn, or a Literal enum. If it is genuinely spoken content, add it to "
            "`spoken_text_fields`."
        )
    return f"{name!r} has unsupported type {annotation!r}"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str  # shown to the model; this is prompt text, write it as such
    tier: Tier
    args_model: type[BaseModel]
    handler: Callable[..., str]
    # Fields that are genuinely spoken content rather than identifiers.
    # Opt-in and explicit, so `str` can never slip in by accident.
    spoken_text_fields: frozenset[str] = frozenset()
    max_per_minute: int = 6

    def to_openai_schema(self) -> dict:
        """The tool definition the model sees — generated from the same
        model that validates the call, so the two cannot drift.
        """
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        schema.setdefault("additionalProperties", False)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }


@dataclass
class _Bucket:
    """A plain sliding window. No dependency, and easy to reason about
    when it is the thing standing between a loop and 400 actions.
    """

    limit: int
    window_s: float = 60.0
    stamps: list[float] = field(default_factory=list)

    def allow(self, now: float) -> bool:
        self.stamps = [t for t in self.stamps if now - t < self.window_s]
        if len(self.stamps) >= self.limit:
            return False
        self.stamps.append(now)
        return True


class ToolRegistry:
    """Everything Neiro can do, and everything she cannot.

    `confirm` is injected rather than imported so the confirmation
    mechanism (voice + a clickable notification, where the click wins)
    can be swapped and, more importantly, tested without a desktop.
    """

    #  A global ceiling across all side-effecting tools, so many
    #  individually-allowed actions cannot add up to a runaway.
    SIDE_EFFECT_BUDGET_PER_MIN = 30

    def __init__(
        self,
        confirm: Callable[[ToolSpec, BaseModel, str], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._buckets: dict[str, _Bucket] = {}
        self._global = _Bucket(self.SIDE_EFFECT_BUDGET_PER_MIN)
        self._confirm = confirm
        self._clock = clock

    # -- registration ---------------------------------------------------

    def register(self, spec: ToolSpec) -> ToolSpec:
        """Add a tool. Raises at *registration* time for an unsafe shape.

        Deliberately at import time rather than at call time: a tool with
        a free-form string argument should stop the process on startup,
        not fail once in production when a 4B model finally emits the
        wrong thing.
        """
        if spec.tier is Tier.RED:
            raise ValueError(
                f"{spec.name!r} is RED and must not be registered at all. RED tools "
                "are excluded from the tools array so the model cannot express them; "
                "gating them here would still let it try."
            )
        if spec.name in self._tools:
            raise ValueError(f"duplicate tool {spec.name!r}")
        for name, info in spec.args_model.model_fields.items():
            reason = _field_is_safe(info.annotation, name, spec.spoken_text_fields)
            if reason:
                raise ValueError(f"{spec.name!r}: {reason}")
        if spec.args_model.model_config.get("extra") != "forbid":
            raise ValueError(
                f"{spec.name!r}: args model must set extra='forbid'. An unexpected "
                "key from the model is a rejection, not something to ignore."
            )
        self._tools[spec.name] = spec
        self._buckets[spec.name] = _Bucket(spec.max_per_minute)
        return spec

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self, tier_ceiling: Tier | None = None) -> list[dict]:
        """The `tools` array for the LLM request.

        Stable order, because the system prompt and tool list are part of
        the KV prefix — reordering them costs a full prefill every turn
        (0.3-1.3 s, per the research) with no error to explain it.
        """
        allowed = (Tier.GREEN,) if tier_ceiling is Tier.GREEN else (Tier.GREEN, Tier.YELLOW)
        return [
            self._tools[name].to_openai_schema()
            for name in sorted(self._tools)
            if self._tools[name].tier in allowed
        ]

    # -- execution ------------------------------------------------------

    def validate(self, name: str, args: dict) -> tuple[ToolSpec, BaseModel]:
        """Resolve and type-check a call without running it."""
        spec = self._tools.get(name)
        if spec is None:
            # Naming the real tools is safe (the model already has the
            # list) and turns a hallucinated call into a recoverable turn.
            raise ToolRejected(
                f"There's no tool called {name!r}. I have: {', '.join(self.names())}."
            )
        try:
            parsed = spec.args_model.model_validate(args)
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(p) for p in first.get("loc", ())) or "arguments"
            raise ToolRejected(f"{name}: {where} — {first.get('msg', 'invalid')}") from exc
        return spec, parsed

    def call(self, name: str, args: dict, turn_id: int = 0) -> str:
        """Validate, gate, and run. Returns what Neiro says about it."""
        spec, parsed = self.validate(name, args)
        now = self._clock()

        if not self._buckets[spec.name].allow(now):
            raise RateLimited(
                f"I've done {name} too many times in the last minute. Give it a second."
            )
        if spec.tier is not Tier.GREEN and not self._global.allow(now):
            raise RateLimited("That's a lot of changes at once. I've stopped for a minute.")

        if spec.tier is Tier.YELLOW:
            if self._confirm is None:
                raise ToolNotConfirmed(
                    f"{name} needs confirming and I have no way to ask right now."
                )
            # The nonce binds the confirmation to THIS tool, THESE
            # arguments and THIS turn. It is computed here and never sent
            # to the model, so a reply cannot forge one — the confirmation
            # is not something the LLM participates in.
            nonce = self.nonce(spec, parsed, turn_id)
            if not self._confirm(spec, parsed, nonce):
                raise ToolNotConfirmed(f"Okay, not doing {name}.")

        return spec.handler(**parsed.model_dump())

    @staticmethod
    def nonce(spec: ToolSpec, parsed: BaseModel, turn_id: int) -> str:
        import hashlib

        payload = f"{spec.name}|{parsed.model_dump_json()}|{turn_id}"
        return hashlib.sha256(payload.encode()).hexdigest()[:12]
