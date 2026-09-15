"""The actual tools, wired into the registry.

Each entry is the pairing of a pydantic argument model with a handler
that already exists in `system.py`, `media.py` or `hyprland.py`. Nothing
here contains logic — if it did, the logic would be untested against the
real machine, which is the point of those modules.

**Descriptions are prompt text.** The model decides whether to call a
tool from this string alone, so it says *when to use it*, not what it
does internally. A description that reads like an API doc produces a
model that calls the tool in the wrong situations.

**`focus_window` takes an index, never a name.** The index refers to the
list `list_windows_tool()` produced *in the same turn*, which is the
whole defence against the Lua injection this registry exists to prevent.
Resolution from index to window address happens in `hyprland.py`, out of
the model's reach.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from elizabeth.tools import apps, hyprland, media, system, websearch
from elizabeth.tools.audit import AuditLog
from elizabeth.tools.registry import ToolRegistry, ToolSpec
from elizabeth.tools.tiers import Tier


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VolumeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    percent: int = Field(ge=0, le=100, description="Target volume, 0-100.")


class MuteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: Literal["on", "off", "toggle"]


class BrightnessArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    direction: Literal["up", "down"]
    steps: int = Field(default=1, ge=1, le=6, description="5% per step.")


class MediaArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # The same Literal `media.media_control` checks against — not a copy
    # of it — so the schema cannot advertise an action the handler
    # refuses. A member the model may legally emit must be executable.
    action: media.MediaAction


class WindowIndexArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int = Field(ge=0, description="Position in the window list from list_windows.")


class WorkspaceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    number: int = Field(ge=1, le=10)


class OpenAppArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: Literal["here", "pc"] = Field(
        description="'here' is this machine; 'pc' is Yash's other machine, the 3090 Ti box."
    )
    app: apps.App


class WebSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # The only `str` field in this registry that isn't an identifier —
    # it's what he actually asked to search for, so it's opted into
    # spoken_text_fields below rather than being a bare disallowed str.
    query: str = Field(min_length=1, max_length=200)


# --- handlers: thin adapters, no logic ---------------------------------


def _system_stats() -> str:
    return system.get_system_stats().describe()


def _audio_state() -> str:
    return system.get_audio_state().describe()


def _now_playing() -> str:
    playing = media.get_now_playing()
    return playing.describe() if playing else "Nothing's playing."


def _list_windows() -> str:
    windows = hyprland.list_windows()
    if not windows:
        return "No windows open."
    return ", ".join(f"[{i}] {w.app}" for i, w in enumerate(windows))


def _active_window() -> str:
    window = hyprland.active_window()
    return f"{window.app} is focused." if window else "Nothing's focused."


def _focus_window(index: int) -> str:
    # The index is resolved against a list fetched right now, inside this
    # process. The model never sees or supplies an address.
    return hyprland.focus_window_by_index(index, hyprland.list_windows())


def build_registry(confirm=None, *, audit: AuditLog | None = None) -> ToolRegistry:
    """Everything Elizabeth can do on this machine.

    `audit` is passed through rather than created here so a test can
    point it at a temporary file — the default log lives under
    ~/.local/state and a suite must never write there.
    """
    registry = ToolRegistry(confirm=confirm, audit=audit)

    green = [
        ToolSpec(
            name="system_stats",
            description="Battery, memory and free disk. Use when he asks how the machine is doing.",
            tier=Tier.GREEN,
            args_model=NoArgs,
            handler=_system_stats,
        ),
        ToolSpec(
            name="audio_state",
            description="Current volume and whether sound is muted.",
            tier=Tier.GREEN,
            args_model=NoArgs,
            handler=_audio_state,
        ),
        ToolSpec(
            name="now_playing",
            description="What music or video is playing right now, if anything.",
            tier=Tier.GREEN,
            args_model=NoArgs,
            handler=_now_playing,
        ),
        ToolSpec(
            name="list_windows",
            description=(
                "The open windows, numbered. Call this before focus_window — the "
                "number you pass there refers to this list."
            ),
            tier=Tier.GREEN,
            args_model=NoArgs,
            handler=_list_windows,
        ),
        ToolSpec(
            name="active_window",
            description="Which window he's looking at right now.",
            tier=Tier.GREEN,
            args_model=NoArgs,
            handler=_active_window,
        ),
    ]

    yellow = [
        ToolSpec(
            name="set_volume",
            description="Set the speaker volume to a specific percentage.",
            tier=Tier.YELLOW,
            args_model=VolumeArgs,
            handler=lambda percent: system.set_volume(percent),
        ),
        ToolSpec(
            name="set_mute",
            description="Mute, unmute, or toggle the sound.",
            tier=Tier.YELLOW,
            args_model=MuteArgs,
            handler=lambda state: system.set_mute(state),
        ),
        ToolSpec(
            name="adjust_brightness",
            description=(
                "Make the screen brighter or dimmer, in 5% steps. Relative only — "
                "there is no way to set an absolute brightness on this machine."
            ),
            tier=Tier.YELLOW,
            args_model=BrightnessArgs,
            handler=lambda direction, steps: system.adjust_brightness(direction, steps),
        ),
        ToolSpec(
            name="media_control",
            description="Play, pause or skip whatever is playing.",
            tier=Tier.YELLOW,
            args_model=MediaArgs,
            handler=lambda action: media.media_control(action),
        ),
        ToolSpec(
            name="focus_window",
            description=(
                "Switch to one of the windows from list_windows, by its number. "
                "Call list_windows first in the same turn."
            ),
            tier=Tier.YELLOW,
            args_model=WindowIndexArgs,
            handler=_focus_window,
        ),
        ToolSpec(
            name="switch_workspace",
            description="Go to workspace 1 to 10.",
            tier=Tier.YELLOW,
            args_model=WorkspaceArgs,
            handler=lambda number: hyprland.switch_workspace(number),
        ),
        ToolSpec(
            name="open_app",
            description=(
                "Launch an application, here or on Yash's other machine (the PC). "
                "Use when he asks to open, start or launch something by name."
            ),
            tier=Tier.YELLOW,
            args_model=OpenAppArgs,
            handler=lambda target, app: apps.open_app(target, app),
        ),
        ToolSpec(
            name="web_search",
            description=(
                "Search the web. Use only when he asks for something current or "
                "outside what you already know — not for things you can already answer."
            ),
            tier=Tier.YELLOW,
            args_model=WebSearchArgs,
            handler=lambda query: websearch.search_and_describe(query),
            spoken_text_fields=frozenset({"query"}),
        ),
    ]

    for tool in green + yellow:
        registry.register(tool)
    return registry
