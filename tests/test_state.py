"""Regression tests for the Turn spine (src/neiro/state.py).

Promised in the Stage 0 Task 1 commit but never actually written — the
whole reason this matters: VRM 1.0's expression preset names are exact
strings that `expressionManager.setValue()` matches silently-or-not-at-all
against. Get one wrong (VRM 0.x's uppercase A/I/U/E/O instead of 1.0's
aa/ih/ou/ee/oh, or "calm" instead of "relaxed") and the face simply never
moves, with no error anywhere.
"""

from __future__ import annotations

from neiro.state import (
    EmotionLabel,
    Locality,
    NeiroState,
    Tier,
    Turn,
    UserAffect,
)

# The VRM 1.0 normative expression preset names, verbatim from the spec:
# https://github.com/vrm-c/vrm-specification/.../VRMC_vrm-1.0/expressions.md
VRM_1_0_EXPRESSION_PRESETS = {
    "happy",
    "angry",
    "sad",
    "relaxed",
    "surprised",
    "neutral",
}


def test_emotion_labels_match_vrm_1_0_exactly() -> None:
    values = {label.value for label in EmotionLabel}
    assert values == VRM_1_0_EXPRESSION_PRESETS


def test_emotion_labels_are_not_vrm_0_x_spelling() -> None:
    # VRM 0.x used uppercase single-letter vowel names for MOUTH shapes
    # (A/I/U/E/O) — a different axis (visemes, not expressions) that this
    # module doesn't model, but the classic mistake is reaching for that
    # spelling out of habit. Confirm none of it leaked in here.
    values = {label.value for label in EmotionLabel}
    assert not values & {"A", "I", "U", "E", "O", "calm", "Happy", "Angry"}


def test_neiro_state_defaults_to_neutral() -> None:
    state = NeiroState()
    assert state.label is EmotionLabel.NEUTRAL
    assert state.valence == 0.0
    assert state.arousal == 0.0


def test_neiro_state_from_label_looks_up_valence_arousal() -> None:
    state = NeiroState.from_label(EmotionLabel.HAPPY, intensity=0.8)
    assert state.label is EmotionLabel.HAPPY
    assert state.intensity == 0.8
    assert state.valence > 0  # happy is positive valence...
    assert state.arousal > 0  # ...and activated, not sleepy


def test_user_affect_none_is_inert() -> None:
    # Stage 0's whole point: this must read as "say nothing" everywhere
    # it's consulted, not as a real (if boring) reading.
    assert UserAffect.NONE.confidence == 0.0
    assert UserAffect.NONE.baseline_n == 0


def test_user_affect_and_neiro_state_are_distinct_types() -> None:
    # The rule in CLAUDE.md: what she heard and what she feels must never
    # be assignable to each other. Enforced at the type level — this test
    # exists so nobody "simplifies" Turn into one shared field later.
    assert UserAffect is not NeiroState
    turn = Turn.new()
    assert type(turn.user_affect) is UserAffect
    assert type(turn.neiro_state) is NeiroState


def test_turn_starts_with_inert_defaults() -> None:
    turn = Turn.new(turn_id=7)
    assert turn.id == 7
    assert turn.tier is Tier.LOCAL
    assert turn.tool_calls == []
    assert turn.transcript is None
    assert not turn.cancel.is_set()


def test_turn_stamp_records_a_named_timestamp() -> None:
    turn = Turn.new()
    assert "endpoint" not in turn.timeline
    turn.stamp("endpoint")
    assert "endpoint" in turn.timeline
    assert isinstance(turn.timeline["endpoint"], float)


def test_turn_cancel_is_independent_per_instance() -> None:
    # asyncio.Event as a dataclass field(default_factory=...) — confirm
    # two Turns don't accidentally share one Event object.
    a, b = Turn.new(1), Turn.new(2)
    a.cancel.set()
    assert a.cancel.is_set()
    assert not b.cancel.is_set()


# The locality rule from the plan, as a full truth table. The point of
# having it in code is that a future TierResolver cannot "helpfully"
# promote STT over the tunnel, or a tool onto the box — that would be a
# red test, not a latency cliff discovered at the hostel.
LOCALITY_TRUTH_TABLE = {
    (Locality.LOCAL_PINNED, Tier.LOCAL): True,
    (Locality.LOCAL_PINNED, Tier.LAN): False,
    (Locality.LOCAL_PINNED, Tier.TUNNEL): False,
    (Locality.LAN_TIERABLE, Tier.LOCAL): True,
    (Locality.LAN_TIERABLE, Tier.LAN): True,
    (Locality.LAN_TIERABLE, Tier.TUNNEL): False,
    (Locality.TIERABLE, Tier.LOCAL): True,
    (Locality.TIERABLE, Tier.LAN): True,
    (Locality.TIERABLE, Tier.TUNNEL): True,
}


def test_locality_allows_matches_the_plan() -> None:
    # Every (locality, tier) pair is covered — no case is left to a default.
    assert set(LOCALITY_TRUTH_TABLE) == {(lo, ti) for lo in Locality for ti in Tier}
    for (locality, tier), expected in LOCALITY_TRUTH_TABLE.items():
        assert locality.allows(tier) is expected, f"{locality} on {tier}"


def test_every_tier_is_reachable_by_something_and_local_by_everything() -> None:
    # LOCAL is the floor: nothing is ever forbidden from running on the
    # machine Yash sits at, which is what makes the hostel tier always work.
    assert all(lo.allows(Tier.LOCAL) for lo in Locality)
    assert any(lo.allows(Tier.TUNNEL) for lo in Locality)
