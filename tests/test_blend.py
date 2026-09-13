"""Expression blending — the visible half of "one shared emotional state".

Most of these assert things a human eye would notice. A face that fades
in and out at the same speed reads as mechanical; one that snaps reads
as a puppet; one that keeps 0.003 of "angry" all evening reads as
haunted. None of that shows up in a type check.
"""

from __future__ import annotations

import pytest

from neiro.config import Neiro
from neiro.emotion.blend import EXPRESSIVE, ExpressionBlender
from neiro.state import NEUTRAL_STATE, EmotionLabel, NeiroState

FRAME = 1.0 / 60.0


def run(blender: ExpressionBlender, state: NeiroState, seconds: float) -> dict[str, float]:
    for _ in range(max(1, int(seconds / FRAME))):
        blender.step(state, FRAME)
    return dict(blender.weights)


HAPPY = NeiroState.from_label(EmotionLabel.HAPPY, 0.8)
ANGRY = NeiroState.from_label(EmotionLabel.ANGRY, 0.9)
SURPRISED = NeiroState.from_label(EmotionLabel.SURPRISED, 0.9)


class TestApproach:
    def test_it_approaches_rather_than_snapping(self) -> None:
        # One frame must move the face part of the way, never all of it.
        b = ExpressionBlender()
        b.step(HAPPY, FRAME)
        assert 0.0 < b.weights["happy"] < 0.8

    def test_it_arrives(self) -> None:
        b = ExpressionBlender()
        assert b.settle(HAPPY)["happy"] == pytest.approx(0.8, abs=0.02)

    def test_intensity_from_the_tag_is_the_ceiling(self) -> None:
        # <e:happy:3> must not settle at the same place as <e:happy:9>.
        weak = ExpressionBlender().settle(NeiroState.from_label(EmotionLabel.HAPPY, 0.3))
        strong = ExpressionBlender().settle(NeiroState.from_label(EmotionLabel.HAPPY, 0.9))
        assert weak["happy"] < strong["happy"]
        assert weak["happy"] == pytest.approx(0.3, abs=0.02)

    def test_rise_is_faster_than_fall(self) -> None:
        # THE number in this file. Equal time constants are what make an
        # avatar look mechanical even when everything else is right.
        cfg = Neiro()
        assert cfg.expression.tau_rise_s < cfg.expression.tau_fall_s

        rising = ExpressionBlender()
        run(rising, HAPPY, 0.15)

        falling = ExpressionBlender()
        falling.settle(HAPPY)
        before = falling.weights["happy"]
        run(falling, NEUTRAL_STATE, 0.15)
        travelled_down = before - falling.weights["happy"]

        assert rising.weights["happy"] > travelled_down

    def test_dt_is_real_elapsed_time_not_a_fixed_step(self) -> None:
        # A dropped frame must not slow the expression down.
        many = ExpressionBlender()
        for _ in range(10):
            many.step(HAPPY, 0.01)
        one = ExpressionBlender()
        one.step(HAPPY, 0.1)
        assert one.weights["happy"] == pytest.approx(many.weights["happy"], abs=0.02)

    def test_zero_or_negative_dt_changes_nothing(self) -> None:
        b = ExpressionBlender()
        b.settle(HAPPY)
        before = dict(b.weights)
        assert b.step(HAPPY, 0.0) == before
        assert b.step(HAPPY, -1.0) == before


class TestReturningToRest:
    def test_neutral_releases_everything(self) -> None:
        b = ExpressionBlender()
        b.settle(HAPPY)
        assert b.settle(NEUTRAL_STATE)["happy"] == 0.0

    def test_tiny_weights_snap_to_zero(self) -> None:
        # An exponential never quite arrives. 0.003 of "angry" left on
        # her face all evening is a real thing that happens.
        b = ExpressionBlender()
        b.settle(ANGRY)
        final = b.settle(NEUTRAL_STATE, seconds=3.0)
        assert all(v == 0.0 for v in final.values())
        assert b.dominant() == ("neutral", 0.0)

    def test_crossfade_between_two_expressions(self) -> None:
        b = ExpressionBlender()
        b.settle(HAPPY)
        mid = run(b, ANGRY, 0.1)
        # Both present mid-transition — that is what a crossfade is.
        assert mid["angry"] > 0
        assert mid["happy"] > 0
        after = b.settle(ANGRY)
        assert after["angry"] > 0.5
        assert after["happy"] == 0.0


class TestSurprise:
    def test_it_releases_itself_without_a_new_state(self) -> None:
        # Physiologically brief. Held past a second it stops reading as
        # surprise and starts reading as a stare.
        b = ExpressionBlender()
        peak = run(b, SURPRISED, 0.9)["surprised"]
        later = run(b, SURPRISED, 1.6)["surprised"]
        assert peak > 0.5
        assert later < peak / 2

    def test_the_hold_is_long_enough_to_be_seen(self) -> None:
        b = ExpressionBlender()
        assert run(b, SURPRISED, 0.4)["surprised"] > 0.5

    def test_a_fresh_surprise_after_release_works_again(self) -> None:
        # The age counter must reset, or she can only ever be surprised
        # once per session.
        b = ExpressionBlender()
        run(b, SURPRISED, 3.0)
        b.settle(NEUTRAL_STATE)
        assert run(b, SURPRISED, 0.5)["surprised"] > 0.4


class TestVrmConstraints:
    def test_total_weight_never_exceeds_the_cap(self) -> None:
        # VRM expressions are additive on one mesh; past 1.0 the geometry
        # looks broken rather than expressive.
        cfg = Neiro()
        b = ExpressionBlender()
        b.settle(NeiroState.from_label(EmotionLabel.HAPPY, 1.0))
        for _ in range(200):
            b.step(NeiroState.from_label(EmotionLabel.ANGRY, 1.0), FRAME)
            assert sum(b.weights.values()) <= cfg.expression.max_total_weight + 1e-6

    def test_clamping_preserves_the_mix(self) -> None:
        # Scaling together, not zeroing the smaller ones, or a crossfade
        # would visibly jump.
        b = ExpressionBlender()
        b.weights["happy"] = 0.9
        b.weights["angry"] = 0.9
        ratio_before = b.weights["happy"] / b.weights["angry"]
        b._clamp_total()
        assert sum(b.weights.values()) == pytest.approx(1.0)
        assert b.weights["happy"] / b.weights["angry"] == pytest.approx(ratio_before)

    def test_neutral_is_not_driven_as_a_weight(self) -> None:
        # In VRM, neutral is the rest pose. Driving it fights the others
        # instead of blending with them; neutral is "all of these near 0".
        assert EmotionLabel.NEUTRAL not in EXPRESSIVE
        assert "neutral" not in ExpressionBlender().weights

    def test_only_the_five_vrm_presets_are_driven(self) -> None:
        presets = set(ExpressionBlender().weights)
        assert presets == {"happy", "angry", "sad", "relaxed", "surprised"}


class TestMissingPresets:
    """`surprised` is often unbound in real VRM models."""

    def test_a_missing_preset_falls_back_instead_of_doing_nothing(self) -> None:
        # A face that does nothing is indistinguishable from a bug.
        b = ExpressionBlender(available=frozenset({"happy", "angry", "sad", "neutral"}))
        weights = b.settle(SURPRISED)
        assert weights["surprised"] == 0.0
        assert weights["happy"] > 0.5

    def test_fallback_keeps_the_direction_right(self) -> None:
        # Angry falls back to sad, not to happy — still negative valence.
        b = ExpressionBlender(available=frozenset({"sad", "relaxed", "neutral"}))
        assert b.settle(ANGRY)["sad"] > 0.5

    def test_an_avatar_with_nothing_close_stays_still(self) -> None:
        b = ExpressionBlender(available=frozenset({"neutral"}))
        assert all(v == 0.0 for v in b.settle(SURPRISED).values())

    def test_no_available_list_means_drive_everything(self) -> None:
        # Before `ready{expressions[]}` arrives, assume a complete model.
        # Measured inside the hold window, not via settle() -- surprise
        # has correctly decayed to ~0.26 by settle()'s 2 seconds.
        assert run(ExpressionBlender(), SURPRISED, 0.5)["surprised"] > 0.5


class TestDominant:
    def test_reports_the_strongest_expression(self) -> None:
        b = ExpressionBlender()
        b.settle(ANGRY)
        name, value = b.dominant()
        assert name == "angry" and value > 0.5

    def test_reports_neutral_when_the_face_is_at_rest(self) -> None:
        assert ExpressionBlender().dominant() == ("neutral", 0.0)
