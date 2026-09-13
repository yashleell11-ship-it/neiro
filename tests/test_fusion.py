"""Combining the two affect lanes.

The idea being protected: Lane A speaks in z-scores against Yash's own
voice, Lane B speaks in circumplex coordinates learned from actors.
Averaging those directly is a category error that would look like it
worked, so Lane B gets baselined onto the same scale first — which also
removes any standing bias the model has on his accent.
"""

from __future__ import annotations

import pytest

from neiro.affect.fusion import (
    AGREEMENT_BONUS,
    DISAGREEMENT_PENALTY,
    LANE_A_WEIGHT,
    LANE_B_WEIGHT,
    LaneBCalibration,
    fuse,
)
from neiro.config import Neiro
from neiro.state import UserAffect


class TestLaneBCalibration:
    def test_no_history_means_no_z_score(self) -> None:
        # Same rule as the prosody baseline: "no idea" must not look like
        # "exactly average".
        assert LaneBCalibration().to_z(0.5, 0.2) == (None, None)

    def test_a_typical_reading_scores_near_zero(self) -> None:
        cal = LaneBCalibration()
        for _ in range(20):
            cal.observe(0.10, -0.05)
        arousal_z, _ = cal.to_z(0.10, -0.05)
        assert arousal_z is not None and abs(arousal_z) < 0.5

    def test_a_model_biased_on_his_accent_is_corrected(self) -> None:
        # The practical payoff. A SER model that reads every
        # Indian-accented voice as somewhat angry has a non-zero median
        # on him; subtracting that median removes the bias entirely, so
        # his ordinary voice stops reading as activated.
        cal = LaneBCalibration()
        for _ in range(30):
            cal.observe(0.62, -0.30)  # "angry-ish" on every ordinary utterance
        arousal_z, _ = cal.to_z(0.62, -0.30)
        assert arousal_z is not None and abs(arousal_z) < 1.0

    def test_a_genuinely_raised_reading_still_scores_high(self) -> None:
        cal = LaneBCalibration()
        for i in range(30):
            cal.observe(0.10 + 0.01 * (i % 3), 0.0)
        arousal_z, _ = cal.to_z(0.85, 0.0)
        assert arousal_z is not None and arousal_z > 2.0

    def test_one_outlier_does_not_move_the_calibration(self) -> None:
        cal = LaneBCalibration()
        for _ in range(30):
            cal.observe(0.1, 0.0)
        before = cal.to_z(0.1, 0.0)[0]
        cal.observe(0.99, 0.99)
        assert abs(cal.to_z(0.1, 0.0)[0] - before) < 0.5

    def test_window_is_bounded(self) -> None:
        cal = LaneBCalibration(window=5)
        for i in range(20):
            cal.observe(i / 20, 0.0)
        assert cal.n == 5


class TestFusion:
    def test_no_lanes_is_nothing(self) -> None:
        assert fuse(None, None) == UserAffect.NONE

    def test_zero_confidence_lanes_are_nothing(self) -> None:
        assert fuse(UserAffect(arousal_z=2.0, confidence=0.0), None) == UserAffect.NONE

    def test_one_lane_passes_through_unchanged(self) -> None:
        # For most of this project's life Lane A is alone. Penalising
        # "only one opinion" would punish the normal case.
        only = UserAffect(arousal_z=1.5, confidence=0.8)
        assert fuse(only, None) == only
        assert fuse(None, only) == only

    def test_agreement_raises_confidence(self) -> None:
        # Relative to the configured dead-band, so tuning it does not
        # silently turn this test into one about something else.
        band = Neiro().affect.dead_band_z
        a = UserAffect(arousal_z=band * 1.5, confidence=0.6)
        b = UserAffect(arousal_z=band * 1.2, confidence=0.6)
        fused = fuse(a, b)
        assert fused.arousal_z > 0
        assert fused.confidence > 0.6

    def test_disagreement_lowers_confidence_rather_than_averaging_it_away(self) -> None:
        # Two independent measurements pointing opposite ways mean the
        # reading is unreliable — the prompt's "ignore it at low
        # confidence" rule is what should act on this.
        band = Neiro().affect.dead_band_z
        a = UserAffect(arousal_z=band * 1.5, confidence=0.8)
        b = UserAffect(arousal_z=-band * 1.5, confidence=0.8)
        fused = fuse(a, b)
        assert fused.confidence < 0.8 * DISAGREEMENT_PENALTY * 1.01
        assert fused.confidence < Neiro().affect.confidence_floor

    def test_a_neutral_lane_is_not_a_disagreement(self) -> None:
        # Sitting inside the dead-band is declining to say, not
        # contradicting. Counting it as conflict would suppress every
        # reading where one lane is simply quiet.
        a = UserAffect(arousal_z=Neiro().affect.dead_band_z * 1.5, confidence=0.8)
        quiet = UserAffect(arousal_z=0.0, confidence=0.8)
        assert fuse(a, quiet).confidence > 0.8 * DISAGREEMENT_PENALTY * 1.5

    def test_an_unsure_lane_contributes_proportionally_little(self) -> None:
        confident = UserAffect(arousal_z=2.0, confidence=0.9)
        unsure = UserAffect(arousal_z=-2.0, confidence=0.05)
        # Weighted by confidence, so the near-silent lane must not drag
        # the answer to the midpoint.
        assert fuse(confident, unsure).arousal_z > 1.0

    def test_lane_a_leads_while_lane_b_is_unproven(self) -> None:
        # Until G3b says otherwise on HIS recordings, the lane measured
        # against his own voice outranks the one trained on actors.
        assert LANE_A_WEIGHT > LANE_B_WEIGHT
        a = UserAffect(arousal_z=2.0, confidence=0.7)
        b = UserAffect(arousal_z=1.0, confidence=0.7)
        assert fuse(a, b).arousal_z > 1.5

    def test_confidence_stays_in_range(self) -> None:
        a = UserAffect(arousal_z=2.0, confidence=1.0)
        b = UserAffect(arousal_z=2.0, confidence=1.0)
        assert 0.0 <= fuse(a, b).confidence <= 1.0
        assert AGREEMENT_BONUS > 1.0

    def test_events_are_merged_without_duplicates(self) -> None:
        a = UserAffect(arousal_z=1.0, confidence=0.6, events=("laughter",))
        b = UserAffect(arousal_z=1.0, confidence=0.6, events=("laughter", "sigh"))
        assert fuse(a, b).events == ("laughter", "sigh")

    def test_baseline_n_takes_the_less_mature_lane(self) -> None:
        # A fused reading is only as warmed-up as its coldest input.
        a = UserAffect(arousal_z=1.0, confidence=0.6, baseline_n=50)
        b = UserAffect(arousal_z=1.0, confidence=0.6, baseline_n=6)
        assert fuse(a, b).baseline_n == 6

    @pytest.mark.parametrize("z", [-3.0, -1.0, 0.0, 1.0, 3.0])
    def test_fusing_a_lane_with_itself_is_that_lane(self, z: float) -> None:
        one = UserAffect(arousal_z=z, valence_z=z / 2, confidence=0.7)
        fused = fuse(one, one)
        assert fused.arousal_z == pytest.approx(z)
        assert fused.valence_z == pytest.approx(z / 2)
