"""The dataset manifest's contract: licence flags and publishability
cannot disagree, every entry has exactly one source, and selection pulls
essentials first. No network — everything here is inline TOML.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from neiro.training.manifest import NON_PUBLISHABLE_FLAGS, TARGETS, Manifest

GOOD = """
schema_version = 1

[[dataset]]
name = "svarah"
target = "stt_indian_english"
priority = 1
hf_id = "ai4bharat/Svarah"
license = "CC-BY-4.0"
flags = ["permissive"]
access = "direct"
size_gb = 2.0
hours = 9.6
weights_publishable = "yes"

[[dataset]]
name = "ravdess"
target = "ser_lane_b"
priority = 2
url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
license = "CC-BY-NC-SA-4.0"
flags = ["NC", "SA"]
access = "direct"
size_gb = 0.2
hours = 1.5
weights_publishable = "no"

[[dataset]]
name = "soda"
target = "persona_lora"
priority = 1
hf_id = "allenai/soda"
license = "CC-BY-4.0"
flags = ["permissive"]
access = "direct"
size_gb = 1.2
weights_publishable = "yes"
"""


def _one(**overrides: object) -> str:
    base = {
        "name": "x",
        "target": "ser_lane_b",
        "priority": 1,
        "hf_id": "org/x",
        "url": "",
        "license": "MIT",
        "flags": ["permissive"],
        "access": "direct",
        "size_gb": 1.0,
        "weights_publishable": "yes",
    }
    base.update(overrides)
    lines = ["schema_version = 1", "", "[[dataset]]"]
    for k, v in base.items():
        if isinstance(v, str):
            lines.append(f'{k} = "{v}"')
        elif isinstance(v, list):
            lines.append(f"{k} = {v!r}".replace("'", '"'))
        else:
            lines.append(f"{k} = {v}")
    return "\n".join(lines)


class TestLoading:
    def test_good_manifest_loads(self) -> None:
        m = Manifest.loads(GOOD)
        assert [d.name for d in m.dataset] == ["svarah", "ravdess", "soda"]
        assert m.dataset[0].is_hf and not m.dataset[1].is_hf
        assert m.dataset[0].source.endswith("/ai4bharat/Svarah")

    def test_empty_manifest_is_valid(self) -> None:
        assert Manifest.loads("schema_version = 1").dataset == []

    def test_unknown_schema_version_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Manifest.loads("schema_version = 99")

    def test_unknown_field_rejected(self) -> None:
        # extra='forbid': a typo like "licence" must not silently vanish.
        with pytest.raises(ValidationError):
            Manifest.loads(_one() + '\nlicence = "MIT"')

    def test_duplicate_names_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Manifest.loads(_one() + "\n" + _one().split("\n", 2)[2])


class TestSourceRules:
    def test_needs_exactly_one_source(self) -> None:
        with pytest.raises(ValidationError):
            Manifest.loads(_one(hf_id="", url=""))
        with pytest.raises(ValidationError):
            Manifest.loads(_one(hf_id="org/x", url="https://x"))

    def test_sha256_only_for_urls(self) -> None:
        with pytest.raises(ValidationError):
            Manifest.loads(_one(sha256="ab" * 32))
        Manifest.loads(_one(hf_id="", url="https://x/y.zip", sha256="ab" * 32))

    def test_name_must_be_a_safe_dirname(self) -> None:
        for bad in ("../x", "Has Space", "UPPER", ""):
            with pytest.raises(ValidationError):
                Manifest.loads(_one(name=bad))


class TestPublishability:
    """The one rule the manifest exists to enforce."""

    @pytest.mark.parametrize("flag", sorted(NON_PUBLISHABLE_FLAGS))
    def test_non_publishable_flag_cannot_claim_yes(self, flag: str) -> None:
        with pytest.raises(ValidationError):
            Manifest.loads(_one(flags=[flag], weights_publishable="yes"))
        Manifest.loads(_one(flags=[flag], weights_publishable="no"))

    def test_unclear_licence_cannot_claim_yes(self) -> None:
        with pytest.raises(ValidationError):
            Manifest.loads(_one(flags=["unclear"], license="unclear", weights_publishable="yes"))
        Manifest.loads(_one(flags=["unclear"], license="unclear", weights_publishable="unclear"))

    def test_permissive_may_claim_yes(self) -> None:
        Manifest.loads(_one(flags=["permissive"], weights_publishable="yes"))

    def test_sa_alone_is_not_a_blocker(self) -> None:
        # Share-alike constrains the licence of derived data, not whether
        # weights can be released at all — a judgement call, so it's allowed
        # either way and must be decided per dataset in the note.
        Manifest.loads(_one(flags=["SA"], weights_publishable="yes"))


class TestSelection:
    def test_default_is_tier_one_only(self) -> None:
        m = Manifest.loads(GOOD)
        assert [d.name for d in m.select()] == ["svarah", "soda"]

    def test_tier_two_adds_the_rest_essentials_first(self) -> None:
        m = Manifest.loads(GOOD)
        assert [d.name for d in m.select(max_priority=2)] == ["svarah", "soda", "ravdess"]

    def test_target_filter(self) -> None:
        m = Manifest.loads(GOOD)
        assert [d.name for d in m.select(max_priority=3, target="ser_lane_b")] == ["ravdess"]

    def test_only_wins_over_tier(self) -> None:
        m = Manifest.loads(GOOD)
        assert [d.name for d in m.select(only=["ravdess"])] == ["ravdess"]

    def test_only_with_unknown_name_is_loud(self) -> None:
        with pytest.raises(KeyError):
            Manifest.loads(GOOD).select(only=["nope"])

    def test_totals(self) -> None:
        m = Manifest.loads(GOOD)
        assert Manifest.total_gb(m.dataset) == pytest.approx(3.4)
        assert Manifest.total_hours(m.dataset) == pytest.approx(11.1)

    def test_by_target_covers_every_target(self) -> None:
        groups = Manifest.loads(GOOD).by_target()
        assert set(groups) == set(TARGETS)
        assert [d.name for d in groups["persona_lora"]] == ["soda"]


def test_real_manifest_on_disk_loads() -> None:
    # The committed manifest must always validate — this is the test that
    # fails when someone hand-edits data/datasets.toml wrong.
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "data" / "datasets.toml"
    Manifest.load(path)
