"""The model manifest's contract.

The rule worth a test: a spec marked `purpose="training"` must not point
at GGUF or CTranslate2 weights. Those are quantised for inference and
cannot be fine-tuned — a mistake that costs a 17 GB download and an hour
of writing a training script around a file that will never accept a
gradient.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from elizabeth.modelspec import MODELS, ModelSpec, by_name, select, total_gb


def _spec(**overrides: object) -> ModelSpec:
    base = {
        "name": "x",
        "component": "llm",
        "purpose": "runtime",
        "runs": "local",
        "hf_id": "org/repo",
        "license": "mit",
        "size_gb": 1.0,
    }
    base.update(overrides)
    return ModelSpec(**base)  # type: ignore[arg-type]


class TestValidation:
    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _spec(licence="mit")

    @pytest.mark.parametrize("bad", ["../etc", "Has Space", "UPPER", ""])
    def test_name_must_be_a_safe_dirname(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            _spec(name=bad)

    @pytest.mark.parametrize("bad", ["noslash", "a/b/c", "/leading", "trailing/"])
    def test_hf_id_must_be_org_slash_repo(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            _spec(hf_id=bad)


class TestTrainableWeights:
    def test_gguf_pattern_cannot_be_training(self) -> None:
        with pytest.raises(ValidationError, match="inference only"):
            _spec(purpose="training", allow=["Model-Q4_K_M.gguf"])

    def test_gguf_repo_cannot_be_training(self) -> None:
        with pytest.raises(ValidationError, match="inference only"):
            _spec(purpose="training", hf_id="unsloth/Qwen3.5-4B-GGUF")

    def test_ct2_repo_cannot_be_training(self) -> None:
        with pytest.raises(ValidationError, match="inference only"):
            _spec(purpose="training", hf_id="distil-whisper/distil-large-v3.5-ct2")

    def test_safetensors_repo_may_be_training(self) -> None:
        assert _spec(purpose="training", hf_id="Qwen/Qwen3.5-4B").purpose == "training"

    def test_gguf_is_fine_for_runtime(self) -> None:
        assert _spec(purpose="runtime", allow=["x-Q4_K_M.gguf"]).allow == ["x-Q4_K_M.gguf"]


class TestSelection:
    def test_runs_filter_includes_both(self) -> None:
        # A model that runs on either machine is wanted by each of them.
        names = {m.name for m in select(runs="local")}
        assert "qwen3.5-4b-gguf" in names  # runs="local"
        assert "silero-vad" in names  # runs="both"
        assert "qwen3.6-35b-a3b-q3" not in names  # runs="box"

    def test_purpose_filter(self) -> None:
        assert all(m.purpose == "training" for m in select(purpose="training"))
        assert {m.name for m in select(purpose="training")} >= {
            "qwen3.5-4b-safetensors",
            "distil-large-v3.5",
            "w2v-bert-2.0",
        }

    def test_only_wins_and_is_loud_when_wrong(self) -> None:
        assert [m.name for m in select(only=["kokoro-82m"])] == ["kokoro-82m"]
        with pytest.raises(KeyError):
            select(only=["nope"])

    def test_by_name(self) -> None:
        assert by_name("kokoro-82m").component == "tts"
        with pytest.raises(KeyError):
            by_name("nope")

    def test_totals(self) -> None:
        assert total_gb(list(MODELS)) > 50  # the whole set is a real download


class TestTheRealSpec:
    def test_names_are_unique(self) -> None:
        names = [m.name for m in MODELS]
        assert len(names) == len(set(names))

    def test_every_licence_is_one_we_can_ship(self) -> None:
        # `elizabeth licences` (T18) enforces this at runtime; this catches it
        # the moment a model is added instead.
        allowed = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "cc0-1.0", "cc-by-4.0"}
        for m in MODELS:
            assert m.license in allowed, f"{m.name}: {m.license}"

    def test_the_laptop_tier_can_answer_a_turn_alone(self) -> None:
        # The hostel has no box. Whatever `--runs local` pulls must cover
        # a whole turn: brain, ears, voice.
        local = {m.component for m in select(runs="local", purpose="runtime")}
        assert {"llm", "stt", "tts"} <= local

    def test_gguf_specs_pull_exactly_one_quant(self) -> None:
        # A GGUF repo holds every quantisation. Without `allow`, "download
        # the model" means ~70 GB instead of 2.74.
        for m in MODELS:
            if "GGUF" in m.hf_id:
                assert len(m.allow) == 1 and m.allow[0].endswith(".gguf"), m.name


def _fetch_models():
    """The script, imported the way `elizabeth fetch-models` imports it."""
    import importlib
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("fetch_models")


class TestWhereAModelLands:
    def test_weights_land_under_the_models_directory(self, tmp_path) -> None:
        fetch_models = _fetch_models()
        assert fetch_models.dest_for(_spec(name="brain"), tmp_path) == tmp_path / "brain"

    def test_an_avatar_lands_where_the_page_looks(self, tmp_path, monkeypatch) -> None:
        # The daemon serves the avatar to the browser; the page finds it
        # under web/public/avatar. A VRM downloaded anywhere else is a
        # download the face never sees.
        from elizabeth.server import avatar_url

        fetch_models = _fetch_models()
        web_dir = tmp_path / "web"
        monkeypatch.setattr(fetch_models, "AVATAR_DEST", web_dir / "public" / "avatar")
        landed = fetch_models.dest_for(_spec(name="elizabeth-vrm", component="avatar"), tmp_path)
        assert landed != tmp_path / "elizabeth-vrm"
        landed.mkdir(parents=True)
        (landed / "elizabeth.vrm").write_bytes(b"glTF")
        assert avatar_url(web_dir) == "public/avatar/elizabeth-vrm/elizabeth.vrm"
