"""Tests for the two tools that leave this machine: apps.py (crosses
onto the box, over SSH) and websearch.py (crosses onto the internet).

Neither hits the network or launches a real process here — the
subprocess/httpx seam is stubbed, same pattern as media.py's `_run` in
test_builtin_tools.py. What's under test is the *shape*: an unknown app
raises rather than launching something arbitrary, a bad target is
refused by the registry before the handler ever sees it, and a failed
request is spoken as "couldn't", not swallowed into an empty result.
"""

from __future__ import annotations

import subprocess

import httpx
import pytest

from elizabeth.tools import apps, websearch


class TestOpenApp:
    def test_every_app_has_both_a_local_and_a_box_entry(self) -> None:
        # If these two tables ever drift, "here" and "pc" stop being
        # interchangeable for the same `app` value the model picked —
        # a silent asymmetry the schema doesn't advertise.
        assert set(apps._LOCAL_APPS) == set(apps._BOX_COMMANDS)

    def test_local_launch_uses_argv_never_a_shell_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = []
        monkeypatch.setattr(
            apps.subprocess, "Popen", lambda argv, **kw: calls.append((argv, kw)) or object()
        )
        apps.open_local("browser")
        (argv, kwargs) = calls[0]
        assert argv == apps._LOCAL_APPS["browser"]
        assert isinstance(argv, list)
        assert kwargs.get("start_new_session") is True

    def test_unknown_app_is_a_launch_error_not_a_crash(self) -> None:
        with pytest.raises(apps.AppLaunchError):
            apps.open_local("nonexistent")  # type: ignore[arg-type]

    def test_missing_binary_is_a_spoken_error_not_a_traceback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def missing(*a, **k):
            raise FileNotFoundError

        monkeypatch.setattr(apps.subprocess, "Popen", missing)
        with pytest.raises(apps.AppLaunchError):
            apps.open_local("browser")

    def test_box_launch_builds_a_fixed_ssh_command_no_model_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

        monkeypatch.setattr(apps.subprocess, "run", fake_run)
        apps.open_on_box("vscode")
        argv = captured["argv"]
        assert argv[0] == apps.SSH_EXE
        assert apps.BOX_HOST in argv
        assert apps.BOX_KEY in argv
        # The command Elizabeth built is the fixed table entry — nothing
        # about it came from the model, which only ever supplied the
        # Literal "vscode".
        assert apps._BOX_COMMANDS["vscode"] in argv[-1]

    def test_box_unreachable_is_a_spoken_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            apps.subprocess,
            "run",
            lambda argv, **kw: subprocess.CompletedProcess(
                argv, 255, stdout="", stderr="timed out"
            ),
        )
        with pytest.raises(apps.AppLaunchError):
            apps.open_on_box("vscode")

    def test_open_app_routes_on_target(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = []
        monkeypatch.setattr(apps, "open_local", lambda app: seen.append(("local", app)) or "ok")
        monkeypatch.setattr(apps, "open_on_box", lambda app: seen.append(("box", app)) or "ok")
        apps.open_app("here", "browser")
        apps.open_app("pc", "browser")
        assert seen == [("local", "browser"), ("box", "browser")]


class TestWebSearch:
    _SAMPLE_HTML = """
    <div class="result">
      <a class="result__a" href="https://a.example">First Result Title</a>
      <a class="result__snippet" href="https://a.example">A short snippet.</a>
    </div>
    <div class="result">
      <a class="result__a" href="https://b.example">Second &amp; Title</a>
      <a class="result__snippet" href="https://b.example">Another <b>snippet</b>.</a>
    </div>
    """

    def test_parses_titles_and_snippets_from_result_markup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            httpx,
            "post",
            lambda *a, **k: httpx.Response(
                200, text=self._SAMPLE_HTML, request=httpx.Request("POST", websearch.SEARCH_URL)
            ),
        )
        results = websearch.search("anything")
        assert [r.title for r in results] == ["First Result Title", "Second &amp; Title"]
        assert results[1].snippet == "Another snippet."

    def test_caps_at_max_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        many = self._SAMPLE_HTML * 10
        monkeypatch.setattr(
            httpx,
            "post",
            lambda *a, **k: httpx.Response(
                200, text=many, request=httpx.Request("POST", websearch.SEARCH_URL)
            ),
        )
        assert len(websearch.search("anything")) == websearch.MAX_RESULTS

    def test_a_failed_request_raises_rather_than_returning_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*a, **k):
            raise httpx.ConnectTimeout("timed out")

        monkeypatch.setattr(httpx, "post", boom)
        with pytest.raises(websearch.SearchUnavailable):
            websearch.search("anything")

    def test_describe_reads_naturally(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            httpx,
            "post",
            lambda *a, **k: httpx.Response(
                200, text=self._SAMPLE_HTML, request=httpx.Request("POST", websearch.SEARCH_URL)
            ),
        )
        text = websearch.search_and_describe("anything")
        assert "First Result Title" in text
        assert "{" not in text and "[" not in text

    def test_no_results_says_so_instead_of_an_empty_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            httpx,
            "post",
            lambda *a, **k: httpx.Response(
                200, text="<html></html>", request=httpx.Request("POST", websearch.SEARCH_URL)
            ),
        )
        text = websearch.search_and_describe("something obscure")
        assert "no results" in text.lower()
