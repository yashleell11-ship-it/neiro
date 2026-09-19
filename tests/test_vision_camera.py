"""The camera layer, tested without a camera.

The load-bearing test here is `pin_dynamic_framerate_off` never raising.
It shells out to `v4l2-ctl`, which on some machines is absent, on some
drivers lacks the control, and on some devices simply refuses the write
— and the correct response to all three is a degraded camera and a note,
not a dead daemon. A webcam running at 10 fps instead of 28 is worse
hand tracking; a webcam that refuses to start is no hand tracking.

Why this control is worth a module and a test file at all: measured on
this laptop, leaving it at its default of 1 pins capture to exactly
100.0 ms per frame in a dark room — identical across MJPG/YUYV and
480p/720p, which is what proved it was exposure and not bandwidth. See
docs/DECISIONS.md, 2026-09-19.
"""

from __future__ import annotations

import subprocess

import pytest

from elizabeth.config import Elizabeth
from elizabeth.vision import camera as cam


class TestPinDynamicFramerate:
    def test_reports_ok_when_v4l2_ctl_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cam.shutil, "which", lambda _: "/usr/bin/v4l2-ctl")
        monkeypatch.setattr(
            cam.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout="", stderr=""),
        )
        ok, detail = cam.pin_dynamic_framerate_off(0)
        assert ok is True
        assert cam.DYNAMIC_FRAMERATE_CONTROL in detail

    def test_a_missing_v4l2_ctl_is_reported_not_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cam.shutil, "which", lambda _: None)
        ok, detail = cam.pin_dynamic_framerate_off(0)
        assert ok is False
        assert "v4l2-ctl" in detail

    def test_a_driver_that_refuses_the_control_is_reported_not_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Some drivers simply do not expose exposure_dynamic_framerate.
        monkeypatch.setattr(cam.shutil, "which", lambda _: "/usr/bin/v4l2-ctl")
        monkeypatch.setattr(
            cam.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(
                a[0], 1, stdout="", stderr="unknown control 'exposure_dynamic_framerate'\n"
            ),
        )
        ok, detail = cam.pin_dynamic_framerate_off(0)
        assert ok is False
        assert "unknown control" in detail

    def test_an_os_error_is_reported_not_raised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cam.shutil, "which", lambda _: "/usr/bin/v4l2-ctl")

        def boom(*a: object, **k: object) -> None:
            raise OSError("no such device")

        monkeypatch.setattr(cam.subprocess, "run", boom)
        ok, detail = cam.pin_dynamic_framerate_off(0)
        assert ok is False
        assert "OSError" in detail

    def test_a_timeout_is_reported_not_raised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A wedged USB device can hang the ioctl; the daemon must not
        # inherit that hang.
        monkeypatch.setattr(cam.shutil, "which", lambda _: "/usr/bin/v4l2-ctl")

        def hang(*a: object, **k: object) -> None:
            raise subprocess.TimeoutExpired(cmd="v4l2-ctl", timeout=5)

        monkeypatch.setattr(cam.subprocess, "run", hang)
        ok, detail = cam.pin_dynamic_framerate_off(0)
        assert ok is False
        assert "TimeoutExpired" in detail

    def test_the_device_index_reaches_the_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A two-camera machine must not silently configure video0 while
        # capturing from video2.
        seen: list[list[str]] = []
        monkeypatch.setattr(cam.shutil, "which", lambda _: "/usr/bin/v4l2-ctl")

        def record(argv: list[str], **k: object) -> subprocess.CompletedProcess:
            seen.append(argv)
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

        monkeypatch.setattr(cam.subprocess, "run", record)
        cam.pin_dynamic_framerate_off(2)
        assert "/dev/video2" in seen[0]
        assert f"--set-ctrl={cam.DYNAMIC_FRAMERATE_CONTROL}=0" in seen[0]


class TestCameraConfig:
    def test_the_camera_is_off_by_default(self) -> None:
        # The one sensor that can see the room does not turn itself on
        # because a config file happened to exist.
        assert Elizabeth().camera.enabled is False

    def test_the_measured_defaults_are_the_defaults(self) -> None:
        camcfg = Elizabeth().camera
        assert camcfg.pin_dynamic_framerate_off is True, (
            "leaving this off costs ~64 ms per frame in a dark room — see "
            "docs/DECISIONS.md, 2026-09-19"
        )
        assert (camcfg.width, camcfg.height) == (640, 480)
        assert camcfg.buffer_size == 1, "a queued frame is latency, not data"
