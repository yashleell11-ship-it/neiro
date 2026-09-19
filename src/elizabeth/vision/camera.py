"""Webcam capture, and the one v4l2 control that decides whether v2 is
usable after dark.

**The measurement this module exists to encode.** Hand tracking's cost is
not the inference — MediaPipe's hand landmarker runs p50 10.2 ms on this
CPU. It is the capture. `cap.read()` measured **p50 89.8-100.0 ms** here,
about 10 fps against a device that advertises 30, and the number was
*exactly* 100.0 ms across MJPG and YUYV, 640x480 and 1280x720 alike.
Identical timing across formats and resolutions rules out USB bandwidth:
a 720p uncompressed stream cannot cost the same as a 480p compressed one
by accident. The cause is the v4l2 control `exposure_dynamic_framerate`,
which defaults to **1** on this webcam and lets the driver trade frame
rate for exposure in low light — a dark room at 05:00 pins it to exactly
1/10 s. Setting it to 0 gives **p50 36.3 ms (~28 fps)**.

So `open_camera` pins that control, and the whole reason is written down
here rather than living as a bare `subprocess.run` nobody dares delete.
Full numbers in docs/DECISIONS.md, 2026-09-19.

**Why v4l2-ctl and not OpenCV.** `cv2.CAP_PROP_EXPOSURE` and friends map
onto a different, narrower set of controls and cannot reach
`exposure_dynamic_framerate` at all on this driver. The control is set
out-of-band, before the device is opened, and a failure to set it is
reported rather than raised — a camera that runs at 10 fps is degraded,
not broken, and refusing to start would be the worse outcome.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from elizabeth.config import CameraConfig, Elizabeth

# v4l2 control names this module touches. Named here rather than inline
# so a driver that spells them differently fails in one readable place.
DYNAMIC_FRAMERATE_CONTROL = "exposure_dynamic_framerate"


@dataclass(frozen=True)
class CameraOpenResult:
    """What actually happened when the camera was opened.

    The requested settings and the granted ones are kept apart on
    purpose: v4l2 silently substitutes a size or rate it prefers, and a
    pipeline that assumed it got what it asked for is how "why is this
    slow at night" becomes a three-hour investigation. `notes` carries
    anything degraded-but-working, which the caller logs rather than
    dies on.
    """

    width: int
    height: int
    fps: float
    fourcc: str
    dynamic_framerate_pinned: bool
    notes: tuple[str, ...] = ()


def pin_dynamic_framerate_off(device_index: int) -> tuple[bool, str]:
    """Turn `exposure_dynamic_framerate` off. `(ok, detail)`.

    Never raises. A machine without `v4l2-ctl`, a driver without this
    control, or a device that refuses the write all mean the same thing
    to the caller: capture may run at a third of the expected rate in
    poor light, and that is worth a line in the log, not a crash.
    """
    binary = shutil.which("v4l2-ctl")
    if binary is None:
        return False, "v4l2-ctl not installed — cannot pin exposure_dynamic_framerate"
    try:
        proc = subprocess.run(
            [
                binary,
                "-d",
                f"/dev/video{device_index}",
                f"--set-ctrl={DYNAMIC_FRAMERATE_CONTROL}=0",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # noqa: BLE001
        return False, f"v4l2-ctl failed: {type(exc).__name__}"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return False, f"v4l2-ctl exited {proc.returncode}: {detail[0] if detail else 'no output'}"
    return True, f"{DYNAMIC_FRAMERATE_CONTROL}=0"


def _fourcc_of(cap: object) -> str:
    """The pixel format the device actually granted, as its four
    characters. What was ASKED for is frequently not what arrives.
    """
    import cv2

    raw = int(cap.get(cv2.CAP_PROP_FOURCC))  # type: ignore[attr-defined]
    return "".join(chr((raw >> (8 * i)) & 0xFF) for i in range(4))


@contextmanager
def open_camera(cfg: Elizabeth | None = None) -> Iterator[tuple[object, CameraOpenResult]]:
    """Open the webcam with this project's settings, yield
    `(capture, what_we_actually_got)`, and always release it.

    The release matters more than usual here: a webcam left open keeps
    its LED lit, which in a system that promises the camera is off when
    it is off is not a cosmetic bug.
    """
    import cv2

    cfg = cfg or Elizabeth()
    cam: CameraConfig = cfg.camera
    notes: list[str] = []

    pinned = False
    if cam.pin_dynamic_framerate_off:
        pinned, detail = pin_dynamic_framerate_off(cam.device_index)
        if not pinned:
            notes.append(f"capture may be slow in low light: {detail}")

    # CAP_V4L2 explicitly: the default backend on this machine has picked
    # a different one before, and the controls below only mean anything
    # to v4l2.
    cap = cv2.VideoCapture(cam.device_index, cv2.CAP_V4L2)
    try:
        if not cap.isOpened():
            raise RuntimeError(
                f"camera /dev/video{cam.device_index} would not open — "
                "is it in use by another process, or is the index wrong?"
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.height)
        cap.set(cv2.CAP_PROP_FPS, cam.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, cam.buffer_size)

        granted = CameraOpenResult(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            fourcc=_fourcc_of(cap),
            dynamic_framerate_pinned=pinned,
            notes=tuple(notes),
        )
        yield cap, granted
    finally:
        cap.release()
