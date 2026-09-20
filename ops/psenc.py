#!/usr/bin/env python3
"""Encode a PowerShell script for `powershell -EncodedCommand`, without the
comments, and refuse rather than truncate when it still will not fit.

`-EncodedCommand` is how this project talks to the box at all: cmd.exe
mangles quoting badly enough over ssh that passing a script inline is not
reliable. But the encoding is base64 of UTF-16LE, which is **2.67 bytes
of command line per byte of script**, and cmd.exe's command line stops at
**8191 characters**.

That limit was hit on 2026-09-20 by writing a better comment. probe-3090.ps1
gained a paragraph explaining why it checks three conditions instead of one;
the file went to ~3.2 KB, the encoded command to **8744 characters**, and the
box replied "The command line is too long." The watchdog's own logs then
read, one second apart:

    laneA  reachable on the LAN only -- tailscale is down on the box
    laneA  3090Ti PROBE FAILED on both routes

— connected fine, and reported the machine unreachable. A documentation
change silently disabled the monitoring.

So: comments are stripped before encoding (8744 -> 1600 for that file), which
means the explanation can live in the script where it belongs instead of
being something you must keep short to stay under a limit nobody sees. And
what does not fit is an error with the numbers in it, never a truncation and
never a silent empty string.
"""

from __future__ import annotations

import base64
import pathlib
import sys

# cmd.exe's hard limit, less room for `ssh host "powershell -NoProfile
# -EncodedCommand "` and the shell quoting around it.
CMD_LIMIT = 8191
HEADROOM = 512


def encode(path: pathlib.Path) -> str:
    # Whole-line comments only. A '#' inside a string or a regex -- and this
    # project's probes contain both, e.g. '^(python|uv)' -- must survive.
    code = "\n".join(
        line for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    return base64.b64encode(code.encode("utf-16-le")).decode()


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: psenc.py <script.ps1>", file=sys.stderr)
        return 2
    path = pathlib.Path(sys.argv[1])
    if not path.exists():
        print(f"psenc: no such script: {path}", file=sys.stderr)
        return 1
    encoded = encode(path)
    if len(encoded) > CMD_LIMIT - HEADROOM:
        print(
            f"psenc: {path.name} encodes to {len(encoded)} chars, over the "
            f"{CMD_LIMIT - HEADROOM} usable of cmd.exe's {CMD_LIMIT}. Shorten the "
            "CODE (comments are already stripped), or copy the script to the box "
            "and run it with -File instead.",
            file=sys.stderr,
        )
        return 1
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
