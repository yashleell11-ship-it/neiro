#!/usr/bin/env bash
# Keep a local port pointed at the desktop's LLM server.
#
# Ollama binds 127.0.0.1 on the desktop, which is right -- an inference server
# with no auth should not be listening on a LAN. So the port is forwarded
# rather than exposed.
#
# The LAN cable is tried first and tailscale second: the cable is
# sub-millisecond and does not depend on a coordination server, so when both
# are up it is strictly the better route. A dropped tunnel takes the attribution
# pass down with it, and a bare `ssh -N` gives no second chance, so this
# reconnects rather than exiting.
set -uo pipefail
PORT="${1:-11434}"
LOG=/home/yash/code/neiro/ops/llm-tunnel.log

while true; do
    for host in box-lan box; do
        # `echo`, not `true`: the desktop is WINDOWS, so a command runs in
        # cmd.exe, which has no `true` and exits 1 — a liveness probe that
        # always reported the host down.
        if timeout 12 ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" "echo ok" >/dev/null 2>&1; then
            printf '%s up via %s\n' "$(date -Is)" "$host" >>"$LOG"
            ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes \
                -o ServerAliveInterval=20 -o ServerAliveCountMax=3 \
                -L "${PORT}:127.0.0.1:${PORT}" "$host" >>"$LOG" 2>&1
            printf '%s dropped (%s)\n' "$(date -Is)" "$host" >>"$LOG"
            break
        fi
    done
    sleep 5
done
