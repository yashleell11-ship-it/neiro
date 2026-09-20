#!/usr/bin/env bash
# Make room on the 3090 Ti for the TTS model, without changing what lane A learns.
#
#   ./lane-a-vram.sh shrink    batch 12 accum 2  ->  batch 6 accum 4
#   ./lane-a-vram.sh restore   batch 6  accum 4  ->  batch 12 accum 2
#   ./lane-a-vram.sh status
#
# EFFECTIVE BATCH IS UNCHANGED: 12x2 and 6x4 are both 24 samples per optimizer
# step, so the gradient is the same gradient and the run is the same run. Only
# the activation memory halves, because activations scale with the micro-batch
# and not with the accumulation count. This is the reason to do it this way
# rather than by lowering the batch alone, which would quietly change the
# training dynamics of a job that is hours in.
#
# Expected: ~17.4 GB held -> ~12.7 GB, i.e. roughly 6.9 GB free -> ~11.6 GB.
# IndexTTS-2.5 wants about 6 GB, and the render worker's own rule is to claim
# only at >=8 GB free -- 6.9 GB is inside the band where it refuses to start, so
# without this the TTS model simply never runs.
#
# Lane A carries --resume and checkpoints every 25 steps, so the switch costs at
# most those steps, never the run. PAUSE-laneA is set for the duration so the watchdog
# does not read the gap as a crash and start a second trainer on the same --out.

set -uo pipefail
OPS="/home/yash/code/neiro/ops"
MODE="${1:-status}"

case "$MODE" in
  shrink)  BATCH=6;  ACCUM=4 ;;
  restore) BATCH=12; ACCUM=2 ;;
  status)  BATCH=""; ACCUM="" ;;
  *) echo "usage: $0 {shrink|restore|status}" >&2; exit 2 ;;
esac

enc() { python3 "$OPS/psenc.py" "$1"; }

# Tailscale and the LAN fail independently; try both. Without this, `status`
# printed NOTHING AT ALL the evening tailscale went down on an otherwise
# healthy box -- no output, exit 0, nothing to tell you which it was.
source "$OPS/box-host.sh"
BOX=$(box_host) || { echo "box unreachable on both tailscale and the LAN" >&2; exit 1; }
[[ "$BOX" == "box-lan" ]] && echo "(reaching the box on the LAN; tailscale is down)" >&2

runps() {
    local payload
    payload=$(enc "$1") || { echo "cannot encode $1 -- see psenc.py" >&2; return 1; }
    timeout 60 ssh -o BatchMode=yes "$BOX" \
        "powershell -NoProfile -EncodedCommand $payload" 2>/dev/null
}

if [[ "$MODE" == status ]]; then
    runps "$OPS/probe-3090.ps1" | grep -aE "FREE=|UTIL=|RUNS="
    runps "$OPS/lane-a-settings.ps1" | grep -a "BATCH="
    exit 0
fi

echo "==> pausing lane A so the restart is not read as a crash"
# PAUSE-laneA, not PAUSE. This script only touches the box, and the
# all-lanes flag would stop the LAPTOP's trainer too -- silently, for the
# duration, for a change that has nothing to do with it.
touch "$OPS/PAUSE-laneA"

echo "==> setting lane A to --batch $BATCH --accum $ACCUM (effective batch stays 24)"
# Via the scheduled task's .bat, not by respawning a process: Neiro re-chains
# the task, so a hand-started replacement is overwritten within seconds.
timeout 120 ssh -o BatchMode=yes "$BOX" \
  "powershell -NoProfile -Command \"\$env:MM_BATCH='$BATCH'; \$env:MM_ACCUM='$ACCUM'; & powershell -NoProfile -EncodedCommand $(enc "$OPS/lane-a-set-batch.ps1")\"" \
  2>/dev/null | grep -aE "RUNS=|NOW_BATCH=|ERR=|WARN="

echo "==> settling, then reporting"
sleep 60
runps "$OPS/probe-3090.ps1" | grep -aE "FREE=|RUNS="
runps "$OPS/lane-a-settings.ps1" | grep -a "BATCH="

rm -f "$OPS/PAUSE-laneA"
echo "==> watchdog re-armed"
