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
# most those steps, never the run. PAUSE is set for the duration so the watchdog
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

enc() { python3 -c "import base64,pathlib,sys;print(base64.b64encode(pathlib.Path(sys.argv[1]).read_text().encode('utf-16-le')).decode())" "$1"; }
runps() { timeout 60 ssh -o BatchMode=yes box "powershell -NoProfile -EncodedCommand $(enc "$1")" 2>/dev/null; }

if [[ "$MODE" == status ]]; then
    runps "$OPS/probe-3090.ps1" | grep -aE "FREE=|UTIL=|RUNS="
    runps "$OPS/lane-a-settings.ps1" | grep -a "BATCH="
    exit 0
fi

echo "==> pausing the watchdog so the restart is not read as a crash"
touch "$OPS/PAUSE"

echo "==> setting lane A to --batch $BATCH --accum $ACCUM (effective batch stays 24)"
# Via the scheduled task's .bat, not by respawning a process: Neiro re-chains
# the task, so a hand-started replacement is overwritten within seconds.
timeout 120 ssh -o BatchMode=yes box \
  "powershell -NoProfile -Command \"\$env:MM_BATCH='$BATCH'; \$env:MM_ACCUM='$ACCUM'; & powershell -NoProfile -EncodedCommand $(enc "$OPS/lane-a-set-batch.ps1")\"" \
  2>/dev/null | grep -aE "RUNS=|NOW_BATCH=|ERR=|WARN="

echo "==> settling, then reporting"
sleep 60
runps "$OPS/probe-3090.ps1" | grep -aE "FREE=|RUNS="
runps "$OPS/lane-a-settings.ps1" | grep -a "BATCH="

rm -f "$OPS/PAUSE"
echo "==> watchdog re-armed"
