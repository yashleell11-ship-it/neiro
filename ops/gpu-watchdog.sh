#!/usr/bin/env bash
# Keep both GPUs working, chain to the next job when one finishes, and never
# stop on its own.
#
# Neiro trains on two cards: lane B (bilingual persona LoRA) on this laptop's
# 8 GB 5070, lane A (qwen3.5-4b persona LoRA, then English STT) on the
# desktop's 24 GB 3090 Ti. Both machines have now rebooted unexpectedly
# mid-run, so nothing here may depend on a process, a shell or a login
# surviving. Every restart costs at most `--save-every` steps, never the run.
#
# Four rules learned the hard way, each from this script getting it wrong:
#
#   1. A FAILED PROBE IS NOT A DEAD TRAINER. The first version sent
#      `nvidia-smi ...; echo; tasklist` over ssh to a Windows host, where cmd
#      does not take `;` as a separator -- so the whole thing became arguments
#      to nvidia-smi, the output held no "python.exe", and a healthy four-hour
#      run was declared dead. Restarts fire only on a probe that SUCCEEDED and
#      reported exactly zero trainers.
#   2. Count `persona_train`, not `python`. The desktop always has other
#      pythons; "any python is alive" masks a dead trainer forever.
#   3. A FINISHED JOB IS NOT A CRASH. The completion signal is the FINAL
#      adapter (`<out>/adapter/adapter_model.safetensors`), never the periodic
#      `checkpoint/` dir, which exists from step 25 onward and proves nothing.
#      Without that distinction a watchdog relaunches the run it just finished
#      with --resume and loops another whole epoch -- the exact opposite of
#      "move to the next job".
#   4. ONE OWNER PER LANE. This script used to restart lane B with `setsid
#      nohup`, which cannot survive a reboot and races the systemd unit that
#      now owns it. It starts the unit instead; `systemctl start` on a running
#      unit is a no-op, so the two can never both win.
#
# It does NOT stop by itself. An earlier version carried a deadline file and
# ran `systemctl --user disable --now` when it elapsed -- which is what it did
# at 06:23 on 2026-09-20, silently, while both lanes still had days of work
# left. Stopping is now a decision somebody makes, and it is per lane:
# `ops/PAUSE-laneA` holds the box, `ops/PAUSE-laneB` holds this laptop,
# `ops/PAUSE` holds both, `ops/STOP` disarms the timer completely.
#
# One check per invocation; the systemd --user timer supplies the cadence and
# the ten-minute delay after boot.

set -uo pipefail

OPS="/home/yash/code/neiro/ops"
LOG="$OPS/gpu-watchdog.log"
LOG_MAX=8000        # this runs forever now; the log may not grow forever
REPO="/home/yash/code/neiro"
TRAIN_DIR="$REPO/training"
LANE_B_UNIT="elizabeth-persona-laptop.service"
LANE_B_OUT="$REPO/models/persona-lora-bilingual"
LANE_B_ADAPTER="$LANE_B_OUT/adapter/adapter_model.safetensors"
LANE_B_CKPT="$LANE_B_OUT/checkpoint"
COOLDOWN=1800       # no lane may be restarted twice inside half an hour

# A timer tick overlapping a manual run is two watchdogs racing to restart the
# same lane. Take the lock or leave.
exec 9>"$OPS/.watchdog.lock"
flock -n 9 || exit 0

say() { printf '%s %s\n' "$(date -Is)" "$*" >>"$LOG"; }

if [[ -f "$LOG" ]] && (( $(wc -l <"$LOG") > LOG_MAX )); then
    tail -n $(( LOG_MAX / 2 )) "$LOG" >"$LOG.trim" && mv "$LOG.trim" "$LOG"
fi

# Restarts are rate-limited per lane, so a lane that dies on startup cannot be
# respawned every five minutes forever.
may_restart() {
    local stamp="$OPS/.restart-$1"
    [[ -f "$stamp" ]] && (( $(date +%s) - $(stat -c %Y "$stamp") < COOLDOWN )) && return 1
    touch "$stamp"; return 0
}

now=$(date +%s)

# Ten minutes of quiet after a reboot, by request. This lives here rather than
# in the timer because a monotonic OnBootSec= has no anchor once its window has
# passed and simply stops scheduling -- see the comment in gpu-watchdog.timer.
# Reading uptime is immune to that: it is true on every tick, forever.
BOOT_GRACE=600
uptime_s=$(cut -d. -f1 /proc/uptime)
if (( uptime_s < BOOT_GRACE )); then
    say "boot grace: up ${uptime_s}s of ${BOOT_GRACE}s -- letting the machine settle, no action"
    exit 0
fi

if [[ -f "$OPS/STOP" ]]; then
    say "STOP file present -- disarming timer. Remove ops/STOP and re-enable to resume."
    systemctl --user disable --now gpu-watchdog.timer >/dev/null 2>&1
    exit 0
fi

# Pausing is PER LANE, because the thing actually asked for is almost
# never "stop everything": it is "stop the box, leave the laptop
# running". An all-or-nothing PAUSE forces that to be done by editing
# the watchdog or disabling a scheduled task, which is how a machine
# ends up silently not coming back a week later.
pause_a=0; pause_b=0
[[ -f "$OPS/PAUSE" ]]       && { pause_a=1; pause_b=1; }
[[ -f "$OPS/PAUSE-laneA" ]] && pause_a=1
[[ -f "$OPS/PAUSE-laneB" ]] && pause_b=1

# --- lane B: this laptop --------------------------------------------------
read -r _ _ b_free b_util < <(
    nvidia-smi --query-gpu=memory.total,memory.used,memory.free,utilization.gpu \
               --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ','
)

lane_b_alive() { pgrep -f 'persona_train.py.*persona-lora-bilingual' >/dev/null 2>&1; }

if [[ -f "$LANE_B_ADAPTER" ]]; then
    # Rule 3: the persona LoRA finished. Hand off, do not relaunch.
    if [[ ! -f "$OPS/.laneB-smoke-done" ]]; then
        say "laneB  persona LoRA FINISHED (adapter saved) -- running the chat smoke test"
        ( cd "$REPO" && timeout 1800 uv run python scripts/chat_smoke_test.py \
            --checkpoint "$LANE_B_OUT/adapter" >"$OPS/lane-b-smoke.txt" 2>&1 )
        rc=$?
        touch "$OPS/.laneB-smoke-done"
        say "laneB  smoke test exit $rc -- transcript in ops/lane-b-smoke.txt"
        b_state="handed off to smoke test"
    else
        # Next job goes here. Until one is wired, say so plainly: an idle card
        # with no explanation reads as a crash, and somebody spends an hour
        # proving it was not.
        b_state="IDLE BY DESIGN -- persona done, smoke test done, no next lane-B job wired yet"
    fi
elif lane_b_alive; then
    age=-1
    [[ -e "$LANE_B_CKPT" ]] && age=$(( now - $(stat -c %Y "$LANE_B_CKPT") ))
    b_state="alive ckpt_age=${age}s"
elif (( pause_b )); then
    b_state="down (paused)"
elif may_restart laneB; then
    say "laneB  DOWN -- starting $LANE_B_UNIT"
    systemctl --user start "$LANE_B_UNIT" >/dev/null 2>&1
    sleep 12
    lane_b_alive && b_state="RESTARTED" \
                 || b_state="RESTART FAILED -- journalctl --user -u $LANE_B_UNIT"
else
    b_state="down, restart on cooldown"
fi
say "laneB  5070   free=${b_free:-?}MiB util=${b_util:-?}% $b_state"

# --- lane A: the 3090 Ti, over tailscale OR the LAN ------------------------
enc() { python3 -c "import base64,pathlib,sys;print(base64.b64encode(pathlib.Path(sys.argv[1]).read_text().encode('utf-16-le')).decode())" "$1"; }

# Two routes to the same machine, tried in order. On 2026-09-20 the box's
# TAILSCALE link dropped while the machine itself stayed up and answered on
# the LAN in 0.37 ms -- `ssh box` timed out, and a watchdog with one route
# would have logged PROBE FAILED every five minutes all night and never
# touched a perfectly healthy 24 GB card. Which route worked is logged,
# because "reachable only on the LAN" is a fact about the network worth
# seeing, not an implementation detail to paper over.
box_host() {
    local host
    for host in box box-lan; do
        if timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" "exit 0" >/dev/null 2>&1; then
            printf '%s' "$host"; return 0
        fi
    done
    return 1
}

BOX=$(box_host) || BOX=""
if [[ -z "$BOX" ]]; then
    pc=""
else
    [[ "$BOX" == "box-lan" ]] && say "laneA  reachable on the LAN only -- tailscale is down on the box"
    pc=$(timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=30 "$BOX" \
            "powershell -NoProfile -EncodedCommand $(enc "$OPS/probe-3090.ps1")" 2>/dev/null)
fi
a_free=$(grep -ao 'FREE=[0-9]*'  <<<"$pc" | head -1 | cut -d= -f2)
a_util=$(grep -ao 'UTIL=[0-9]*'  <<<"$pc" | head -1 | cut -d= -f2)
a_num=$( grep -ao 'RUNS=[0-9]*'  <<<"$pc" | head -1 | cut -d= -f2)
a_procs=$(grep -ao 'PROCS=[0-9]*' <<<"$pc" | head -1 | cut -d= -f2)

if [[ -z "$a_num" ]]; then
    # Unreachable, asleep, or the probe itself broke. Explicitly NOT a dead
    # trainer -- rule 1, the case that caused a spurious restart before.
    say "laneA  3090Ti PROBE FAILED on both routes (tailscale AND LAN) -- desktop asleep, off, or ssh refused. No action"
elif (( a_num > 0 )); then
    flag=""; (( pause_a )) && flag=" (PAUSED but still running)"
    say "laneA  3090Ti free=${a_free:-?}MiB util=${a_util:-?}% alive runs=$a_num procs=${a_procs:-?}$flag"
    if [[ -n "$a_util" ]] && (( a_util < 40 )); then
        say "laneA  WARN util ${a_util}% with a live trainer -- stalled or data-starved"
    fi
    if (( a_num > 1 )); then
        say "laneA  WARN $a_num concurrent runs -- duplicates corrupt a shared --out"
    fi
elif (( pause_a )); then
    say "laneA  3090Ti free=${a_free:-?}MiB down (paused)"
elif may_restart laneA; then
    # box_next.bat, NOT a direct trainer launch. The box's dispatcher is the
    # one place that knows whether persona is finished and English STT is what
    # comes next; launching persona_train.py from here would relaunch a job
    # that had already completed (rule 3).
    say "laneA  DOWN (probe confirmed 0 runs) -- running the box dispatcher over $BOX"
    timeout 90 ssh -o BatchMode=yes "$BOX" "D:\\neiro-data\\box_next.bat" >/dev/null 2>&1
    sleep 20
    again=$(timeout 60 ssh -o BatchMode=yes "$BOX" \
        "powershell -NoProfile -EncodedCommand $(enc "$OPS/probe-3090.ps1")" 2>/dev/null \
        | grep -ao 'RUNS=[0-9]*' | head -1 | cut -d= -f2)
    say "laneA  dispatcher run; runs now=${again:-unknown}"
else
    say "laneA  3090Ti down, restart on cooldown"
fi

# A healthy run must exit 0. Without this the script inherited the status of
# whatever test ran last -- a false `(( a_num > 1 ))` is exit 1 -- so systemd
# marked every successful check as Failed, which is exactly the signal that
# has to stay meaningful for a watchdog.
exit 0
