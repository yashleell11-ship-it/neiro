#!/usr/bin/env bash
# Which name actually reaches the 3090 Ti right now. Source this, then call
# `box_host`; it prints `box` or `box-lan`, or fails if neither answers.
#
# There are two routes to one machine and they fail independently. On
# 2026-09-20 the box's TAILSCALE link went down while the machine itself was
# perfectly healthy -- `ssh box` timed out, `tailscale status` said
# "yashpc ... offline, last seen 2h ago", and the box answered on the LAN in
# 0.37 ms with a training run on the card. Every script here that hardcoded
# `ssh box` was blind to a working desktop, which is the expensive version of
# "a failed probe is not a dead trainer".
#
# One copy, because there were already two scripts with their own ssh calls
# and a third would have been a third thing to remember to fix.

box_host() {
    local host
    for host in box box-lan; do
        if timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" "exit 0" >/dev/null 2>&1; then
            printf '%s' "$host"; return 0
        fi
    done
    return 1
}
