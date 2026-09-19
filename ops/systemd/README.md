# Laptop training units

The laptop's half of "never stop training". Copies of what is installed at
`~/.config/systemd/user/`, kept here because a reboot-survival story that
lives only in an untracked dotfile is not a story.

    cp ops/systemd/*.service ops/systemd/*.timer ~/.config/systemd/user/
    systemctl --user daemon-reload
    systemctl --user enable --now gpu-watchdog.timer
    loginctl enable-linger yash     # user units without a login session

`gpu-watchdog.timer` is the only thing enabled. It is deliberately the ONLY
owner of when a lane starts: `elizabeth-persona-laptop.service` is installed
but NOT enabled, so nothing races it at boot.

Three things here were each a real failure first:

- **The unit is persistent, not `systemd-run --transient`.** A transient unit
  dies with the reboot and does not come back. That is exactly what happened
  on 2026-09-20 — the laptop rebooted mid-run at step 7880 and training simply
  stayed down until somebody looked.
- **The timer is `OnCalendar`, not `OnBootSec` + `OnUnitActiveSec`.** The
  latter pair reads like "ten minutes after boot, then every five" and is not:
  `OnBootSec` is monotonic, `Persistent=` applies only to calendar timers, and
  a timer enabled after its boot window has passed has no anchor — it shows
  `NEXT` as `-` and fires never. The ten-minute post-boot grace is enforced in
  `gpu-watchdog.sh` off `/proc/uptime`, where it cannot be lost.
- **`StartLimitIntervalSec=0`, and in `[Unit]`.** With `Restart=on-failure`,
  the default burst limit parks a fast-failing unit in `failed` and then
  refuses the watchdog's `systemctl start` as well — the one thing meant to
  recover it. In `[Service]` this build ignores the directive silently;
  `systemctl show -p StartLimitIntervalUSec` is what proves it took.

To stop: `touch ops/PAUSE` holds the lanes where they are; `touch ops/STOP`
disarms the timer.
