# Deploying the interlock server

Draft for the maintainer's review. One `geecs-interlocks` unit per
experiment: the main experiment's from `deploy/geecs-interlocks.service`,
an experiment instance's from `deploy/geecs-interlocks@.service`. Both are
templates rendered by `deploy/render_units.sh` from the host's
`/etc/geecs/site.env` (`docs/platform/site_profile.md`); every facility
value below is a placeholder.

## What the unit does

- **`Type=notify`, `WatchdogSec=10`.** The server says `READY=1` once its
  stream, listeners and page are up, and the evaluation loop feeds the
  watchdog after every evaluation that succeeds (`geecs_interlocks.notify`).
  A loop that stalls or keeps failing stops feeding it, and systemd kills
  and restarts the server. The stream itself already trips after one
  second without a new evaluation, so Master Control is never sent an old
  verdict while it waits.
- **`Restart=on-failure`, `RestartSec=2`.** Master Control treats the gap as
  unsafe and reconnects on its own.
- **Never `--arm-at-start`.** A restart would arm whatever settings arrive
  first, possibly one someone has just changed. The server refuses the flag
  under systemd (exit 2, `RestartPreventExitStatus=2`). After every start
  (and every reload) the settings checks are unarmed, so unsafe, until an
  operator arms them from the page.
- **Its own clone**, `<checkout root>/interlocks-checkout`: no other
  service's pull moves the code that reports to Master Control
  (`docs/platform/fleet_map.md` § "One clone per service").

## site.env

The `GEECS_INTERLOCKS_*` block of `deploy/site.env.example`:

| Key | Meaning |
|---|---|
| `GEECS_INTERLOCKS_CONFIG` | The configuration it runs: `interlocks/<name>.yaml` in the experiment's configs folder. Unset: not deployed (`bootstrap_host.sh` skips it) |
| `GEECS_INTERLOCKS_STATUS_HOST`, `_STATUS_PORT` | Master Control's stream. The port must match the GEECS DB's entry for the interlock server: Master Control reads the address and port from the DB |
| `GEECS_INTERLOCKS_WEB_HOST`, `_WEB_PORT` | The page. Not 8600 on a box that runs Canopy |
| `GEECS_INTERLOCKS_EXTERNAL_HOST` | Where the external (UDP) signals arrive: an interface of this host. The senders are repointed at it |
| `GEECS_INTERLOCKS_EXTRA_ARGS` | Extra flags (`--log-level debug`) |

The experiment is `GEECS_EXPERIMENT`. The configs repository comes from the
service account's `~/.config/geecs_python_api/config.ini` (`[Paths]`), as
for the other services that read it; the devices are reached through the
GEECS DB named there.

An instance's file `/etc/geecs/experiments/<Instance>.env` overrides
`GEECS_INTERLOCKS_CONFIG`, `_STATUS_PORT` and `_WEB_PORT`
(`deploy/experiments.env.example`: stream `5001+100N`, page `8650+N` for
slot N).

## First install (main experiment)

1. Set the `GEECS_INTERLOCKS_*` keys in `/etc/geecs/site.env`.
2. `deploy/bootstrap_host.sh --only interlocks` clones
   `interlocks-checkout`, installs its env, renders and prints the install
   lines (or by hand: `deploy/render_units.sh /etc/geecs/site.env <staging>
   GEECS-Interlocks/deploy/geecs-interlocks.service`).
3. `sudo systemctl enable --now geecs-interlocks`.
4. Point the GEECS DB's interlock-server entry at this host and the stream
   port, and repoint the UDP senders at `GEECS_INTERLOCKS_EXTERNAL_HOST`.
5. Check (below), then arm the settings checks from the page.

An experiment instance renders `geecs-interlocks@.service` the same way and
runs `systemctl enable --now geecs-interlocks@<Instance>`. Owed:
`bootstrap_host.sh --experiment` and `experiment.sh` do not know this
service yet.

## Check

- `systemctl status geecs-interlocks`: `active (running)`.
- `GET http://<host>:<web port>/healthz` → `{"status": "ok"}`; the page shows
  the configuration's name and a verdict.
- `poetry run geecs-interlocks listen <host>:<stream port>` from any clone
  prints the frames Master Control receives.
- `journalctl -u geecs-interlocks`: the start line names the file and the
  stream's address; every arm and reload names the operator.

## Upgrade

```bash
git -C <checkout root>/interlocks-checkout pull
cd <checkout root>/interlocks-checkout/GEECS-Interlocks && poetry install
sudo systemctl restart geecs-interlocks
```

A restart, like a reload, starts every interlock waiting and every settings
check unarmed: arm them again from the page.

## A new configuration

A configuration change merged in the configurator reaches the live
checkout; press **Reload** on the page (or restart the unit) to run it. A
file that does not validate is refused and the running interlocks are kept.
To switch to another configuration, change `GEECS_INTERLOCKS_CONFIG` and
restart.
