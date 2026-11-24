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
- **A start it cannot fix is refused, exit 2.** A missing or invalid
  configuration file, a blank configuration name, a bad flag: the unit stops
  in a plain `failed` state instead of restarting (`systemctl status` and the
  journal say why). Master Control sees no stream, which is unsafe. Fix the
  cause, then `systemctl restart`.
- **The page's actions have no login.** Anyone who reaches the page and
  types a name can arm or reload; arming after a setting moved makes the
  moved value the one held. The actions accept only a JSON body, so another
  web page cannot fire them from a browser. On a box with one interface and
  no front door, choose `GEECS_INTERLOCKS_WEB_HOST`: the lab address (open
  to the subnet) or `127.0.0.1`, reached through an SSH tunnel
  (`ssh -L 8650:127.0.0.1:8650 <box>`). Read the journal's operator lines
  after an incident.
- **Every external signal names its sender.** Without `sender`, a packet
  from any host on the network is read, so a stray "safe" can clear what
  the real sender reported unsafe. Set each `external` check's `sender` to
  the sending computer's IP address.
- **Its own clone**, `<checkout root>/interlocks-checkout`: no other
  service's pull moves the code that reports to Master Control
  (`docs/platform/fleet_map.md` § "One clone per service").

## site.env

`bootstrap_host.sh` refuses to run when any of `CONFIG`, the two hosts, the
two ports or `EXTERNAL_HOST` is blank.

The `GEECS_INTERLOCKS_*` block of `deploy/site.env.example`:

| Key | Meaning |
|---|---|
| `GEECS_INTERLOCKS_CONFIG` | The configuration it runs: `interlocks/<name>.yaml` in the experiment's configs folder. Unset: not deployed (`bootstrap_host.sh` skips it) |
| `GEECS_INTERLOCKS_STATUS_HOST`, `_STATUS_PORT` | Master Control's stream. The port must match the GEECS DB's entry for the interlock server: Master Control reads the address and port from the DB |
| `GEECS_INTERLOCKS_WEB_HOST`, `_WEB_PORT` | The page. Not 8600 on a box that runs Canopy |
| `GEECS_INTERLOCKS_EXTERNAL_HOST` | Where the external (UDP) signals arrive: an interface of this host. The senders are repointed at it |
| `GEECS_INTERLOCKS_EXTRA_ARGS` | Extra flags (`--log-level debug`). A flag the server does not know exits 2 and the unit stays down |

The experiment is `GEECS_EXPERIMENT`. The configs repository comes from the
service account's `~/.config/geecs_python_api/config.ini` (`[Paths]`), as
for the other services that read it; the devices are reached through the
GEECS DB named there.

An instance's file `/etc/geecs/experiments/<Instance>.env` carries
`GEECS_INTERLOCKS_INSTANCE_CONFIG`, `_INSTANCE_STATUS_PORT` and
`_INSTANCE_WEB_PORT` (`deploy/experiments.env.example`: stream `5001+100N`,
page `8650+N` for slot N): names site.env never sets, so an instance file
without them makes the instance unit exit 2 rather than run on the main
experiment's configuration and stream port.

## First install (main experiment)

0. A configuration exists: `interlocks/<name>.yaml` merged into the
   experiment's folder of the configs repo through the configurator, every
   `external` check with its `sender`.
1. Set the `GEECS_INTERLOCKS_*` keys in `/etc/geecs/site.env`.
2. `deploy/bootstrap_host.sh /etc/geecs/site.env --only interlocks` clones
   `interlocks-checkout` at `--ref` (default `master`; until GEECS-Interlocks
   is promoted to `master`, pass `--ref <branch that carries it>`), installs
   its env, renders and prints the install
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
- `GET http://<host>:<web port>/healthz` → `{"status": "ok"}` while it
  evaluates; 503 with a `reason` when the loop has stalled or the
  configuration holds no interlock. The page shows the configuration's
  name and a verdict.
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
