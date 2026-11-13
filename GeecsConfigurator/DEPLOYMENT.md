# GeecsConfigurator — the configs-root switch

The runbook for moving the Linux services' configs root from the share's
clone to the configurator's live checkout (ruling 4 on #1307), and for the
applies that switch turns on. Every step is the maintainer's: it needs
`sudo`, a unit restart, a `site.env` edit, or his own git credential. Values
in angle brackets are the site's; nothing here names a host or an account.

After the switch:

- the Linux services (worker, scanner, portal, Canopy, Control Room, the
  archiver's CLI) read `<STATE_DIR>/live`, which holds **accepted commits of
  main only** and moves only by the configurator's fast-forward;
- the share's clone (`GEECS_CONFIGURATOR_SHARE_CLONE`) is fast-forwarded by
  the configurator after every accepted change, so LabVIEW keeps reading the
  file it always read; it is **read-only for humans and services**;
- the worker writes its one file (`shot_offsets.yaml`) into the inbox
  (`GEECS_CONFIGS_WRITE_ROOT`), and the configurator commits it;
- the scanner saves presets through the configurator (`--configurator-url`);
- the portal's analysis config editor is read-only until slice 6 moves the
  analysis kinds onto the configurator.

`<STATE_DIR>` is `GEECS_CONFIGURATOR_STATE_DIR` (the unit's
`StateDirectory`, `/var/lib/geecs-configurator`).

## Preconditions

1. The configurator runs with the commit path on (`STATE_DIR`, `ORIGIN`, the
   GitHub App) and its box acceptance on #1320 passed; `/healthz` reads
   `"main": "applied"` and `"share": "current"`.
2. `GEECS_CONFIGURATOR_SHARE_CLONE` was set **before the configurator's first
   start**, so its live checkout started at the share's commit (else it
   started *unverified*: `/healthz` says so, and the maintainer looks at
   main and deletes `<STATE_DIR>/UNVERIFIED` before going on).
3. The lab is quiet: nothing runs or waits on the worker (the switch restarts
   the worker, the scanner, the portal and Canopy).
4. The trigger-profile catalogue is in the new schema, or the maintainer has
   said the resolver's legacy converter stays (#1307, his own items).

## 1. The share's edits, committed once (his credential)

The share's clone may hold edits made by hand or by the old writers (the
scanner's presets, the worker's shot offsets, the portal's editor). They are
committed once, before anything fast-forwards it:

```
cd "<the share's configs clone>"
git status --short
git add <each file to keep>
git commit -m "Edits on the share before the configurator switch"
git push origin main
git status --short          # must print nothing
```

Then wait for the configurator's next fetch (`GEECS_CONFIGURATOR_FETCH_SECONDS`,
60 s by default) and check `/healthz`: `"main": "applied"`, `"share": "current"`.

## 2. site.env

`sudoedit /etc/geecs/site.env`, then set (keep each line's other flags):

```
GEECS_CONFIGURATOR_SHARE_CLONE=<the share's configs clone, the old GEECS_CONFIGS_ROOT>
GEECS_CONFIGS_ROOT=<STATE_DIR>/live
GEECS_CONFIGS_WRITE_ROOT=<STATE_DIR>/inbox
GEECS_CONFIGURATOR_SCANNER_URL=<experiment>=http://127.0.0.1:<its scanner's port>,<instance experiment>=http://127.0.0.1:<its scanner's port>
GEECS_CONFIGURATOR_ARCHIVER_COMMAND=<checkout root>/archiver-checkout/GeecsArchiver/.venv/bin/geecs-archiver
GEECS_SCANNER_EXTRA_ARGS=<as before> --configurator-url http://127.0.0.1:<GEECS_CONFIGURATOR_PORT>
GEECS_PORTAL_EXTRA_ARGS=<as before> --config-editor-read-only
```

`GEECS_CONFIGURATOR_SCANNER_URL` names each experiment's scanner (an instance's
scanner is on its slot's port, `deploy/experiment.sh list`): a document of an
experiment with no scanner named is never applied by another's reopen; its row
stays pending, named. One plain URL is the served experiment's alone.

The worker, the scanner and the configurator run as the one service account,
which reads `<STATE_DIR>` (mode 0755) and writes the inbox (the configurator
creates each experiment's folder there at its start; a new experiment needs
a configurator restart).

## 3. The rendered config.ini (by hand)

The bootstrap writes the service account's
`~/.config/geecs_python_api/config.ini` only when it is absent, so edit it:

```
[Paths]
scanner_config_root_path = <STATE_DIR>/live
scan_analysis_configs_path = <STATE_DIR>/live/scan_analysis_configs
```

## 4. Units and Canopy's compose, re-rendered

```
cd <checkout root>/qs-checkout
deploy/render_units.sh /etc/geecs/site.env ~/deploy-staging
GeecsCanopy/deploy/render_conf.sh /etc/geecs/site.env ~/deploy-staging/canopy
```

Install with the `sudo` lines each prints. Canopy's compose now mounts the
live checkout's own git dir (`<STATE_DIR>/repo.git/worktrees/live`) read-only
at the same path, so its readers see a fast-forward's rewrite lock and never
read the device registry or the operator policy as missing mid-rewrite.

## 5. Restart, in order

```
sudo systemctl restart geecs-configurator
sudo systemctl restart geecs-qserver geecs-qserver-ready geecs-scanner geecs-data-portal geecs-logbook geecs-canopy geecs-control-room
```

(Experiment instances: their `@<Instance>` units too, the logbook's included.)

## 6. Check

- `curl -s http://127.0.0.1:<GEECS_CONFIGURATOR_PORT>/healthz` →
  `"main": "applied"`, `"share": "current"`.
- The scanner's `/api/status` → `readiness: "ready"`.
- `git -C <STATE_DIR>/live status --short` prints nothing, now and after a
  scan.
- The portal's `/configs/` page says read-only; a save there answers 405.

## The acceptance this switch enables (OWED, on the PulsedWire bench instance)

Laser off, nothing of the instance running or queued; never an Undulator
device. From a browser on the configurator's page:

1. Open the laser-off trigger profile, change one amplitude, enter the
   operator's name, Save. Expected: the log (`/log`) shows a `saved` row by
   "<name> via configurator", state **queued**; `git -C <the share's configs
   clone> log -1` shows the same commit within seconds.
2. Within `GEECS_CONFIGURATOR_APPLY_SECONDS` (30 s): the scanner's chip reads
   "reopening", then ready; the row reads **applied** ("the worker
   environment reopened (… s)").
3. Run the laser-off scan. Expected: the trigger box fires at the new
   amplitude (its `:SP` readback in the scan's s-file, or a `caget` of the
   gateway PV).
4. The refusal: start a scan, then save another amplitude change. Expected:
   the row stays **queued** ("waits for the worker idle and empty") until the
   scan ends, then applies.

## After the switch

- **Nothing writes the live checkout.** An untracked or edited file there
  stops its every fast-forward (`/healthz` and `/log` say so) until it is
  removed by hand. A new experiment's folder goes into the configs repo by
  pull request; then restart `geecs-configurator` (it creates the inbox
  folder at start). `deploy/experiment.sh up` refuses to scaffold into the
  live checkout and says so.
- **A save during an apply pass is answered busy** (503, `Retry-After: 60`):
  the pass holds the repository while the worker reopens (a minute or
  more). Nothing was saved; save again. The scanner's Save as preset shows
  the same message.

## Rollback

Put back `GEECS_CONFIGS_ROOT` (the share's clone), empty
`GEECS_CONFIGS_WRITE_ROOT`, drop `--configurator-url` and
`--config-editor-read-only`, restore `config.ini`'s two paths, re-render and
restart as in steps 4 and 5. The share's clone is current with main, so
nothing is lost; presets then cannot be saved from the scanner until the
switch is redone.
