# Capture daemon deployment runbook

The central PVA capture daemon (`geecs_bluesky.capture`, CLI
`geecs-capture-daemon`) runs as a systemd service **on the queueserver
worker host** — that co-location is a requirement, not a preference: the
start document's save paths are composed on the worker's filesystem view,
and the liveness heartbeat (`capture/heartbeat.py`) that gates toggle-off
scans is a file the engine reads locally. When the services box migrates,
the worker and the daemon move together.

## Install (once per host)

1. The host already has the GEECS-Plugins checkout and the worker's poetry
   env (`qserver/deploy/DEPLOYMENT.md`). Add the capture extra:

   ```bash
   cd /opt/geecs/GEECS-Plugins/GeecsBluesky
   poetry install --extras "ca qserver tiled capture"
   ```

   (`capture` = p4p + h5py + pyzmq. Keep whatever extras the worker
   already uses — this env is shared.)

2. `geecs-capture.service` is a template rendered from the host's
   `site.env` by `deploy/render_units.sh` (same file, same values as the
   qserver unit — the experiment and the document-stream address
   `GEECS_QS_DOC_ADDR` come from `site.env`; see the
   [Site Profile](../../../docs/platform/site_profile.md)), then:

   ```bash
   sudo install -m 0644 ~/deploy-staging/geecs-capture.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now geecs-capture
   ```

3. Verify: `journalctl -u geecs-capture -n 5` shows the discovery line
   ("capture discovery: N eligible cameras") and the running banner; the
   heartbeat file appears at `~/.local/state/geecs-capture/heartbeat.json`
   **in the service user's home** — under sudo that is NOT your own `~`
   (override: `[capture] heartbeat_path` in the shared config.ini, absolute
   paths only) — and its timestamp refreshes every ~10 s. On a clean stop
   the daemon removes the file, so toggle-off scans are refused
   immediately, not after the 30 s stale window.

## Operational contract

- **Pure consumer**: subscribes (doc stream + PVA) and writes stacks;
  commands nothing except the eager `save="off"` writes the *engine* plans
  for capture-owned cameras. If the daemon dies, scans and native saving
  are unaffected — but **toggle-off scans are refused pre-claim** by the
  engine's liveness preflight until it is back (that is the safety design,
  not a malfunction).
- One reconciliation log line per scan per camera (`journalctl`); the
  counter identity is documented in `geecs_bluesky/capture/FORMAT.md`.
- Restarting is always safe between scans (`systemctl restart
  geecs-capture`). A restart mid-scan loses that scan's capture; on a
  **dual-write** scan the native files are untouched, but on a
  **toggle-off** scan there are no native files — a mid-scan restart
  loses those images outright, so never restart while a toggle-off scan
  is running. (A daemon killed without cleanup — SIGKILL, power loss —
  leaves its heartbeat behind for up to 30 s, during which a toggle-off
  submission would still pass preflight; a clean stop removes it.)
- **Restart the daemon after camera roster changes** (devices added to the
  experiment, devicetype fixes): targets are discovered at startup, the
  daemon errors loudly on engine-listed devices it has no target for, and
  the engine's toggle-off preflight refuses scans whose capture devices
  are missing from the heartbeat's roster.
- The Phase-6 evidence log is produced by a recurring `geecs-capture-diff`
  sweep — schedule it, e.g. a service-user cron line diffing yesterday's
  scans nightly:

  ```
  15 6 * * * cd /opt/geecs/GEECS-Plugins/GeecsBluesky && <poetry> run geecs-capture-diff <yesterday's scans dir>/Scan* --log ~/.local/state/geecs-capture/diff-evidence.jsonl
  ```

  (exit 0 clean, 1 on a mismatch, 2 on operational errors such as an
  unmounted share — alert on non-zero.)
- Upgrades: `git pull` in the checkout + `systemctl restart geecs-capture`
  (the env is editable-installed; re-run poetry install when dependencies
  changed — or when console scripts were added, since bin stubs are
  generated at install time).
- **Upgrade the PVA gateway fleet to >=0.4.4 before any toggle-off scan.**
  On 0.4.3 a camera with a broken timestamp ladder publishes negative
  timestamps, so the daemon stale-filters 100% of its frames — harmless
  under dual-write, but with native saving off those images would exist
  nowhere, and the liveness preflight cannot see gateway versions.

## Migration to a new host

Copy the unit, install per step 1–2, ensure the shared `config.ini` and
the data-share mount match the worker's view, start. Nothing else is
host-bound — the daemon derives everything from the DB, the shared
config, and the document stream.
