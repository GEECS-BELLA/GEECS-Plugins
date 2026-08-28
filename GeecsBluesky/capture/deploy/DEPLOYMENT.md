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

2. Copy `geecs-capture.service` to `/etc/systemd/system/`, replace the
   checkout-path and poetry-path placeholders (same values as the qserver
   unit on this host), set the experiment name, then:

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now geecs-capture
   ```

3. Verify: `journalctl -u geecs-capture -n 5` shows the discovery line
   ("capture discovery: N eligible cameras") and the running banner; the
   heartbeat file appears at `~/.local/state/geecs-capture/heartbeat.json`
   (override: `[capture] heartbeat_path` in the shared config.ini) and its
   timestamp refreshes every ~10 s.

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
  geecs-capture`); a restart mid-scan loses that scan's capture only —
  native files (dual-write) are untouched.
- **Restart the daemon after camera roster changes** (devices added to the
  experiment, devicetype fixes): targets are discovered at startup, and
  the daemon errors loudly on engine-listed devices it has no target for.
- Upgrades: `git pull` in the checkout + `systemctl restart geecs-capture`
  (the env is editable-installed; re-run poetry install only when
  dependencies changed).

## Migration to a new host

Copy the unit, install per step 1–2, ensure the shared `config.ini` and
the data-share mount match the worker's view, start. Nothing else is
host-bound — the daemon derives everything from the DB, the shared
config, and the document stream.
