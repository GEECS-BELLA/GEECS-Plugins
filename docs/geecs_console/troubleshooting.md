# Troubleshooting

Work outside-in: the health chips first, then the scan's own log, then
the triage tools.

## The health chips (session bar)

| Chip | Down means | First checks |
|---|---|---|
| **Gateway** | The CA gateway isn't serving PVs — no readbacks, no scans | Is the gateway service running on the lab server? `caget <Expt>:CAGateway:HEARTBEAT` |
| **Tiled** | Runs won't be archived / browser has nothing to read | Tiled service on the lab server; the `[tiled] uri` in `config.ini` |
| **DB** | Name completion and DB-driven variable lists unavailable | MySQL on the lab server; credentials chain in `config.ini` |

A scan can still run with Tiled or DB down (with reduced recording); it
cannot run without the gateway.

## Common situations

**"Pre-flight: all sync devices stale" dialog.** The cameras haven't
produced frames recently — usually the trigger is off or not free-running.
If you're about to arm a trigger profile that turns it on, Continue is
correct; the scan will fail loudly at t0-sync if the trigger really is
dead.

**Strict scan refuses to start.** Strict needs a reachable shot-control
device and an `ARMED` state in the trigger profile. Use free-run for
free-running-trigger acquisition.

**A scan folder exists but has no data.** The scan was claimed and then
aborted/failed — folders are never deleted once claimed (a hard rule with
incident history), so an empty `ScanNNN/` with a `scan.log` is the normal
residue of an aborted run. The log says why. Delete it manually if you
wish.

**Scans run at half the trigger rate (or slower).** Camera timestamp
deltas that are *exact multiples* of the trigger period mean the trigger
and cameras are healthy but the per-row software cycle exceeds one period.
That family of problems was systematically addressed (staged reads,
batched telemetry — 2026-07); if it recurs, run the scan-timing audit
below — the delta signature identifies the culprit class immediately.

**Start feels slow.** The first scan after launching the console pays a
one-time cost (creating CA channels for every telemetry device — tens of
seconds over VPN, a few on the lab network). Subsequent scans start in
about a second. Consistently slow *warm* starts are worth a report.

**Optimization mode refused.** The optimization stack is an optional
extra (`poetry install --extras optimization` in GEECS-Console); without
it, optimize submissions are refused with a status-bar message and every
other mode works normally.

## The log tools

Every scan writes its own `scans/ScanNNN/scan.log` — the first thing to
read for any single-scan question.

For systematic checks (with a Claude/agent session on this repo):

- **`/triage`** — parse a day's scan logs into a structured error report
  with fingerprinted, classified failures (bug vs config vs hardware vs
  operator). CLI: `geecs-log-triage --date YYYY-MM-DD --experiment NAME`.
- **`/scan-audit`** — the timing counterpart: phase timelines, per-shot
  cadence analysis, start-latency breakdown.
- **`/env-doctor`** — when the tools themselves won't run (Python env
  problems).

One maintenance habit: after editing devices/variables in the GEECS
database GUIs, run `geecs-ca-gateway-audit --experiment NAME` (from
GeecsCAGateway) — the DB editors don't cascade deletions, and orphaned
logging rows become phantom PVs that waste connect time at scan start.
