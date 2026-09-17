# Phase 1 — headless hardware acceptance (PR 3, 2026-09-10)

Raw records and the runbook behind the phase-1 verdict in
`03_clean_room_rebuild.md` §2/§8: the plan layer (PR 2, #821) driven on HTU
by the worker's own wiring and through a second RE Manager, then the
worker flip.  Same conventions as `04_phase0_measurements.md`: each
measurement names what it was for, what was written to hardware, and what
it showed.  Setup: HTU-NoGas, camera `UC_Amp4_IR_input` (native PNGs),
sweep `U_S1H:Current` −1 → +1 A in 5 points, setpoint restored after every
run; the configs corpus from the `presets-v1` branch
(`GEECS_SCANNER_CONFIG_DIR` pointed at a checkout beside the staging
clone — the share stays on `main` for the deployed worker and the
clients).  The test is `GeecsBluesky/tests/test_phase1_hardware.py`
(`GEECS_HW=1`; it fires shots).

## Runbook — a second RE Manager beside the deployed one

The deployed worker (`~/qs-checkout`, systemd `geecs-qserver`, ZMQ
60615/60625, doc proxy 5567/5568, Redis 6379) keeps running on its
branch; the acceptance manager runs from the staging clone on its own
ports and its own Redis key prefix, with the document publisher off (the
production proxy must not receive its documents).  As the service
account:

```bash
cd ~/deploy-staging/GEECS-Plugins && git fetch && git checkout feature/native-bluesky-plans && git pull --ff-only
cd GeecsBluesky && poetry env use python3.11 && poetry install --extras "ca tiled qserver qs-client"
cd ~/deploy-staging && git clone -b presets-v1 https://github.com/GEECS-BELLA/GEECS-Plugins-Configs.git

cd ~/deploy-staging/GEECS-Plugins/GeecsBluesky
export GEECS_SCANNER_CONFIG_DIR=$HOME/deploy-staging/GEECS-Plugins-Configs/scanner_configs/experiments
export QS_EXPERIMENT=Undulator QS_DOC_PUBLISH_ADDR=OFF EPICS_CA_ADDR_LIST=192.168.6.14 EPICS_CA_AUTO_ADDR_LIST=NO
V=$(poetry env info -p)
nohup $V/bin/start-re-manager --startup-dir $PWD/qserver/startup \
  --user-group-permissions $PWD/qserver/user_group_permissions.yaml --keep-re \
  --zmq-publish-console ON --zmq-control-addr "tcp://*:60635" --zmq-info-addr "tcp://*:60645" \
  --redis-name-prefix qs_acceptance > ~/deploy-staging/acceptance-manager.log 2>&1 &
$V/bin/geecs-qserver-ensure-ready --control-addr tcp://localhost:60635 --timeout 300
#   → "ready: 19 allowed plans, all 19 expected present"

GEECS_HW=1 poetry run python -u -m pytest tests/test_phase1_hardware.py -m hardware -s -k in_process
GEECS_HW=1 GEECS_HW_QSERVER=tcp://localhost:60635 poetry run python -u -m pytest tests/test_phase1_hardware.py -m hardware -s -k manager
pkill -f "start-re-manager.*60635"      # never two managers able to arm the box at once
```

`QSERVER_ZMQ_CONTROL_ADDRESS=tcp://localhost:60635 qserver status` inspects
the acceptance manager from the CLI.  The two `redis-name-prefix`es keep
the queues apart in the one Redis.

## M4 — the plan layer in process (Scans 104 and 105, 26_0910)

**Purpose.** Prove PR 2's deliverable with the worker's own wiring:
`make_run_engine(claim=True, path_provider, telemetry=namespace.telemetry())`,
the namespace on the shared `GeecsScanPathProvider`, `TriggerProfiles`
from the configs repo, the bound `count` and `scan`.

**Result — `1 passed in 72 s`.**

| | Scan104 (`count`, 3 shots) | Scan105 (`scan`, 5 points × 2 shots) |
|---|---|---|
| start doc | `plan_name`, `experiment`, `trigger_profile=HTU-NoGas`, `shots_per_step` 1 / 2, `scan_folder`, `geecs_scalar_headers` | same; `scan_number` = previous + 1 |
| native files | 3 PNGs in `Scan104/UC_Amp4_IR_input/`, named by the rows' stamps | 10 PNGs, named by the rows' stamps |
| `ScanInfoScanNNN.ini` | `Scan Parameter = "Shotnumber"`, `Shots per step = 3`, `ScanEndInfo = "success"`, `ScanMode = "noscan"` | `Scan Parameter = "U_S1H Current"`, `Start/End/Step size = -1.0/1.0/0.5`, `Shots per step = 2`, `Plan = "scan"`, `Trigger profile = "HTU-NoGas"` |
| s-file (`analysis/sNNN.txt` = `ScanDataScanNNN.txt`) | 3 rows, `Bin # = 1,1,1`, headers `UC_Amp4_IR_input MeanCounts` … `acq_timestamp` | 10 rows, `Bin # = 1,1,2,2,…,5,5`, `U_S1H Current` = −0.99984 … 0.99989 |
| `scan.log` | one `finished (success)` | one `finished (success)`, the five `moving Current →` lines |
| `baseline` stream | 419 columns, 2 rows | 2 rows |
| trigger box | driven back to STANDBY after the run (`ShotControl.standing_state`, the last state *written* — a bookkeeping check; ARMED is observed by the shots landing, not read back) | same; `save` reads `off` after the run |
| build | 26 s (108 devices, 375 telemetry objects connected; two unservable devices dropped loudly) | |

**Cadence (from the rows' stamps).** Scan104: `[2.0, 1.0]` — 1 Hz after
the first shot.  Scan105: `[2.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0]` —
the repeat shot at a position lands on the next edge, a **moved step
lands on the third edge** (M2 saw the second).

## M5 — a preset through the RE Manager (Scans 106 and 108)

**Purpose.** The queue-item contract end to end: `Preset` →
`run_submit_preflight` (expand, the worker lists the plan and every
reference, `CONNECTED`) → `submit_preset` → the manager runs the bound
`scan` → the same files.

**Result — `1 passed in 39 s` (Scan108; Scan106 was the first run, whose
test wait returned early — see below).**  Preflight
`[validate, worker_ready, gateway_liveness] passed`; history item
`exit_status: completed`, `scan_ids: [106]`; 10 rows, 10 PNGs, bins
1,1,…,5,5, `ScanStartInfo` = the preset description, `Trigger profile =
"HTU-NoGas"`; the restore `mv` queued and ran; `U_S1H` setpoint 0.0
after.

**Cadence.** Scan106: `[2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0]` —
through the manager the repeat shot lost an edge too (see M6).

Re-run after the #822 review (the restore path now waits for the
manager to go idle and retries): Scan001 of 26_0911, `1 passed in 35 s`,
restore queued and ran, setpoint 0.0.

**Two lessons from the first run.** (1) A status poll right after
`queue_start` still reads idle — "done" is the item reaching the
manager's history (`_wait_for_item`). (2) The restore `mv` submitted
while the scan item was still queued was refused by the #648
front-of-queue guard, leaving `U_S1H` at 1 A until restored by hand; the
test now awaits its own item and restores with `clear_pending=True`.

## M6 — where the shot period goes (Scan107, a 6-shot count, DEBUG timings)

```
shot phases: fire 247 ms, frame wait 1289 ms   (first shot)
shot phases: fire 235 ms, frame wait 1685 ms
shot phases: fire 108 ms, frame wait  886 ms   ×4
count stamp gaps: [2.0, 1.0, 1.0, 1.0, 1.0]
mv U_S1H → -0.5: 1.28 s;  mv → 0.0: 1.26 s   (bare 0.5 A moves)
prepare (localsavingpath + save puts): ~2 ms   (values unchanged)
```

**Reading these numbers correctly (Sam's point, 2026-09-11).** The fire
request is not synchronized to the laser: in *single shot external rising
edges* the DG645 fires on the **next** edge after the put, so for a shot
whose request lands at a random phase the request-to-frame delay is
uniform over one period.  That is the first shot of each run here (frame
waits of 1289 and 1685 ms).  Every shot after it is **phase-locked**: the
fire is issued a fixed time after the previous frame arrived, so it lands
at a fixed phase before the edge, and the "886 ms frame wait" is simply
1000 ms minus the ~115 ms of fire put + plan work — it says the loop is
locked at 1 Hz, not what the camera's latency is.

The camera's latency is in the documents: event arrival time minus the
row's stamp (converted from the LabVIEW epoch) is **0.75 s** on every
locked shot (0.87–0.88 s on the first shot of a run, which includes the
first `prepare`), and the stamp precedes the edge-to-message path by the
~65 ms drain (M1), so edge → message ≈ **0.8 s** — the camera's 0.70 s
exposure (`UC_Amp4_IR_input exposure` in the s-file) plus readout and the
TCP push.  The stamps' fractional part is constant (.553 s) — the laser
edges are phase-stable at 1 Hz.  So the budget per period is:
edge → message ≈ 800 ms, fire put ≈ 108 ms, plan work ≈ 7 ms, leaving
**≈ 85–100 ms** for anything else per shot (a document callback's HTTP
round trip, the manager's per-message overhead).  In process that holds
(1 Hz); in the manager's worker process it did not (2 s repeats, M5).
A shorter exposure buys margin directly; nothing in the plan layer does.

A moved step adds the 1.3 s blocking GEECS set of the magnet, which
starts phase-locked (after the previous frame) and therefore ends
phase-locked: previous frame (+0.8 s) → move done (+2.1 s) → fire
(+2.2 s) → the edge at +2 s is gone, the shot lands at +3 s, every time
(the 3.001 s gaps).  M2's 2 s per step (2026-09-09) means the same move
was under ~1 s that day — device-side variation, not the plan.

**Consequence (§11).** The stamp governs the join, the *message arrival*
governs the wait (§11.4) — and on this camera the arrival leaves ~100 ms
of the period.  Strict single-shot cannot sustain 1 Hz with any per-shot
cost beyond the fire unless the exposure is shortened; the native answer
for rep-rate is phase 2's gated batch (the edges flow, the detector
counts), not a faster fire.  Overlapping the magnet move with the
previous shot's wait would recover the moved step (2 s → still not 1 s)
— a plan-layer option, not taken here.

## M7 — the same runs at a 1 ms exposure (Scans 002 and 003 of 26_0911)

**Purpose.** Test M6's reading: if the ~0.8 s edge-to-message latency is
the camera's 0.70 s exposure, a short exposure should restore the margin
and both cadence losses should go.  Sam set `UC_Amp4_IR_input`'s
exposure to 1 ms (readback 0.001016 s).

**Result.**

| | 0.70 s exposure (M4–M6) | 1 ms exposure (M7) |
|---|---|---|
| stamp → RE event (edge-to-message minus the drain) | 0.75 s | **0.05 s** |
| in-process `count`, stamp gaps | `[2.0, 1.0, 1.0, 1.0, 1.0]` | `[1.0, 1.0, 1.0, 1.0, 1.0]` — 1 Hz from the first shot |
| per-shot phases (fire put / frame wait) | 108 / 886 ms | 140–228 / 766–781 ms |
| preset through the manager, `scan` 5 × 2 | `[2.0, 3.0, 2.0, 3.0, …]` | `[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0]` |
| moved step | third edge (3 s) | **second edge (2 s)**; step period 3 s |

The plan layer did not change between the two.  The repeat shot through
the manager's worker process is back at 1 Hz (the ~650 ms of margin now
covers its per-message overhead), and a moved step lands on the second
edge, as the 1.3 s move alone dictates.  The fire put itself read
140–230 ms this time (108 ms in M6) — the gateway put's own variance,
inside the margin either way.  Files, ScanInfo, s-file bins and the
restore all as in M5 (`1 passed in 31 s`).

**So (§11):** the per-shot budget at 1 Hz is the camera's exposure +
readout + push, the fire put, and whatever runs per event; the plan
layer's own ~7 ms is noise.  A long exposure eats the period directly.

## The worker flip (prepared 2026-09-10; done 2026-09-11)

**Done 2026-09-11:** the restart below happened in the morning (19 plans,
Scan005 of 26_0911 through the manager), and a second restart at 15:00
put the worker on the #823 merge with the file-plugin host list —
`07_806_acceptance.md` has that record.  The rest of this section is the
runbook as prepared.

What the flip is: `~/qs-checkout` (the deployed worker's clone, shared
with `geecs-capture`; the MCP venv is baked separately and does not
follow) on `feature/native-bluesky-plans`, its environment re-installed
(`poetry install --extras "ca tiled qserver"`, the optimize extra left in
place), then `sudo systemctl restart geecs-qserver` — the readiness
oneshot re-runs and asserts the 19 plans.  The deployed worker then runs
stock plans over presets; the master Console and MCP can no longer submit
(they name the retired funnel plan; their preflight refuses first) until
their rewire — the accepted state of §10.5.

Done as the service account (no restart, which needs sudo): the
acceptance manager stopped; `~/qs-checkout` fetched, checked out on
`feature/native-bluesky-plans` at the #821 merge, `poetry install` clean,
`geecs_bluesky.plans.registry` / `callbacks` / `capture.daemon`
importable in the production env.  The running `geecs-qserver` still
holds master's code until the restart:

```bash
sudo systemctl restart geecs-qserver      # pulls geecs-qserver-ready along
journalctl -u geecs-qserver-ready -n 3 --no-pager   # "ready: 19 allowed plans, all 19 expected present"
sudo systemctl restart geecs-capture      # picks up the same clone's code (optional; not required for production)
```

The clients' configs stay on the share's `main` (the worker reads no
presets; the trigger profiles, scan-variable catalog and action library
are unchanged there); the `presets-v1` branch merged into the configs
`main` on 2026-09-11, after the flip.
