# C1: Canopy on the simulator host — protocol facts

Lane C1 of `Planning/geecs_canopy.md`, run 2026-10-25 on an Apple-silicon Mac
(Docker Desktop 28.3, aarch64) against the geecs-epics simulator IOCs on
127.0.0.1. Everything below was observed with raw clients (`websockets`,
`httpx`, `caproto`) unless marked *(source)*, which means read from Canopy's
code at the pinned commit and not exercised. No Canopy file was patched.

## 1. Setup

| Item | Fact |
|---|---|
| Image | `ghcr.io/slaclab/canopy:0a5ee23`, digest `sha256:219da36a98713bf524cd623d1364ead9eef95016fc474e15ab27fcc62d4d8abd`, revision label `0a5ee23b6634…`, app version `0.0.2.dev211+g0a5ee23b6.d20261023` |
| Architectures | **amd64 only.** The index has one `linux/amd64` manifest plus an attestation (`unknown/unknown`); a plain `docker pull` on arm64 fails "no matching manifest". It runs emulated (`platform: linux/amd64`) |
| Networking | **(a) bridge worked first time**: `EPICS_CA_ADDR_LIST` = one `host.docker.internal:<port>` per IOC, `EPICS_CA_AUTO_ADDR_LIST=NO`, `EPICS_PVA_NAME_SERVERS=host.docker.internal:5085`. (b) and (c) were not needed. On the Linux box the reference compose's `network_mode: host` applies |
| Stack | `compose.yaml` here: api + one worker (the reference has two) from the image; `postgres:16-alpine` on 127.0.0.1:5432; `redis:7-alpine` on 127.0.0.1:6380; API on 127.0.0.1:8000 |
| Bring-up | `alembic upgrade head` 11 s (4 revisions: squirrel, magpie, core api_key, api_key modules); `up` to `/health` 200 in about 10 s, emulated |
| Keys | Env layer `CANOPY_API_KEYS` (read / read+write / admin) plus `scripts/create_key.py` (DB layer, works with only `CANOPY_DATABASE_URL`), `list_keys.py`, `deactivate_key.py` — all as documented |
| Log at start | `Mole started: 0 archivers`, `Raccoon started without a configured directory adapter`, `PVBroker started (backend=epics, batch=100ms)`; Vole subscribes to Hare internally |

PVs used: `undulator:simmotor1:m1.{VAL,RBV,DMOV,MOVN}`, `m2`;
`undulator:eztest1:{current,current:SP,enable_output,enable_output:SP}`;
`undulator:fpstest1:{current,current:SP,…}`; `undulator:uc_visaebeam6:`
`cam1:{ArrayCounter_RBV,AcquireTime_RBV,ImageMode,Manufacturer_RBV}`,
`STAT:{ArrayCounter_RBV,MeanValue_RBV,ProfileAverageX_RBV}`, `pva://…PVA:OUTPUT`.

## 2. Hare (Q1–Q3)

| Q | Fact observed |
|---|---|
| Q1 CA scalars | Every scalar connected, no patch. Integer records (`longin` counters) arrive as floats (`178001.0`) with `valueKind: scalar` |
| Q1 enums | `bi`/`bo`/`mbbo` → `valueKind: enum`, `value` = index, `enumChoices` in meta (`["off","on"]`, `["Single","Multiple","Continuous"]`) |
| Q1 waveforms | `STAT:ProfileAverageX_RBV` → `waveform`, 512 numbers in JSON; arrays truncated past `max_array_elements` 100 000 *(source)*; a `stringin` → `string` |
| Q1 pvAccess | `pva://` + TCP name server works through Docker Desktop. `EPICS_PVA_NAME_SERVERS` must list every IOC's PVA port: `pva://…simmotor1:m1.RBV` was unreachable (only the camera's 5085 was listed) |
| Q1 NTNDArray | `valueKind: unsupported`, `value: null`, `error: "unsupported PV type: epics:nt/NTNDArray:1.0"`, `connected: true` — and it **streams a null diff on every frame** (148 in 15 s at 24 Hz): Hare monitors the full image and throws it away. Describe: all-null meta, `writable: null` |
| Q2 `tsNs` | **IOC timestamp** for CA (`FORMAT_TIME`); server time only when the IOC sends none or on disconnect *(source + observed)*. The idle motor's RBV carried a 55-min-old `tsNs`; an unprocessed `ao` carried 1990-01-01 (UDF). The motor record posted `DMOV=0` with its *old* timestamp (screenshot 02): `tsNs` is not "time of this event" for every field |
| Q2 monitors | DBE_VALUE only: a record scanned at 1 Hz whose value does not change sends nothing (eztest1 `enable_output`: 1 diff in 16 s). **`tsNs` age cannot judge staleness; a producing counter can**: `STAT:ArrayCounter_RBV` and `cam1:ArrayCounter_RBV` arrive in every diff, so rule 1 (freshness by the plugin's counter) is computable client-side from Hare alone. Hare has no notion of `Compute*` flags |
| Q2 batching | `diff` every 100 ms (median 0.102 s, 0.085–0.153), changed PVs only, **latest-wins per window**: the 24 Hz counter steps by 2–4 between diffs, intermediate values are dropped. `rx − tsNs` ≈ 28 ms for the camera (emulated server) |
| Q2 heartbeat | Every 5.00 s, `{"type":"heartbeat","timestamp":…,"backend":"epics"}`, unaffected by an IOC outage (it says the broker is alive, nothing per-PV) |
| Q2 frame order | `initial` is **not** the first frame: on 2 of 3 fresh subscribes one or two `diff`s arrived before it. `initial` then carried the same or newer values, but `meta` comes only with `initial`, so a client must hold diffs until it has meta. The first diff after `initial` repeats every PV once |
| Q2 IOC stop | `docker stop fpstest1-sim` (10.7 s, SIGKILL): one `diff` for all 4 PVs about 70 ms after the IOC died: `value: null, connected: false, severity: INVALID, alarmStatus: null, error: "disconnected from IOC"`, `tsNs` = server time. No other frame; heartbeats continue. REST GET during the outage: 200 with that cached value, instantly |
| Q2 IOC start | Reconnected by CA on its own: one `diff` with all 4 PVs `connected: true` 12.9 s after `docker start` (emulated IOC boot). Nothing to do client-side |
| Q2 reconnect | Close + reopen: a new connection gets a fresh `initial` (data + meta) for whatever it subscribes; there is **no resume** (no sequence number, no "since"). Re-subscribing to held PVs on the same socket resends `initial` for them (a cheap refresh); unsubscribe + subscribe of one PV → `initial` for that one |
| Q3 put `wait:true` | `m1.VAL` 2→4: HTTP 200 after **2.21 s**, when the move ended (put-callback through DMOV); RBV streamed 10 Hz meanwhile. Body: `{"pv":"…"}` only |
| Q3 put `wait:false` | 4→2: HTTP 200 in **6 ms**, same body; the move then took 2.2 s. A false success looks exactly like a true one — nothing says "submitted, not done". It still fails fast when CA itself refuses (ACF) |
| Q3 refused by IOC (ACF) | FAST-PS `current:SP`: **502** `{"code":"pv_write_failed","detail":"CA write to …: Write access denied"}` in 30 ms, also with `wait:false`. `describe` had already said `writable: false` (CA access rights of Canopy's own CA identity) |
| Q3 beyond DRVH | eztest1 `current:SP` = 7.0 (DRVH 5): **200**, value clamped to 5.0, no alarm, no hint in the response. Rule 11's "clamp and write" is invisible to a Hare client unless it reads the value back and the record carries HIHI/LOLO |
| Q3 refused by device | eztest1 `current:SP` = 1.0 with output off (the fake NAKs): **200**; the record then reads `1.0`, severity `INVALID`, `alarmStatus 12`. Refusal = record alarm, never a failed put (rule 11), as on the bench |
| Q3 other errors | Bad type (`"abc"` to an `ao`, `"maybe"` to a `bo`): 502 "Channel write request failed". Unreachable PV: 502 after the 5 s CA timeout. Read key: 403 `insufficient_write_access`; no key: 401 |
| Q3 identity | **The API key is the only identity**, and nothing records it: the put route takes `{value, wait, timeout}`, no operator field or header *(source)*; the only log line is uvicorn's `"PUT /hare/v1/pvs/… 200"` from the client IP (here the bridge gateway). The IOC sees one CA client: user `appuser`, the container's hostname (bridge) or the box's (host networking) *(source: image user)*. DB keys record only `last used` |
| GET/describe | Unconnected PV: `GET` returns **200** with `connected: false, error: "CA read failed: 80"` after 5 s, not the documented 503/504. Describe carries an undocumented `value: null`. Unset limits come back as `0.0/0.0`, not `null` (counters, readbacks) — a client must treat 0/0 as "no limits" |
| Caps | 5000 PVs per subscribe, 20 000 per client, both silent drops; queue 100 000; seed concurrency 32 *(docs)*. Bad PV name → `error` frame `invalid_pv_name` listing the names; garbage → `invalid_message`; socket stays open. Bad key on WS: handshake rejected **HTTP 403** (docs say close 1008) |
| Step 17 | Four WebSocket clients at once (three panels + Vole): 10 monitors shared, each client its own 100 ms diff cadence, no starvation (`health/summary` `clients: 4, activeMonitors: 10`) |

## 3. Squirrel (Q4)

- **A "device" is a free-text label** on a registry row (`device`, ≤255
  chars); `GET /squirrel/v1/pvs/devices` lists the distinct labels. A row =
  `setpointAddress` + `readbackAddress` (+ `configAddress`), `absTolerance`,
  `relTolerance`, `description`, tags. Registered via `POST /pvs/multi`
  (camelCase in and out; `createdBy: null`).
- **Save** = `POST /snapshots {title, description}` → `{jobId, statusUrl}`;
  the worker reads **every registered PV** (no per-device or per-tag
  snapshot) and `GET /jobs/v1/{id}` reports `completed`, `result.id`, in
  0.5–1.5 s for 5 rows (5.2 s with one IOC down: the CA timeout). Each value
  is stored as `{value, status, severity, timestamp, units, precision, …}` —
  `units`/`precision` came back `null` (read with `FORMAT_TIME`).
- **Compare** is **snapshot vs snapshot only** (`GET /snapshots/{a}/compare/{b}`
  → `differences[{pvId, pvName, value1, value2, withinTolerance}]`,
  `matchCount`, `differenceCount`). There is no snapshot-vs-live verdict:
  steps 11–12 ("active verdict", "diff after a move") need a fresh snapshot
  or a client-side comparison against Hare. After moving m1 2→3 and m2 0→1
  it reported both setpoints as differences; values within `absTolerance`
  match. `value1/value2` are the setpoints; a readback-only drift is
  reported under the same keys *(source)*.
- **Restore a subset**: `POST /snapshots/{id}/restore {"pvIds":[m1]}` →
  job; m1 went 3→2 and m2 stayed at 1. The job took 1.36 s for a 1 mm move at
  1 mm/s: **restore waits on put-callback** (`connector.put` default
  `wait=True`, 5 s timeout, concurrency-limited *(source)*). Result:
  `{totalPvs, successCount, failureCount, failures[{pvId, pvName, error,
  phase:"Setpoints"}]}`.
- **Unreachable PV** (fpstest1 stopped): save **still reports `completed`,
  "Saved 5/5 values"**; the FAST-PS row is stored with
  `setpointValue: null, readbackValue: null, severity: null` — nothing marks
  it unreachable. Compare against the baseline then lists it as a
  difference (`value2: null`). Restore-all with it down: job `completed`,
  `4/5 PVs restored`, `failures[0].error` = the CA timeout text. A baseline
  that captured the FAST-PS `ao` in UDF/INVALID (1990 timestamp) would be
  restored as-is: no severity guard.
- **Identity**: `createdBy` is always `null` — the create route never passes
  it *(source)*; tags and meta-PVs (context PVs stored with every snapshot)
  exist and were not exercised. Restore hooks: none configured.

## 4. Mole and Raccoon (adapter interfaces, for later lanes)

**Mole** (not exercised: the archiver pilot is on the lab network). The
built-in `epics_archiver_appliance` adapter uses only stock Appliance
endpoints: `/mgmt/bpl/getApplianceInfo` (health), `/retrieval/data/getData.json`
(raw window, zero-width "last before", `optimized_<N>(pv)` for binning,
metadata from the same response), `/retrieval/bpl/searchForPVsRegex`. GeecsArchiver runs
upstream's stock appliance, so **the built-in adapter should be enough**:
`CANOPY_MOLE__DEFAULT_ARCHIVER=geecs`, `CANOPY_MOLE__ARCHIVERS__GEECS__TYPE=epics_archiver_appliance`,
`…__URL=<appliance root>`, optional `…__API_KEY_ENV`. Several named archivers
are allowed. A custom adapter is 8 methods (`from_config(name, cfg)`,
`fetch_window`, `fetch_last_before`, `fetch_binned`, `metadata`, `search`,
`health`, `close`) loaded by short name or `module:Class`. The how-to shows
TOML config; the code reads env (`CANOPY_MOLE__…`) — a doc mismatch.

**Raccoon** (for C2). Exactly **one** directory adapter per instance
(`CANOPY_RACCOON__DIRECTORY__TYPE` = registered short name or
`module:Class`, imported at startup; `TIMEOUT_S`, `BATCH_CONCURRENCY`,
`EXTRAS__<KEY>` → `cfg.extras` dict, `CACHE__…`). Unconfigured it answers
500 `adapter_misconfigured` and its `health/summary` 200 with
`healthy: false`. The protocol is three list calls,
`list_pvs/list_devices/list_elements(pattern, *, tag, sort_by, element_type) -> list[str]`,
plus `from_config`, `health() -> AdapterHealth`, `close()`, attributes
`name`, `type_name`, optional `pattern_metachars`. Patterns, tag, sort and
element type are opaque, passed verbatim (reject with `InvalidQuery`).
Every answer is a **flat list of names**: there is no device → PVs call and
no structure per name, so "the PVs of device X" can only be a pattern the
adapter interprets. `exists`/`validate` are built on `list_pvs(<literal name>)`
plus Raccoon's name cache; names containing metacharacters are refused.
Reference adapter: `MemeDirectoryAdapter` (owns its thread pool for a
synchronous client, the pattern the how-to recommends); contract tests in
`tests/canopy/raccoon/adapters/contract`.

**Mounting code (C2/C3).** The image has no `pip` and no `uv`
(`/app/.venv`, Python 3.12.15, user `appuser`). A bind mount of a directory
of pure-Python packages plus `PYTHONPATH` imports inside the image
(verified: a probe module importing `canopy.raccoon.adapters.base`). A
dependency with compiled parts needs a derived image instead. Modules are
a fixed tuple `API_MODULES` mounted in `main.py`; no plugin entry point
exists for a new module *(source)*.

## 5. Operations (Q7)

| Item | Fact |
|---|---|
| Images | canopy 614 MB, postgres:16-alpine 411 MB, redis:7-alpine 58 MB; pgdata 49 MB after the exercise |
| Idle | api 150 MiB / 0.9 % CPU, worker 121 MiB / 0.4 %, Postgres 33 MiB, Redis 10 MiB (emulated amd64) |
| Under load | camera's 4 scalars at 24 Hz: api 3.4 % CPU; adding the NTNDArray: 7.5 % (the discarded image monitor doubles it); memory flat at 150 MiB. Native amd64 on Linux will be cheaper |
| Needed and not run by GEECS today | Postgres 16 (one schema per module; Alembic run before every image change), Redis 7 (arq queue: Squirrel save/restore go through the worker, so a dead worker = no snapshots), the arq worker process(es), API keys (env + DB registry), an amd64 host or emulation |
| Front door | Root-relative routes (`/hare/v1/…`); the docs page builds OpenAPI URLs from the ASGI `root_path`, so a path prefix needs `--root-path` or the proxy to set it; WebSocket at `/hare/v1/ws/pvs`, key as header or `?api_key=` |
| CORS | `allow_origins=["*"]` with credentials, hard-coded in `main.py` *(source)* |
| Pin policy | No releases yet: tags are short SHAs + `latest`; release tags (`X.Y.Z`) are documented but none exist. Pin a SHA and its digest; re-run `alembic upgrade head` before any new SHA; an older image cannot run on a newer schema |
| Warning | `CANOPY_EPICS__BACKEND` defaults to `simulation`: a forgotten setting passes `/health` with sine waves; check the heartbeat's `backend` |

## 6. Defects and surprises (each a candidate upstream issue)

1. No arm64 image (a Mac or Pi bench runs it emulated).
2. NTNDArray: monitored at full rate and emitted as null diffs; a client
   pays the bandwidth and CPU for nothing. Skip the monitor (or send only
   `uniqueId`/timestamp) until `image` lands.
3. `diff` can precede `initial` on a fresh subscribe; `initial` is the only
   carrier of `meta`.
4. `wait:false` and `wait:true` return the same 200 body: no "accepted, not
   confirmed" marker.
5. No write audit at all: no key name, value or result in any log; Squirrel's
   `createdBy` is never filled.
6. Snapshot save says "Saved 5/5" and `completed` when a PV was unreachable;
   rows with null values carry no `connected: false`.
7. `GET` of an unconnected PV is 200 with `connected:false`, not the
   documented 503/504; WS auth failure is HTTP 403, not close 1008;
   describe has an undocumented `value`; unset limits are `0.0`, not `null`.
8. Integer records arrive as floats.
9. Mole's how-to shows TOML; the code reads env only.
10. (Ours, not Canopy's) a zsh heredoc ate `$KEY:read` (`:r` is a zsh
    modifier), and Canopy rightly skipped the malformed entries with a
    warning that names positions only — good behaviour, worth knowing.

The exercise stopped `fpstest1-sim` twice (Hare outage, Squirrel
unreachable save/restore) and put back every value it changed: m1 2.0,
m2 0.0, eztest1 `current:SP` 0.0 with output on, no alarms.

## 7. Questions only SLAC can answer

1. When do `image` (binary frames) and an arm64 image land, and will a
   subscribed NTNDArray stop being monitored until then?
2. Is a write audit (key app name, value, result) or a client-supplied
   operator field on `PUT /hare/v1/pvs` welcome upstream, and would
   Squirrel's `createdBy` come from the key's app name?
3. Is there a plan for module plugins (entry points) beyond the fixed
   `API_MODULES` tuple, and for a snapshot-vs-live compare?
4. Release cadence: when do `vX.Y.Z` tags start, and is the schema
   migration path between SHAs supported?

## Files

`compose.yaml` (the exact runtime file), `.env.example` (placeholders),
`screenshots/`: 01 the Hare demo on 14 sim PVs, 02 a motor move with
DMOV's stale timestamp, 03 the plot demo on three PVs, 04 a motor move
2→5→2 plotted (RBV against VAL).
