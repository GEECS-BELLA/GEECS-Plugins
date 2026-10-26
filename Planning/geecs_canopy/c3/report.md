# C3: the GEECS device module — in-process, mounted, no patch

Lane C3 of `Planning/geecs_canopy.md` (Q6, steps 05, 07, 08, 09), run
2026-10-26 against lane C1's Canopy (`ghcr.io/slaclab/canopy:0a5ee23`) on the
geecs-epics simulator. Source citations are `src/canopy/…:line` at slaclab/canopy
commit `0a5ee23` (a shallow clone, read, never patched).

## 1. Decision: in-process (shape A), mounted

A wrapper ASGI module imports `canopy.main:app`, mounts `/geecs` on it and is
run by the same image in place of `canopy.main:app`. It works with no Canopy
patch, and the live instance is running it now. It costs two things, both
upstreamable (§5): only `@*` keys reach the mount, and the in-process put
goes through `get_broker()`, which is not a declared gateway function.

| Question | Finding (source) |
|---|---|
| How modules mount | `main.py:45-56` `API_MODULES` is a literal tuple; `main.py:59-83` composes the ten lifespans by name ("Mounted sub-apps don't run their own"); `main.py:112-121` `app.mount`; `main.py:123-132` `tag_module`. Nothing is discovered: no entry point, no registry |
| Can a wrapper add one | Yes. `app` is a module-level FastAPI object (`main.py:86`); `app.mount("/geecs", sub)` after import adds a `Mount` route, and Canopy's lifespan, which binds Hare's broker (`hare/app.py:18-42`), runs unchanged. Verified in the image: mounts = the ten plus `/geecs` |
| Tag | Required: `core/security.py:210-219` reads `module_for_app(request.app)` (`core/modules.py:55-72`); an untagged mount raises `AttributeError` on the first keyed request. `tag_module(sub, "geecs")`. Metrics then label it `module="geecs"` on their own (`core/metrics.py:7-12`, observed) |
| `@geecs` key | `core/modules.py:19-33` `MODULES` is a hard-coded frozenset pinned to the mounts by `tests/canopy/core/test_modules.py:13-27`. `split_modules` (`modules.py:36-52`) drops unknown names: the env layer warns and **skips** a key whose modules are all unknown (`security.py:131-145`); the DB key service **rejects** it, 400 `Unknown module(s)` (`core/keys/service.py:55-57`). Observed in the image: `t:read@geecs` → skipped; `t:read@geecs+hare` → a hare-only key, 403 at `/geecs`; `t:read@*` → reaches `/geecs` (`allows_module`, `security.py:46-47`) |
| Canopy's own drift test | Pins Canopy's `main.py` only; it never sees the wrapper, so nothing in Canopy's CI breaks. `/docs` lists `API_MODULES` (`main.py:147-153`), so `/geecs` is absent there; `/geecs/openapi.json` is served |
| Reaching Hare in-process | `hare/gateway.py:1-29`, "other Canopy modules import only from this file", exposes `get_value(pv)` and `get_broker()`, **no put**. Hare's own PUT route calls `broker.put` (`hare/api/v1/pvs.py:41-53`, `hare/services/broker.py:570-581`), and Vole subscribes through `get_broker().add_client/subscribe` (`vole/app.py:6,49`). The module uses `gateway.get_value` and `gateway.get_broker().put`, the same path as Hare's route: the PV stays the same, and so do the ACF identity and the `PVWriteFailed` 502 (`epics/base.py:37-39`) |
| Import contracts | `pyproject.toml:93-115`: one import-linter contract, binding `canopy.converter` only ("reaches other modules only through their public gateway"). Nothing constrains an outside package; by that convention `get_broker()` is public by location, internal by type (`PVBroker` lives in `hare/services`) |
| Keys and errors | The module's routes depend on `require_read_access` / `require_write_access` (`security.py:245-255`), so the 401/403 envelope and scopes are Canopy's (observed: a read key's put → 403 `insufficient_write_access`). `register_canopy_error_handler` (`core/errors.py:192`) on the sub-app gives `{code, detail}`. `ErrorCode` (`errors.py:12`) is a closed `StrEnum` and `CanopyError.error_code` is typed by it (`errors.py:102`), so a module cannot raise its own slug through the shared handler: `denied` is a hand-built `JSONResponse` in the same shape |
| DB | `core/db.py:1-5`: one Postgres, one schema per module, shared engine. Alembic is **one tree** for all modules, version table in `public` (`alembic.ini:1-2`, `migrations/env.py:30,58-59`), so an outside module's schema needs its own tree and version table, or a revision in Canopy's. The spike uses SQLite |

**Shape A vs shape B (sidecar).**

| | A: in-process wrapper (built) | B: sidecar (not built) |
|---|---|---|
| Canopy guarantees today | Nothing named: `main.app`, `tag_module`, `require_*_access`, the error handler, `gateway.get_value` are stable-looking module-level names; `get_broker().put` is the internal `PVBroker` API | Hare's documented REST/WS wire (C1's report) and the key model; a service key per sidecar |
| Breaks on a Canopy upgrade when | `main.app` stops being importable, the security dependencies or `PVBroker.put` change signature, or a new module claims `/geecs` | the Hare wire changes (versioned `/v1/`) |
| Identity | Canopy's key resolved in-process (`APIKey.name` journaled) | the sidecar holds one write key: every put reaches Hare as the sidecar; the user's key must be re-checked by the sidecar (no introspection route) |
| Cost | none extra: same process, broker cache shared with every WS client | a second service, a hop, and it must keep its own WS subscription for freshness |
| Upstream to make it first-class | §5 rows 1–4 | none needed; less fit (two front doors, key duplication) |

A sidecar only buys independence from Canopy's Python surface, which is
small. In-process keeps one front door, one key model and one broker, so A.

## 2. What mounted, and how

- Package `geecs_canopy_module/` (pip-installable, pure Python; FastAPI and
  pydantic are already in the image): `family.py` (loader for the four
  vendored family files, `git show` of GEECS-Plugins `7f38c12ace88`,
  `feat/web-estate`; and `devices.json`, the four sim instances), `routes.py`
  (the two routes over a `Hare` protocol, no Canopy import), `journal.py`
  (SQLite), `app.py` (the wrapper: `GatewayHare`, mount, tag, handler).
- Runtime (outside the repo): `pip install --target <runtime>/geecs-c3/ext
  --no-deps .`; `compose.c3.yaml` beside C1's `compose.yaml` replaces only the
  api's command with `uvicorn geecs_canopy_module.app:app`, bind-mounts the
  ext dir (ro) and a journal dir (rw), `PYTHONPATH=/opt/geecs-c3:/opt/geecs-ext`.
  The worker is untouched.
- **LOC**: code 399 lines (app 65, family 120, journal 43, routes 170,
  `__init__` 1, all counted, docstrings included); tests 192. `pytest` 10
  passed (Python 3.11, no Canopy installed: Hare is a fake). ruff, ruff-format,
  pydocstyle and import-contracts are green. **Bite**: `b > a` → `b >= a` in
  `_advanced` (a still counter judged fresh) fails `counter-still` and
  `test_a_changing_value_is_not_freshness`; restored, all pass.
- Freshness: the module reads each `fresh_by` counter, waits `windowS`
  (1 s), reads it again; `fresh` iff it advanced. `off` when the
  `computed_by` flag reads 0 while connected; otherwise Hare's `connected`.
  `tsNs` and a changing value are never used (rule 1).
- Put: role not in an allowed action → 403 `denied` (journaled, Hare never
  called); else `broker.put(pv, value, wait=True, timeout=30)`; a
  `CanopyError` → `refused` with Hare's status, `code`, `detail`. One row per
  attempt: time, operator, key name, device, role, axis, value, outcome,
  canopy_code, detail.

## 3. Live proof (C1's instance, final code)

`GET /geecs/v1/devices/U_SimMotor1` (read key), HTTP 200 in 0.39 s, 58 roles
(29 × m1, m2), all `connected`; the first role entry:
```json
{"device": "U_SimMotor1", "family": "motor_record", "windowS": 1.0, "roles": [
 {"pv": "undulator:simmotor1:m1.RBV", "value": 2.0, "valueKind": "scalar", "severity": "NO_ALARM",
  "alarmStatus": 0, "tsNs": 1793026584292224000, "connected": true, "error": null,
  "role": "rbv", "axis": "m1", "verdict": "connected"}, "…57 more"]}
```
`GET …/UC_VisaEBeam6`: 200 in 1.07 s, 82 roles, the 8 STAT/STAT0 roles `fresh`
(counter 972 910 advancing), 74 `connected`:
`{"pv": "undulator:uc_visaebeam6:STAT:MeanValue_RBV", "value": 32857.498046875, …, "role": "mean", "axis": null, "verdict": "fresh"}`

Puts (write key, `operator: "c3-exercise"`):
```text
twf on m1         HTTP 200 1.32 s  {"outcome":"accepted","device":"U_SimMotor1","role":"twf","axis":"m1","journalId":1}   (m1.RBV 2.0 → 3.0)
output_sp on FPS  HTTP 403 7 ms    {"outcome":"denied","device":"U_FpsTest1","role":"output_sp","axis":null,"journalId":2,"code":"denied","detail":"'output_sp' is in none of U_FpsTest1's allowed actions ['current', 'reset']"}
current_sp on FPS HTTP 502 49 ms   {"outcome":"refused","device":"U_FpsTest1","role":"current_sp","axis":null,"journalId":3,"code":"pv_write_failed","detail":"CA write to undulator:fpstest1:current_ramp:SP failed: undulator:fpstest1:current_ramp:SP: Write access denied"}
twr on m1         HTTP 200 1.21 s  {"outcome":"accepted",…,"role":"twr","journalId":4}   (restored, m1.RBV 2.0)
twr, read key     HTTP 403 4 ms    {"code":"insufficient_write_access","detail":"API key does not have write access"}
```
Journal rows (SQLite, `time` rounded):
```json
{"id": 1, "time": 1793026754.41, "operator": "c3-exercise", "key_name": "env:2", "device": "U_SimMotor1", "role": "twf", "axis": "m1", "value": "1", "outcome": "accepted", "canopy_code": null, "detail": null}
{"id": 2, "time": 1793026754.48, "operator": "c3-exercise", "key_name": "env:2", "device": "U_FpsTest1", "role": "output_sp", "axis": null, "value": "1", "outcome": "denied", "canopy_code": "denied", "detail": "'output_sp' is in none of U_FpsTest1's allowed actions ['current', 'reset']"}
{"id": 3, "time": 1793026754.54, "operator": "c3-exercise", "key_name": "env:2", "device": "U_FpsTest1", "role": "current_sp", "axis": null, "value": "0.5", "outcome": "refused", "canopy_code": "pv_write_failed", "detail": "CA write to undulator:fpstest1:current_ramp:SP failed: undulator:fpstest1:current_ramp:SP: Write access denied"}
```
(The put names the family's roles: FAST-PS `output_sp` is in action `output`,
`current_sp` = `current_ramp:SP` is in action `current`.)

## 4. Checklist steps this lane answers

05 motor tweak journaled with operator: **module adds it** (done above; the
operator is client-asserted, the key name is the authenticated part). 07
refused by IOC: **Hare gives it** (502 passed through); refused by bounds:
the module would add a check (rule 11; Hare clamps silently, C1). 08 denied:
**module adds it** (done). 09 stale page: per-role verdicts done; closing
controls needs the front end to re-poll or the module to stream (§6).

## 5. Canopy shapes extended or worked around — candidate upstream contributions

| # | Title | What it adds | Size | Why SLAC would want it |
|---|---|---|---|---|
| 1 | Module registry by entry point | `canopy.modules` entry-point group (`name, app, lifespan`); `main.py` mounts, composes lifespans, tags and lists `/docs` from it; the ten built-ins register the same way | ~100 LOC + tests | other labs (and SLAC's own facility pieces) add a module without a wrapper or fork; the drift test becomes a registry check |
| 2 | Module scopes from the registry | `MODULES` derived from #1, so `@geecs` keys parse in the env layer and the DB key service | ~20 LOC on #1 | today an extra module forces `@*` keys, which defeats module scoping exactly where a facility wants it |
| 3 | `hare.gateway.put` (+ `subscribe`) | declare the in-process write path (`put(pv, value, *, wait, timeout)`) and the internal-client subscribe Vole already uses | ~20 LOC | Squirrel, Vole and outside modules reach Hare through one declared surface instead of `get_broker()` internals |
| 4 | Write audit / operator on put | optional `operator` (body or header) on `PUT /hare/v1/pvs`, and a put-observer hook on the broker carrying key name, value, outcome | ~60 LOC | C1 found no write audit at all; a hook lets a facility journal subscribe rather than wrap every put |
| 5 | Open error codes | let `CanopyError.error_code` be any `str` (or a module-registered enum) | ~10 LOC | a module's own slugs (`denied`) through the shared handler, one envelope |
| 6 | Per-module Alembic trees | `version_locations` / one branch per module, so a module ships its schema in its package | env.py + docs | a facility's tables without revisions in Canopy's tree |
| 7 | Put-callback timeout ≠ connect timeout | `epics_backend.py:263` uses `ca_timeout_s` (5 s) for a `wait:true` put; a 6 s move reports 502 `pv_write_failed` while the motor still moves. Separate default (`put_timeout_s`) and a `pv_write_timeout` code | ~20 LOC | "refused" and "still going" look the same to a client today |
| 8 | Named env keys | `env:<position>` (`security.py:152`) changes when the list is reordered; allow `token:scopes@mods#name` | ~10 LOC | a stable key name in any audit row |

## 6. Defects, surprises, limits of the spike

- Each device GET is one CA read per role (58 motor, 82 camera):
  `get_value` is a one-shot read unless something subscribes
  (`hare/services/broker.py:552-557`). The freshness window costs 1 s per request.
  The next step is to subscribe as an internal client (as Vole does, #3) and
  judge counters continuously; that is also what step 09 needs.
- All motor fields carry the record's single `tsNs`; the camera's
  `image` role is the IOC's own URL, `http://<container-hostname>:8082/…`,
  unreachable from a browser (an IOC fact, not Canopy's).
- The operator is client-asserted: nothing binds it to the key. A real
  deployment needs a key per operator or the front door's user header (#4).
- Hazard confirm, run lock, bounds (rule 11), snapshots and any front end
  are out of scope, as briefed.

## 7. Questions only SLAC can answer

1. Would you take a module entry-point registry (#1–#2), and is
   `canopy.main:app` meant to be importable as a library meanwhile?
2. Is `hare.gateway.get_broker()` public for outside modules, or will a
   declared `put`/`subscribe` (#3) replace it?
3. Where should a write audit live: Hare's broker hook (#4) or each
   module?
