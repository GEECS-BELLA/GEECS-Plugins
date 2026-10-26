# GeecsCanopy lane C2: a Raccoon directory adapter over GEECS

**Mounted: yes, with no patch to Canopy.** `geecs_canopy_directory.adapter:GeecsDirectoryAdapter`
(Canopy `0a5ee23`, image `ghcr.io/slaclab/canopy:0a5ee23`) loads by dotted path and answers
`/raccoon/v1/{devices,elements,pvs}` for the simulator's native devices with the read key.
The DB path is pinned by tests with GEECS-Core's connection double; there is no DB here, so
the live instance runs with `db=off`. Q5's answer: **yes for device → PV names, in two calls
at most; no for device → roles**: Raccoon carries names only, so the structure is in the patterns.

## 1. How it is mounted

C1's recipe (bind mount + `PYTHONPATH`), the variables in the override's `environment:` rather
than C1's `.env`; `api` only (the worker never loads Raccoon). In the runtime dir
`~/.local/state/geecs-canopy-exercise/`: `geecs-c2/ext/` (`pip install --no-deps --target` of
the spike and GEECS-Core, both pure Python; the image has pydantic 2.13 and PyYAML 6.0),
`geecs-c2/experiments/Undulator/native_devices.yaml` (a temporary configs root: `UC_VisaEBeam6`
aravis_camera, `SimMotor1` epics_motor `[m1, m2]`), and `compose.override.yaml` (two mounts,
`PYTHONPATH=/opt/geecs-ext`, `CANOPY_RACCOON__DIRECTORY__TYPE=geecs_canopy_directory.adapter:GeecsDirectoryAdapter`,
`…__EXTRAS__EXPERIMENT=Undulator`, `…__EXTRAS__EXPERIMENTS_ROOT=/opt/geecs-c2-experiments`,
`…__EXTRAS__DB=off`). C3 runs `api` with its `compose.c3.yaml`, so it is upped as
`docker compose -f compose.yaml -f compose.override.yaml -f compose.c3.yaml up -d api` (C3's
`/geecs` routes still answer). Log: `Raccoon started: adapter='geecs' type='geecs'`.

## 2. The live answer (read key, `127.0.0.1:8000/raccoon/v1`)

```
GET /devices?pattern=*                              200 {"names":["SimMotor1","UC_VisaEBeam6"]}
GET /devices?pattern=sim*&tag=motor_record          200 {"names":["SimMotor1"]}
GET /pvs?pattern=device:SimMotor1                   200 58 names: "undulator:simmotor1:m1.DESC", "…m1.EGU", "…m1.PREC", "…m1.RBV", "…m1.VAL", … "…m2.HOMR"
GET /elements?pattern=SimMotor1:m1:*v               200 {"names":["SimMotor1:m1:rbv","SimMotor1:m1:dmov","SimMotor1:m1:twv"]}
GET /pvs?pattern=element:SimMotor1:m1:val           200 {"names":["undulator:simmotor1:m1.VAL"]}
GET /pvs?pattern=undulator:simmotor1:m?.RBV         200 {"names":["undulator:simmotor1:m1.RBV","undulator:simmotor1:m2.RBV"]}
GET /pvs/undulator:simmotor1:m1.RBV/exists          200 {"exists":true}
GET /pvs/undulator:u_simmotor1:m1.RBV/exists        200 {"exists":false}
GET /pvs?pattern=element:UC_VisaEBeam6:exposure     200 {"names":["undulator:uc_visaebeam6:cam1:AcquireTime_RBV","undulator:uc_visaebeam6:cam1:AcquireTime"]}
GET /pvs?pattern=device:UC_VisaEBeam6               200 119 names (82 roles + 37 write records)
GET /devices?pattern=*&elementType=gateway          503 {"code":"directory_unavailable","detail":"the GEECS DB source is off (db=off)"}
GET /devices?pattern=a|b                            400 {"code":"invalid_query","detail":"not a GEECS directory pattern: 'a|b'"}
GET /health/summary                                 200 adapterHealth {"healthy":true,"latencyMs":11.1,…}
```

## 3. Design decisions

- **Device** = a GEECS device name. **Element** = a PV-owning part of it: a family role
  (`SimMotor1:m1:rbv`, `UC_VisaEBeam6:exposure`) or a gateway-served variable (`U_Jet:Position`,
  `U_Jet:connected`), owning 1–2 PVs, readback first (a role's `write` record, a settable's `:SP`).
- **Patterns** are globs (`*` or `%` any run, `?` one character); `[ ( |` or empty is
  `InvalidQuery`. Device/element globs ignore case (GEECS names carry none); PV globs keep it (EPICS).
  `list_pvs` has two scopes, `device:<glob>` (family order) and `element:<glob>`; any other
  pattern globs PV names, so a literal name answers itself, as `exists`/`validate` require.
- **Filters**: `elementType` = `native` | `gateway`; `tag` = family template (native) or DB
  device type (gateway); `sortBy` = `name`, else device-then-family order.
- **Native**: prefix `pv_name(experiment, device) + ":"` (the namespace's rule), per-axis under
  `<prefix><axis>`, family chosen by `kind` (aravis_camera → ad_camera_chain, epics_motor →
  motor_record). **Gateway**: `geecs_core.db.served_set` over `get_experiment_devices`,
  `…_device_variables`, `get_subscribed_variables`, `…_device_types`, named with `pv_name` /
  `setpoint_pv` / `device_status_pv`; a native device shadows a DB twin (the worker refuses both).
- **The DB down**: `DirectoryUnavailable` from that branch only: `elementType=native` survives;
  an unscoped query is a 503, never a partial answer (Raccoon caches answers as complete, 300 s
  per pattern, 1800 s per `exists`). `db=off`: unscoped = native only, `gateway` = 503 saying so.
- `GeecsDb` is synchronous: an adapter-owned 4-thread pool (the `MemeDirectoryAdapter` pattern).

## 4. The protocol's fit for device → PVs

Raccoon returns `{"names": [...]}` and nothing else: no device → PV relation, no role, type,
writability, readback/setpoint pairing, units or limits. What it costs:
- "PVs of device X" exists only because this adapter invents the `device:` scope; another facility's
  adapter would spell it differently, so a client written against Raccoon is written against
  *an adapter's grammar*, which Raccoon has no way to advertise.
- Device → role → PV takes `list_elements("X:*")` then one `list_pvs("element:…")` per element
  (N + 1 calls; 140 for the two sim devices), or a client parsing element names it was not told
  the shape of. A panel needs the family file anyway for what Raccoon cannot carry (role type,
  `write`, `settled_by`, limits), which is lane C3's ground, not Raccoon's.
- The scopes shadow any PV beginning `device:` / `element:` (none in GEECS). No paging: an
  experiment's gateway set (thousands of PVs) is one list. Registry or DB edits reach clients
  only as Raccoon's caches expire (up to 30 min for `exists`).

## 5. Lines of code and vendored files
Code 299 (`adapter.py` 154, `sources.py` 144, `__init__.py` 1, docstrings included); tests 150
(10 cases: motor per axis, camera with write records, a DB device's gateway PVs through the
double, four bad queries, the failing DB double, `health()` with the DB off, Canopy's contract
suite); `pyproject.toml` 22. Tests run in a Python 3.12 venv with the Canopy clone installed
editable (`PYTHONPATH=<clone>` for `tests.canopy.raccoon.adapters.contract`). **Bite proved**:
dropping the settable `:SP` in `gateway_devices` fails `test_a_db_device_lists_the_gateways_served_pvs`.

**Vendored**: `motor_record`, `ad_camera_chain`, `easydriver`, `fastps`, `scope_contract`
`.json`, verbatim from `GEECS-Schemas/geecs_schemas/epics/families/` at `7f38c12ac` (their last
change on `origin/feat/web-estate`, PR #1148; tip `048c468b1`), read by a 17-line loader.

## 6. Defects and surprises
1. **`U_SimMotor1` is not the sim motor's GEECS name.** The prefix rule derives
   `undulator:u_simmotor1:` from it (`exists: false` above); `SimMotor1` is the name that derives
   the sim's `undulator:simmotor1:`, as GeecsBluesky's and GeecsPanels' tests use. The sim's
   supply is `undulator:eztest1:` (C1), not `eztest1:`.
2. **The registry cannot name a supply or a scope.** It knows `aravis_camera` and `epics_motor`;
   the families also declare `setpoint_device` (easydriver **and** fastps) and `array_detector`
   (scope_contract). A kind does not pick a family when two share one: a registry entry needs its
   family (`template`) named. `eztest1`, `fpstest1` and the scopes stay outside the directory.
3. **GEECS-Core cannot be installed into Canopy's environment by its own metadata**: it declares
   Python `>=3.11,<3.12`, Canopy requires `>=3.12` (image 3.12.15). It runs fine on 3.12; the mount
   used `pip --target` from a 3.11 host and the test venv `--ignore-requires-python`.
4. Not exercised: the image has no `mysql-connector-python` (a DB host needs it in the mount or a
   derived image). The registry is read raw (kind, axes), not through `NativeDevices`, to keep the
   brief's dependencies; production validates with the schema.
5. `health()` reads every source (four DB queries with the DB on) per `/health/summary` request;
   a production adapter keeps a TTL snapshot. `EXTRAS__` keys arrive lowercased.
6. Canopy's contract suite lives in its `tests/` tree, absent from the wheel and the image: an
   out-of-tree adapter must clone Canopy to run it.

## 7. Candidate upstream contributions (phrased for SLAC)
1. **A structured device lookup**: an optional adapter method, e.g. `describe_device(name) ->
   list[{pv, element, attribute}]`, behind `GET /raccoon/v1/devices/{name}/pvs` (501 without it).
2. **Ship the conformance suite in the package** (`canopy.raccoon.adapters.contract` or a
   `canopy.testing` module), so out-of-tree adapters test against the installed Canopy.
3. **Let an adapter describe its grammar** (a `pattern_help` string in `/raccoon/v1/health/summary`
   or OpenAPI), since patterns are opaque by design. 4. **An admin cache flush** for Raccoon.

GEECS-side, not upstream: relax GEECS-Core's Python ceiling to `<3.13`; give the native registry
a `template` per entry (or a `setpoint_device` / `array_detector` kind naming one).
