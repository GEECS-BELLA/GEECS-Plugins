# GEECS Queueserver Deployment

How to install the bluesky-queueserver RE Manager as a service on a dedicated
Ubuntu host, what local services it depends on, and how an operator verifies
the manager is ready to accept scan requests. **A running `geecs-qserver`
means ready**: the companion `geecs-qserver-ready` oneshot opens the worker
environment and asserts the plan list after every (re)start (§ 2, § 3). The launch mechanics are in
`../launch_re_manager.sh`; day-to-day operator checks and known failure modes
are in `../README.md`.

---

## 1. Host prerequisites

Use a dedicated Ubuntu 22.04 host on the lab network. The worker must be able
to reach:

- the Channel Access gateway host,
- the GEECS MySQL database,
- the Tiled server when Tiled publishing is enabled (reached by the
  `geecs-tiled-writer` unit, never by the engine — § The Tiled writer),
- the mounted GEECS data share used for scan-folder claim, ScanInfo writes,
  native per-shot saves, and legacy s-file export.

Install Python 3.11, Poetry, Redis, and a checkout of this repository:

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv redis-server
sudo systemctl enable --now redis-server.service
```

Clone the repository if the host does not have it yet:

**Everything below runs AS THE SERVICE ACCOUNT** (the unit's `User=`,
`geecs` in this template — create it *with a login shell*, or substitute
`sudo -u geecs -H bash` wherever `sudo -u geecs -i` appears; a
`/usr/sbin/nologin` account fails `-i` with "account is currently not
available"): Poetry keys project virtualenvs under the
*invoking user's* cache, so an env installed by an admin account is
invisible to the service — the unit would crash-loop on a fresh,
dependency-less env while admin-side verification passes. Clone, install,
and verify as the service user (`sudo -u geecs -i` or a direct login):

```bash
# <root> = GEECS_CHECKOUT_ROOT from the host's site.env (the service
# account's home, or /opt/geecs after one chown); the worker's clone is
# <root>/qs-checkout — deploy/bootstrap_host.sh creates it, or by hand:
sudo -u geecs git clone https://github.com/GEECS-BELLA/GEECS-Plugins.git <root>/qs-checkout
```

Install Poetry **as the service user** by the method approved for the host
image (note where it lands — the official installer uses `~/.local/bin`,
which systemd's service PATH does not include; the unit template therefore
takes poetry's absolute path). Then install the worker environment,
pointing poetry at 3.11 explicitly (jammy's default `python3` is 3.10,
which this package refuses — the repo's documented top environment
failure):

```bash
sudo -u geecs -i
cd <root>/qs-checkout/GeecsBluesky
poetry env use python3.11
poetry install --extras "ca tiled qserver optimize"
```

The `qserver` extra is the queueserver dependency bundle. If that extra has
not landed on the deployment branch yet, install the branch that provides it
before enabling the systemd unit.

Redis is provided by the Ubuntu `redis-server` package and should stay bound
to `127.0.0.1`; see `redis.conf-notes.md` for the Redis-specific notes and
the `vm.overcommit_memory=1` sysctl fix.

### Shared config

The standard GEECS config file for the service account
(`~/.config/geecs_python_api/config.ini`, mode 600) is **rendered from
`site.env` by `deploy/bootstrap_host.sh`** when it does not exist yet —
do not pre-create an empty one (an empty file is treated as absent, a
non-empty one is never overwritten). Then add the Tiled `api_key` by
hand. To write it entirely by hand instead, follow the reference block in
`docs/tutorials/getting_started.md`:

```bash
sudo -u geecs sh -c 'install -D -m 600 /dev/null "$HOME/.config/geecs_python_api/config.ini"'
sudo -u geecs sh -c 'editor "$HOME/.config/geecs_python_api/config.ini"'   # fill it before running the bootstrap
```

Use the [Getting Started config.ini section](../../../docs/tutorials/getting_started.md)
as the key-by-key reference. Do not invent a queueserver-specific config
format; this worker reads the same `~/.config/geecs_python_api/config.ini`
contract as the rest of GEECS-Plugins.

At minimum, verify that the service account's config resolves:

- the GEECS data root on the mounted data share,
- the scanner configs repository,
- the scan-analysis configs path (`[Paths] scan_analysis_configs_path`) —
  required by optimizer configs with diagnostic measurements. The native
  measurement compiler resolves their documents under `analyzers/` before
  claiming a scan; scalar-only optimization does not need this path,
- database credentials through the normal `Configurations.INI` chain,
- Tiled connection details when Tiled publishing is enabled,
- optional `[epics] ca_addr_list` if the unit-level `EPICS_CA_ADDR_LIST`
  override is not being used.

A missing key does not fail the service at startup — the worker loads and
the plan list populates, with only a warning in the journal (e.g. "Could
not determine base data path", "No Tiled URI configured — Tiled storage
disabled"). Grep the journal for `WARNING`/`ERROR` after the first
`environment open` rather than trusting a clean-looking `qserver status`
(2026-08-21 live-checkpoint lesson: three keys were discovered missing
one failed scan at a time).

The unit takes `EPICS_CA_ADDR_LIST` (+ `EPICS_CA_AUTO_ADDR_LIST=NO`) from
`/etc/geecs/site.env` via `EnvironmentFile=`. To see the active Channel
Access target: `systemctl show geecs-qserver -p EnvironmentFiles` names the
file, `cat /etc/geecs/site.env` shows the value (world-readable by design —
it holds no secrets).

### Data share

Mount the production data share before starting the manager. The worker must
write to the same scan-folder tree the analysis side reads; otherwise scan
number claim, ScanInfo creation, native asset references, and s-file export
will fail or point at the wrong location.

Validate the mount as the service account before enabling the unit:

```bash
sudo -u geecs test -d /mnt/geecs-data
sudo -u geecs test -w /mnt/geecs-data
```

Replace `/mnt/geecs-data` with the path configured in `config.ini`.

---

## 2. Install the service unit

`geecs-qserver.service` is a **template** (see the
[Site Profile](../../../docs/platform/site_profile.md)): the service
account, checkout root, and poetry path are `@PLACEHOLDER@` holes filled
by `deploy/render_units.sh` from the host's `site.env`; the experiment
(`QS_EXPERIMENT`) and CA addressing (`EPICS_CA_ADDR_LIST` +
`EPICS_CA_AUTO_ADDR_LIST=NO`) reach the process from the same file via
`EnvironmentFile=`. Nothing is typed into the unit by hand.

The queueserver is **two units** from the same template directory
(plus the Tiled writer's, its own service — see § The Tiled writer):
`geecs-qserver.service` (the manager) and `geecs-qserver-ready.service`, a
`Type=oneshot` readiness assertion ordered after it (`Requires=` +
`PartOf=`, so it re-runs on every manager restart). bluesky-queueserver
treats `environment open` as an operator gesture — a freshly restarted
manager knows *no* plans until something opens its worker environment, and
until then every submission is refused with "Plan ... is not in the list of
allowed plans" while `qserver status` looks healthy (live 2026-09-04,
GEECS-Plugins#793). For GEECS a running service means ready, so the
readiness unit runs `geecs-qserver-ensure-ready`: wait for the manager,
open the environment if closed, wait for idle, then **assert
`plans_allowed` lists every GEECS plan** (`geecs_bluesky.plan_names`) —
restoring the lists once from the worker's on-disk copy when they read
empty or incomplete with the environment up (a timed-out list download,
GEECS-Plugins#838) — exiting non-zero with a precise message otherwise. A
separate unit on
purpose: the manager's start is never blocked by the optimize-stack import
warm-up, a failed open shows as one failed unit rather than a crash-looping
manager, and `systemctl restart geecs-qserver-ready` is the recovery
gesture. The entry point asserts the manager on *this* host (loopback);
the only override is an optional `QS_CONTROL_ADDR=tcp://host:60615` key in
`site.env` (it never reads the client-side `[qserver]` section of the
service account's `config.ini` — that names the worker *clients* talk to).

```bash
# as the service account — or run deploy/bootstrap_host.sh for the whole host
<root>/qs-checkout/deploy/render_units.sh /etc/geecs/site.env ~/deploy-staging
sudo install -m 0644 ~/deploy-staging/geecs-qserver.service ~/deploy-staging/geecs-qserver-ready.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now geecs-qserver.service geecs-qserver-ready.service
```

Check both units and the journal:

```bash
systemctl status geecs-qserver.service geecs-qserver-ready.service
journalctl -u geecs-qserver.service -n 100 --no-pager
journalctl -u geecs-qserver-ready.service -n 20 --no-pager   # "ready: N allowed plans" or NOT READY: <why>
```

The unit intentionally carries a commented-out
`Environment=QSERVER_ZMQ_PRIVATE_KEY_FOR_SERVER=` line. CurveZMQ control-plane
key management is still open; the design item is tracked in GEECS-Plugins
issue #660.

### Document stream (no extra unit)

The launcher starts a `bluesky-0MQ-proxy` (5567 in / 5568 out) alongside
Redis when one is not already answering; the startup profile publishes
bluesky documents into it for GUI progress (see `../README.md`, "Document
stream"). The proxy runs inside the unit's cgroup on purpose — it is
stateless and dies with the unit; no second systemd unit is needed. Caveat
of that choice: systemd supervises only the main process, so a proxy that
dies *alone* is not restarted until the next manager restart — the symptom
is GUI progress going quiet while scans keep running fine (the proxy's
stderr goes to the journal; the worker also logs a warning at startup when
nothing answers on the publish port). Override the ports with
`Environment="QS_DOC_PROXY_IN=…"` / `QS_DOC_PROXY_OUT=…` (paired with
`QS_DOC_PUBLISH_ADDR` for the worker side), or disable with
`QS_DOC_PROXY=OFF` + `QS_DOC_PUBLISH_ADDR=OFF`.

### Network ports

Queue clients on other hosts (notebooks; the scanner and the MCP run on
the worker host itself) need to reach, on the worker host:

- **60615** — the RE Manager control socket (`bluesky-queueserver-api`),
- **60625** — the manager's console-output stream (log tail / failed-move
  reason lines; text, not documents),
- **5568** — the document-stream out port (live per-shot progress).

Redis (6379) stays loopback-only. The proxy's in port (5567) is only
ever used by the worker itself (same host) — nothing remote needs it,
though the proxy binds all interfaces; firewall it with the rest.

### External subscribers

The document-stream out port (**5568**) is the supported subscription
point for clients beyond the web scanner — the contract OSPREY's bridge and
any future live-progress client build on (#727 item 3). What "supported"
means:

- **Where.** Subscribe to `<worker-host>:5568` from any host that can
  reach the worker; the proxy is a fan-out, so subscribers never touch
  the manager socket, Redis, or the in port. Add 5568 to the same
  firewall allow rule as 60615/60625; leave 5567 closed. The in-repo
  subscribers are the reference practice: the scanner's
  `geecs_scanner/service/streams.py` (`ProgressCache`),
  and GEECS-MCP's `geecs_mcp/scans/progress_stream.py` (`ProgressCache`)
  — each a `bluesky.callbacks.zmq.RemoteDispatcher` on the `doc_addr`
  that `geecs_bluesky.qs_client` reads from the `[qserver]` section of
  `config.ini` (default `<host>:5568`). The `RemoteDispatcher` snippet
  lives in `../README.md`, "Document stream".
- **Wire format.** bluesky's `Publisher` defaults: one zmq frame per
  document, `b"<prefix> <name> <pickled doc>"` with an empty prefix —
  split on the first two spaces, decode the name, `pickle.loads` the
  remainder (that is what `RemoteDispatcher` does). A subscriber is
  therefore a Python process; a non-Python subscriber needs a
  JSON/msgpack serializer added on the worker side first — not offered
  today.
- **Late joiners.** PUB/SUB has no replay: a client connecting mid-scan
  never sees that run's `start`. Ignore documents of a run whose `start`
  you did not see until the next `start` arrives (or resolve the run via
  Tiled by its `run_start` uid) — the in-repo subscribers ignore events
  they cannot attribute.
- **Transport posture.** Plaintext on the lab control network, same as
  the control socket (60615): no encryption, no client auth. A client
  that fronts this stream for users beyond the worker host should say so
  explicitly (a visible plaintext setting, not a silent default) — the
  treatment OSPREY's bridge gives the control socket (als-apg/osprey#817).
  CurveZMQ on the document stream is decided together with the
  control-plane keys, which is issue #660's question, not this
  document's.
- **Stability.** Document shape follows `../../EVENT_SCHEMA.md`; the
  `geecs_event_schema` start-document key carries the version. The stream
  carries *every* RunEngine document — `resource`/`datum` when non-scalar
  saving is on, the free-run `flush` stream — so tolerate unknown document
  names and stream names, not only unknown fields. Additive changes — new columns, new metadata keys — do not bump
  the version; only a rename, removal, or semantics change bumps it.

---

## The Tiled writer

Since GeecsBluesky 0.103.0 the engine **never talks to Tiled**: it spools
every document of a run to one JSON Lines file, and a separate process,
`geecs-tiled-writer`, registers each complete file in the catalog at the
run's close, off the engine thread (`../../CLAUDE.md` § "Tiled: the
spool and the writer service" has the design; the numbers: ~25–28 s per
run on the SQLite catalog, which used to hold the engine at the stop
document). The writer is its own unit, `geecs-tiled-writer.service`, a
template beside this one, rendered and installed through the same path.
A worker on 0.103.0 **with no writer running registers nothing**: the
spool files pile up (nothing is lost — the first writer to run catches up
the whole backlog, oldest first) and no new run appears in Tiled.

### Install

The writer runs from the **worker's** clone and env (`qs-checkout`,
`GeecsBluesky`, the same extras — one spool line format, one env). The
`poetry install` of § 1 registers the `geecs-tiled-writer` console script;
nothing more to install. Render, install, enable — `deploy/bootstrap_host.sh`
does the unprivileged part for the whole host (`--only tiled-writer` for
just this unit; its extras are the worker's set on purpose, so a rerun
never strips the worker's env):

```bash
# as the service account
<root>/qs-checkout/deploy/render_units.sh /etc/geecs/site.env ~/deploy-staging
sudo install -m 0644 ~/deploy-staging/geecs-tiled-writer.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now geecs-tiled-writer.service
journalctl -u geecs-tiled-writer -n 5 --no-pager    # "geecs-tiled-writer 0.103.x: spool /var/lib/geecs-tiled-writer/spool → http://…"
```

Three units share one directory. `StateDirectory=geecs-tiled-writer` is
declared by **both** `geecs-tiled-writer.service` and
`geecs-qserver.service` (systemd allows it; both run as the service
account, so ownership agrees), and `GEECS_TILED_WRITER_STATE=/var/lib/geecs-tiled-writer`
is set explicitly in the writer's, the queueserver's and the scanner's
units — the engine reads only the variable, never `$STATE_DIRECTORY`, and
a two-process contract is never left to a per-unit default. Not a site
value: the same path on every host, so it lives in the units, not in
`site.env`. The catalog (`[tiled] uri` + `api_key`) comes from the service
account's `config.ini`, as for every other Tiled client.

### Switching from a by-hand writer to the unit

The 0.103.0 hardware verification left the worker with a writer started
by hand — a transient user unit (`systemd-run --user --unit
tiled-writer-manual …`) spooling under the developer default,
`~/.local/state/geecs-tiled-writer`. **Two writers on one spool directory
race on the same files**, and the engine reads `GEECS_TILED_WRITER_STATE`
only when its worker environment opens, so the switch is an ordered
hand-over, in a window with no scan running:

1. **Let the by-hand writer drain.** `cat
   ~/.local/state/geecs-tiled-writer/heartbeat.json` (as the service
   account) until `pending` is 0 and `in_progress` is 0; no complete file
   is then waiting under `~/.local/state/geecs-tiled-writer/spool/`.
2. **Stop it:** `systemctl --user stop tiled-writer-manual` (a transient
   unit is gone once stopped; there is nothing to disable). The `.done`
   files it leaves behind are inert — delete the directory once the unit
   below has run for a day, or leave it.
3. **Install the re-rendered units** — the queueserver's now carries the
   `StateDirectory=` and `Environment=` lines, the scanner's the
   `Environment=` line — with the writer's, then `daemon-reload`.
4. **Start the unit:** `sudo systemctl enable --now geecs-tiled-writer`.
   systemd creates `/var/lib/geecs-tiled-writer`; the journal's first line
   names that spool directory.
5. **Restart the queueserver:** `sudo systemctl restart geecs-qserver`
   (the readiness oneshot reopens the environment; the journal's `Tiled
   spool subscribed` line names the new directory). Until this restart the
   engine still spools to the **old** directory, where nobody sweeps —
   that is why step 1 comes first and why no scan may run in between. A
   run that lands there anyway is registered with one sweep of the writer
   pointed at the old directory:
   `GEECS_TILED_WRITER_STATE=~/.local/state/geecs-tiled-writer poetry run geecs-tiled-writer --once`.
6. **Restart the scanner:** `sudo systemctl restart geecs-scanner` — it
   reads the heartbeat at the new path for its "tiled writer" chip.
7. Run a 3-shot count; `journalctl -u geecs-tiled-writer` shows
   `Scan0NN (<uid>): registered in X s (N documents)`; the spool file is
   `.jsonl.done` under `/var/lib/geecs-tiled-writer/spool/`.

### The heartbeat, and what its words mean

`/var/lib/geecs-tiled-writer/heartbeat.json`, rewritten atomically after
every sweep (2 s): `pid`, `version`, `started_at`, `last_sweep`,
`sweep_interval`, `tiled_uri`, `tiled_reachable`, `last_ok` (the last
successful registration), `last_error` (what went wrong in the **latest**
sweep — a clean sweep clears it; the journal keeps history), `pending`
(complete files waiting), `in_progress` (runs still open, or unfinished
files not yet past `--orphan-after`), `failed` (files set aside as
`.jsonl.failed`), `done` (registered since this process started),
`registered` (the last 20 uids). **A warning surface, never a gate**: no
preflight and no plan refuses a run over it (owner's ruling, 2026-09-25) —
with the spool a dead writer loses nothing.

Read it three ways: `cat` on the host; the scanner's `GET /health` →
`tiled_writer` and its "tiled writer" chip; `scripts/fleet_status.sh`'s
"Tiled writer" row (read from the scanner's `/health` — the probe runs
from an operator's machine). The scanner's word, from the measured
25–28 s per run:

| Word | When | Meaning |
|---|---|---|
| `ok` | a fresh heartbeat, Tiled reachable, `failed` 0, `pending` ≤ 1 | the writer keeps up: at most the last run is being registered |
| `degraded` | no heartbeat, or `last_sweep` older than 3 sweep intervals (~6 s: the writer is down or wedged); Tiled unreachable; `pending` 2 | nothing is lost — runs keep spooling — but nothing reaches Tiled until it is fixed |
| `failed` | `failed` > 0 (a file set aside for an operator), or `pending` ≥ 3 (a backlog: the writer is not keeping up, or every registration fails and is backing off — `last_error` says which) | someone has to look |

**A stale heartbeat** with the unit `active` means the process is wedged
(a sweep that never returns — a Tiled call with no timeout): `sudo
systemctl restart geecs-tiled-writer`. Mid-registration is safe: the
next start finds the half-registered container and registers the run
again from the spool, exactly once (verified live 2026-09-25 with
`kill -9`).

**`.jsonl.failed` files** are an operator's call. A corrupt file (a
malformed line, no start document) is set aside at once and will not heal;
anything else was retried on the backoff schedule for `--max-attempts`
(15, ~75 min) first, its half-registered container removed, and the reason
is in the journal (`registration failed 15 time(s) — giving up`). To try
again once the cause is fixed: rename `<file>.jsonl.failed` back to
`<file>.jsonl` — the next sweep picks it up (a container of that uid is
deleted and registered again). Nothing prunes `.failed` files.

Knobs (`--sweep-interval`, `--keep-days` 7, `--max-attempts` 15,
`--max-backoff` 600, `--orphan-after` 1800) are CLI arguments with their
defaults in the unit; change one with a drop-in (`sudo systemctl edit
geecs-tiled-writer`, an `ExecStart=` override), never by editing the
rendered unit.

---

## 3. Verify the manager

Run these commands from a shell that has the `qserver` CLI available. The
Poetry environment from the checkout is the expected source:

```bash
sudo -u geecs -i
cd <root>/qs-checkout/GeecsBluesky
poetry run qserver status
```

Expected: the manager responds, Redis is reachable, the worker environment
exists and is `idle` — the readiness unit opened it. Nothing to type: if
`qserver status` shows `worker_environment_exists: False` the readiness
unit failed, was not installed, or ran fine and the RE worker child died
later while the manager survived (no systemd event fires for that — the
unit stays `active (exited)`; the scanner/MCP preflight refusal is what
names the gesture); read its journal, fix the cause, and re-run it (the
same command a fresh clone's first deploy uses):

```bash
journalctl -u geecs-qserver-ready.service -n 50 --no-pager
sudo systemctl restart geecs-qserver-ready.service
# the equivalent by hand, in the worker's env (exit 0 = ready, 1 = NOT READY: <why>):
poetry run geecs-qserver-ensure-ready
```

`qserver environment open` still works as the raw gesture, but it does not
check the plan list — the readiness entry point does, which is the
invariant that matters (`plans_allowed` non-empty and containing every
GEECS plan). If the open itself fails, inspect the manager's journal first:

```bash
journalctl -u geecs-qserver.service -n 200 --no-pager
```

Follow the troubleshooting section in `../README.md` for the known
queueserver failure shapes — it is the single maintained list; symptoms and
fixes are not restated here.

After the environment is open, submit only the smoke-test queue items approved
for the current startup profile. Do not use a deployment host to discover
machine-control behavior ad hoc; the manager is the production execution
surface once this service is enabled.


## Native optimization acceptance

Install the worker's `optimize` extra alongside `ca tiled qserver`. The scanner
needs no ImageAnalysis, Xopt or Torch dependency. The worker warms numerical
imports on the profile thread before readiness. The PVA monitor must receive an initial image
within five seconds before the plan arms or claims a scan; verify the host's
PVA address configuration. Frame acquisition uses live arrays, never the
partly-written scan files, with the camera timestamp joined within 1 ms.

Copy the six v1 keeper documents from GEECS-Schemas' optimizer fixtures to
the configs repository only after the schema change lands. Keep the two
HiResMagCam legacy documents with a `# LEGACY` header until their diagnostic
exists; remove ebeam_source_opt, hexapod_alignment and multi_device_example.
Validate diagnostics on the worker before an operator day. No deployed
configs were changed by the implementation branch. The resolver lists only
schema-valid native configs; the scanner disables Optimize when that list is
empty. An unmigrated legacy corpus therefore offers no broken choices.

**Beam-free smoke test passed:** Scan010 of 26_0915 ran
`bax_alignment_simulation`, 3 iterations × 2 HTU-NoGas shots, and restored both
original magnet setpoints. Scalar files and Xopt dump passed. The resulting
Tiled column-encoding fix passed a hardware-free replay; the original live
Tiled entry is partial. Evidence and remaining limits are in the #880 /
#920 PR history.

**OWED hardware acceptance:** run TopViewMax with
beam, 5 shots × 10 iterations; require five valid frames per iteration,
compare the objective with saved PNGs after the run, and record RSS before
and after. Finally submit TopViewMax from the scanner, observe live iteration
and best values, and exercise Set to best once while idle. Relative-pseudo
acceptance must show restoration followed by the explicit physical best move.
