# Site profile + fleet uniformity

Execution plan, 2026-09-03 (Sam + Claude session that built `/fleet-status`).
Status: **Phases 1+2 built as PR #777** (same day; merge order #775 → #776 →
#777). **Phase 3 executed 2026-09-04** (cutover on the interim host; fleet map
updated in PR #792; fixes it surfaced: #782 worktree detection, #791
pre-profile template refusal). Written for whoever picks it up; assumes no context from the
originating conversation. Where this document and #777 differed after the
build, the document was corrected to the built behaviour (2026-09-03,
after a Codex review of this PR).

## Why now

`/fleet-status` (PR pending on `feat/fleet-status-skill`) showed the
current services box as it is, not as the fleet map describes it:

| Service | Observed | Fleet map says |
|---|---|---|
| RE Manager :60615, doc proxy :5568 | hand-started processes, no unit (parent = a dead shell / PID 1) | systemd `geecs-qserver` |
| GEECS-MCP :8100 | hand-started, baked venv at `~/geecs-mcp-venv` | systemd unit, venv at `/opt/geecs-mcp-venv` |
| Capture daemon | `systemctl --user` unit (linger on) | system unit `geecs-capture` |
| CA gateway clone | files on disk = commit 9c9681de (2026-08-25) while HEAD says 28749780 — the branch ref advanced without a checkout, so the gateway runs three-week-old GEECS-Core/GEECS-Schemas | clean pinned clone |
| PVA fleet | 4 of 13 roster hosts unreachable | 9 hosts |

Three forcing functions make fixing this now worth more than waiting:

1. **A higher-end services box arrives in ~3 weeks** (late Sep 2026) and
   the fleet migrates to it. The recipe is what migrates, not the
   processes. Making the current box uniform *is* rehearsing that recipe
   on a host we can afford to break.
2. **Other GEECS facilities** will want this fleet — different networks,
   or the same network with a different DB host. Every facility-specific
   literal we bake in now is a fork later.
3. **The docs must stay true.** Deployment runbooks, `CLAUDE.md`s, and
   the mkdocs site are the contract other people (and agents) execute.
   Every phase below carries its doc obligation as a deliverable, not an
   afterthought.

Scope is **HTU/Undulator only** for the build; portability is baked in
by *where values live*, not by testing at a second site.

## The design move: one home per facility value

A facility value (experiment name, gateway address, serving interface,
timezone, checkout root, service account) may live in exactly one of
two places, and committed files may only carry it as an **example or a
placeholder**:

| Side | Home | Consumers | Exists today? |
|---|---|---|---|
| Client | `~/.config/geecs_python_api/config.ini` (`[epics] ca_addr_list`, `[tiled]`, `[qserver] host`, `[mcp]`, `[Experiment]`, `[Paths]`) | every Python client and service process; `scripts/lab_status.sh`, `scripts/fleet_status.sh` | yes — the fleet contract since the PythonAPI days |
| Host | **`site.env`** — one file per service host, consumed by systemd via `EnvironmentFile=` and by the bootstrap script | unit files, the bootstrap/render scripts, unit-time values (CAS interface + beacon, TZ, experiment for CLI args) | **no — this plan creates it** |

On a service host `site.env` is the root: the bootstrap script renders
the host's `config.ini` *from* it on a fresh host, and on an existing
host shows the diff between the two so disagreement is visible and
reconciled by hand (the file also carries the hand-entered Tiled key, so
the bootstrap never overwrites it). Client machines keep their own
`config.ini` exactly as today.

### `site.env` vocabulary (v1)

```ini
# /etc/geecs/site.env — the host's facility profile. One per service host.
# systemd EnvironmentFile syntax: comments ONLY on their own lines (a
# trailing "# comment" becomes part of the value); quote values with spaces.

# identity
GEECS_SITE=htu
GEECS_EXPERIMENT=Undulator
QS_EXPERIMENT=Undulator

# service account + layout (install-time: fill the unit placeholders)
GEECS_SERVICE_USER=geecs
GEECS_SERVICE_HOME=/home/geecs
GEECS_CHECKOUT_ROOT=/home/geecs
GEECS_POETRY=/home/geecs/.local/bin/poetry
GEECS_REPO_URL=https://github.com/GEECS-BELLA/GEECS-Plugins.git

# EPICS — the names EPICS itself reads, delivered straight into the process
EPICS_CA_ADDR_LIST=192.168.6.14
EPICS_CA_AUTO_ADDR_LIST=NO
EPICS_CAS_INTF_ADDR_LIST=192.168.6.14
EPICS_CAS_BEACON_ADDR_LIST=192.168.6.255

# time — the name libc reads
TZ=America/Los_Angeles

# fleet endpoints + data share (install-time: rendered into config.ini)
GEECS_TILED_URI=http://192.168.6.14:8000
GEECS_QSERVER_HOST=192.168.6.14
GEECS_QS_DOC_ADDR=localhost:5568
GEECS_DATA_ROOT=/mnt/hdna2/data
GEECS_CONFIGS_ROOT="/mnt/hdna2/software/control-all-loasis/HTU/Active Version/GEECS-Plugins-Configs"
```

Runtime keys carry the **downstream names** (`TZ`, `EPICS_*`,
`QS_EXPERIMENT`) precisely because nothing re-maps them: `EnvironmentFile=`
puts them in the process environment as-is. There is no `GEECS_TZ` and no
`Environment=TZ=${GEECS_TZ}` line anywhere — see the gotcha below.

Ports are **not** site values: the fleet map fixes them (5064, 8000,
8200, 8100, 60615/60625/5568, 5075/5076) and every client assumes them.

### The systemd gotcha that shapes the units

systemd expands `${VAR}` from `EnvironmentFile=` **only in command
arguments** (`ExecStart=` and friends, after the executable). It does
not expand in `WorkingDirectory=`, in the executable path (first token of
`ExecStart=`), in `User=`, **or in `Environment=` lines** —
`Environment=TZ=${GEECS_TZ}` would set the literal string. So a unit
template has two kinds of holes and one non-hole:

- **Install-time placeholders** (`@CHECKOUT_ROOT@`, `@SERVICE_USER@`,
  `@SERVICE_HOME@`, `@POETRY@`, `@SITE_ENV@`) filled by the render script
  when the unit is written to `/etc/systemd/system/` — paths and identity.
- **Runtime arguments** (`--experiment ${GEECS_EXPERIMENT}`,
  `--doc-addr ${GEECS_QS_DOC_ADDR}`, `--processing-configs
  "${GEECS_CONFIGS_ROOT}/scan_analysis_configs"` — a quoted `"${VAR}"` is
  substituted as one argument, spaces survive) read from `site.env` at
  start.
- **Environment the services read directly** (`TZ`, `EPICS_CA_ADDR_LIST`,
  `EPICS_CAS_INTF_ADDR_LIST`, `QS_EXPERIMENT`, …) needs no hole at all:
  `site.env` defines those variables under their real names and
  `EnvironmentFile=` delivers them. No template carries an
  `Environment=` line for a site value.

The render step is one `sed` invocation; we already do this by hand
today (the checked-in templates carry `geecs` and `/home/geecs`).

### Checkout layout

`${GEECS_CHECKOUT_ROOT}/<service>-checkout/` — the interim host's
convention, promoted to the rule. The qserver runbook's
`/opt/geecs/GEECS-Plugins` layout is the same pattern with a different
root; the new box picks its root in `site.env`. The one-clone-per-service
rule and its two exceptions (qserver + capture share; MCP bakes) stand
unchanged — see the fleet map.

## Phases

Each phase is one PR unless noted, lands per `/land`, and lists its
**live check**, its **doc obligations**, and any **Sam step** (the
service account has no passwordless sudo, so writing to
`/etc/systemd/system` and `/etc/geecs` is always Sam's one-liner; agents
stage everything unprivileged first).

### Phase 0 — land `/fleet-status`

The pending PR. It is the measuring stick for every phase after it:
"done" means the table shows systemd / clean / matching versions.

### Phase 1 — the contract (docs + example, no service change)

- `deploy/site.env.example` — the vocabulary above, HTU values as the
  worked example, every line commented.
- `docs/platform/site_profile.md` — the contract page: the two homes,
  what changes per facility, the systemd gotcha, the checkout layout,
  how a second facility onboards (copy the example, fill it, run the
  bootstrap). Linked from the fleet map and the platform index; added
  to mkdocs nav.
- Root `CLAUDE.md`: a "Facility values have one home" invariant under
  cross-package invariants, pointing at the page. `CONTRIBUTING.md`
  distills it. `docs/tutorials/getting_started.md` gains one sentence
  saying `config.ini` is the client half of the site profile.
- `/land` scope check gains a line: a diff that adds a lab literal
  (`192.168.`, `Undulator`, `/home/geecs`, `America/Los_Angeles`)
  outside a docstring, example block, or `analyzers/<Experiment>/`
  package is flagged in the PR body with where the value should live.
  Instruction only — no hook.
- Live check: none. Docs build (`mkdocs build --strict`).

### Phase 2 — unit templates + the bootstrap (code, no deploy yet)

- The five unit templates (`GeecsCAGateway/deploy`, `GeecsBluesky/qserver/deploy`,
  `GeecsBluesky/capture/deploy`, `GEECS-DataPortal/deploy`, `GEECS-MCP/deploy`)
  rewritten to the placeholder + `EnvironmentFile=/etc/geecs/site.env`
  form. The capture unit gains the `--doc-addr localhost:5568` argument
  the live user unit already carries (the runbook's version lacks it).
  The MCP venv path becomes `@CHECKOUT_ROOT@/geecs-mcp-venv` (matches
  the live host; `/opt` needed sudo for no benefit).
- `deploy/render_units.sh SITE_ENV OUT_DIR` — renders every template
  into a staging directory and prints the `sudo install` one-liner.
- `deploy/bootstrap_host.sh SITE_ENV` — idempotent, unprivileged:
  creates `${GEECS_CHECKOUT_ROOT}/<service>-checkout` clones (or
  fetches them), `poetry install` with each service's extras (from a
  table in the script, mirroring the runbooks), renders `config.ini`
  from `site.env` when absent or empty — and when one exists **prints
  the diff** between it and the rendered form, so a stale client config
  on an existing host is seen and reconciled, never silently kept (it is
  never overwritten: the file also holds the hand-entered Tiled key),
  bakes the MCP venv, renders units to staging, then prints the sudo
  steps in order. It is the fleet map's seven-step bootstrap list made
  executable; the list in the fleet map becomes a pointer to it.
- Each package runbook: replace its hand-copied unit prose with "render
  from the template with your `site.env`; see the site profile page".
  Package patch bumps + CHANGELOGs (docs-only changes still bump, per
  #536 precedent).
- Live check (non-invasive, **done 2026-09-03**): render the units on the
  current host from a `site.env` carrying the host's *real* account,
  home, checkout root and poetry path (so path lines compare like for
  like), into a temp directory, and `diff` the rendered
  `geecs-ca-gateway` / `geecs-data-portal` units against the installed
  ones. Expected differences: the `EnvironmentFile=` line and the values
  it replaced, older `Description` text, and the gateway's checkout path
  (`gateway-checkout` vs the current `~/GEECS-Plugins`, which Phase 3
  migrates). Anything else is a finding — this diff is what surfaced the
  portal's hand-typed `--processing-configs` path and turned it into the
  `GEECS_CONFIGS_ROOT` site value. Run `bootstrap_host.sh --dry-run` for
  the rest; delete the temp directory afterwards.

### Phase 3 — promote the current box (the invasive one)

Order chosen so each step is reversible by restarting what was stopped:

1. **Sam step:** `sudo install -D deploy/site.env /etc/geecs/site.env`
   (rendered from HTU values), then install the five rendered units and
   `systemctl daemon-reload`.
2. **Bring the gateway clone's disk to its HEAD.** The index and working
   tree are exactly commit 9c9681de (verified by tree hash: nothing on
   disk is hand-edited, and the 538 "added" files are the wavekit SDK
   docs that commit still tracked). `git reset --hard HEAD` discards
   nothing that is not in git history. Because the editable install
   runs whatever is on disk, this **does** change the running code
   (GEECS-Core db module, GEECS-Schemas) and needs a gateway restart —
   fold it into step 5. Until then the gateway keeps running the
   August 25 tree, which it has done since 2026-09-02 without incident.
3. **Queueserver family**, one maintenance window, no scans running:
   stop the hand-started RE Manager, the doc proxy, and the user-scope
   capture unit (`systemctl --user disable --now geecs-capture`); start
   `geecs-qserver` then `geecs-capture` as system units; `qserver status`
   + the capture heartbeat file are the health checks. Redis: the launch
   script starts it if absent, which is why `redis-server.service` shows
   inactive today — leave that as is, note it in the runbook.
4. **MCP:** rebake the venv from `~/qs-checkout` (it is one pull behind
   its own source), stop the hand-started process, start `geecs-mcp`.
   OSPREY's profile URL does not change.
5. **Portal and CA gateway:** swap their units for the rendered ones and
   restart — the gateway restart drops every CA client for ~10 s, so it
   goes last and only with Sam's go.
6. `/fleet-status` shows eight systemd rows, no UNMANAGED, no user unit,
   no staged files. **The fleet map's service table is rewritten in the
   same PR** to the observed truth (host, checkout path, unit, venv).

### Phase 4 — the remaining literals

- PVA roster per experiment: `gen_fleet_status.py --experiment X` derives
  `HOSTS` from the DB (the docstring already describes the query) and
  writes `fleet_status_<experiment>.bob`; `scripts/fleet_status.sh`
  reads the `.bob` for its experiment. HTU's file is regenerated and the
  four unreachable hosts are pruned or confirmed with Sam.
- Timezone: the `America/Los_Angeles` defaults in `geecs_bluesky`
  (`assets/tiled_readback.py`; the `analysis/` copies went with #786)
  default to the host zone, resolved per queried date through a naive
  local midnight's `astimezone()` — not `datetime.now().astimezone().tzinfo`,
  whose fixed offset is wrong across a DST change; the portal unit
  inherits `TZ` from `site.env` via `EnvironmentFile=` (no template hole). Minor bump.
- `geecs_paths_config.py` default experiment `"Undulator"` → required,
  or read from `config.ini [Experiment]` (it already reads the file).
  `EXPERIMENT_FILE_IDS` (Google Doc IDs in data-utils code) moves to
  `GEECS-Plugins-Configs` or `config.ini`; flagged for Sam, since it
  touches the LogMaker path we are not refactoring.
- Docstring examples stay as they are; they are examples.

### Phase 5 — the new box (when it arrives)

`site.env` for the new host + `bootstrap_host.sh` + the Sam sudo steps
= the migration. The fleet map gets a "migration" section pointing at
the bootstrap; the old box's rows are removed the day the services
move. Client `config.ini` files change one value (the host) if the
address changes. The dashboard pane tells you when every row is green.

### Deferred, deliberately

- **Containers.** Only the HTTP-shaped services (portal, MCP, Tiled)
  are candidates, and only if Phase 5 proves painful; `site.env` is
  exactly what a compose file would consume, so nothing here is wasted.
  The CA gateway, PVA gateways, and queueserver stay on bare systemd with
  host networking (EPICS UDP, GEECS wire protocol, SMB mounts).
- **Passwordless sudo for deploys.** A `sudoers.d` rule letting the
  service account run `systemctl {daemon-reload,restart,start,stop} geecs-*`
  would remove the Sam step from routine restarts. Worth it on the new
  box; a Sam decision, not a phase.
- **Second-facility validation.** Portability is by construction here.
  The first real second site will find the values we missed; that is
  when this plan gets a Phase 6.

## Sizing

| Phase | Shape | Rough size |
|---|---|---|
| 0 | existing PR | — |
| 1 | docs + example + skill text | one PR, ~300 lines, one session |
| 2 | 5 templates + 2 scripts + 5 runbook edits | one PR, ~500 lines, one session + the scratch live check |
| 3 | host work + fleet map rewrite | one maintenance window with Sam, one small PR |
| 4 | 3 small code PRs | one session |
| 5 | host work | the migration day |

## Open decisions for Sam

1. `/etc/geecs/site.env` (system, needs sudo once) vs
   `~/.config/geecs/site.env` (unprivileged, but user-scope units only).
   Plan assumes `/etc/geecs` + system units.
2. Checkout root on the new box: `/home/<svc>` (no sudo) or `/opt/geecs`
   (cleaner, one sudo chown). Either works; `site.env` carries it.
3. The passwordless-sudo rule above.
4. ~~The 774 staged files~~ Resolved by inspection: they are the August 25
   tree; `git reset --hard HEAD` + restart in Phase 3 step 5. No decision
   needed beyond the restart window.
