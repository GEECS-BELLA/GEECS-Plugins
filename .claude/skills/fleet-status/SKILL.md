---
name: fleet-status
description: >
  Observed picture of the deployed fleet: which host runs which service,
  from which checkout, on which branch/commit and package version, and
  whether the running process matches the code on disk. Use when the user
  asks "what is deployed where", "which version is the portal / gateway /
  worker running", "is the host still on the feature branch", "did the
  deploy take", "does the fleet map match reality", before deploying or
  restarting a service, or after a PR merges that a host is expected to
  pick up. Wraps scripts/fleet_status.sh, which gates on /lab-status
  first. For plain reachability ("am I on the lab network") use
  /lab-status; for the intended picture read docs/platform/fleet_map.md.
---

# /fleet-status — what code is the fleet actually running?

`docs/platform/fleet_map.md` is the **intended** fleet: hosts, ports,
units, runbooks. `scripts/fleet_status.sh` is the **observed** one. The
two drift during rapid development — a host runs a feature-branch
checkout for a live check, a checkout gets pulled but the unit never
restarted, a version bump lands without a `poetry install` — and this
skill exists to see that drift instead of remembering it.

The script owns the probes, timeouts, and how endpoints are derived
(config.ini `[tiled]`/`[qserver]`, ssh aliases from `~/.ssh/config`).
Do not restate or hardcode hosts here. Everything it does is read-only
and bounded: no PV writes, no restarts, no pulls.

## Invocation

| Situation | Command |
|---|---|
| The normal case (full log, for reasoning) | `scripts/fleet_status.sh` |
| Just the PVA image fleet | `poetry -C GeecsPvaGateway run geecs-pva-gateway fleet --experiment NAME` |
| One box table + attention list (for the user) | `scripts/fleet_status.sh --summary` |
| Persistent dashboard pane (cmux/tmux) | `scripts/fleet_status.sh --watch 300` (summary every 300 s; add `--full` for the log) |
| Off the lab network / just want the local worktree picture | `scripts/fleet_status.sh --local-only` |
| No key-based ssh to the hosts from this machine | `scripts/fleet_status.sh --no-ssh` |
| ssh alias not discoverable from `~/.ssh/config` | `scripts/fleet_status.sh --ssh <ip>=<alias>` |
| config.ini has no `[Experiment]` | add `--experiment <Name>` (needed for the CA gateway PVs) |
| Avoid the GitHub round-trip | `--no-fetch` (distances then use the last-fetched `origin/master`) |

The summary table is rendered by `scripts/fleet_table.py` (stdlib only)
from the same probe records; the full log is what you reason over, the
table is what you show. `$ARGUMENTS`, if given, passes straight through. Budget: ~10 s offline,
under a minute on the lab network (each ssh host 25 s max, each PVA
host 2 s per get).

## Stages and what each one proves

**Stage 0 — reachability.** Runs `scripts/lab_status.sh` tier 1 (the
same gate `/lab-status` uses). If the network is DOWN the script stops
after the local picture and says `remote: UNKNOWN`. Report exactly that:
an unreachable service is *unknown*, not down. Never bypass this by
calling the services directly.

**Stage 1 — self-reported versions.** Each service over its own
protocol: Tiled's library version (HTTP), the Data Portal's `/health`
(`ok`, catalog probe, package version), the queueserver's **readiness**
(0MQ `status` + `plans_allowed`, read-only: manager state, worker
environment state, allowed-plan count, queue depth — a manager that
answers with its environment closed knows zero plans and refuses every
submission as "not in the list of allowed plans"; the script prints
`NOT READY` for that, never `OK`, #793), MCP port liveness (no version
endpoint; stage 2 reads its venv), the CA gateway's heartbeat /
`devices_connected` / `version` PVs (from `lab_status.sh --hardware`,
read-only — its `role=CA gateway` record line is the contract, not the
prose), and every PVA image gateway's `version` + `heartbeat` PVs via
`geecs-pva-gateway fleet` (the package owns the roster, the PV names,
and the probe; the script only relays its lines and its record). The
roster is read live from the GEECS DB (endpoint IPs hosting the
experiment's image devices, `geecs_pva_gateway.fleet`); a roster host
absent from `config.ini [pva] addr_list` is **not deployed** — printed
as a `[ -- ]` row, never `[DOWN]` — because no instance was installed
there. MCP not listening is a `[WARN]` + ✗ row: the fleet map has it as
a system unit on the worker host, so silence is a finding, not
"pending deploy". A `[WARN] PVA fleet runs mixed
versions` line means a rollout is incomplete or a box's
pull-on-restart no-oped — the GeecsPvaGateway runbook's known failure.

**Stage 2 — host checkouts (ssh).** On each host derived from
config.ini, services are discovered two ways and deduplicated by pid:
every `geecs-*` (+ `tiled`) systemd unit in **both** system and user
scope, and **whoever owns each fleet port** — a queueserver started by
hand in tmux has no unit and must still show up. Each process is mapped
to the clone it runs from (its cwd, or the baked venv's recorded source
path for the MCP pattern), then: branch, short sha, commit date,
staged/unstaged counts, pyproject version vs the version installed in
the running interpreter's venv, and whether HEAD moved after the process
started (for a baked venv: after the install). The warnings to act on:

| Warning | Meaning | Remedy (on the host, by the user or with their go) |
|---|---|---|
| `no systemd unit owns this process` (UNMANAGED) | started by hand; dies with its tmux/ssh session or a crash, no restart on reboot | install the runbook's unit when the dev phase ends; until then, know it |
| `(user unit)` in the name | runs under `systemctl --user`, not the system scope the runbook describes | fine for development; note that `systemctl status` without `--user` will not see it |
| `pyproject says X but the venv has Y installed` | a bump landed in the checkout without reinstall | `poetry install` in that package dir (pip reinstall for a baked venv), then restart |
| `checkout moved <time> after the process started` | the running process predates the code on disk | restart it — after confirming that checkout is the intended one |
| `checkout moved <time> after the baked install` | the MCP-style venv was baked from an older checkout state | pip reinstall from the checkout, then restart |
| `STAGED: N` / `UNSTAGED: N` | the deployed tree differs from its HEAD | look before touching: it may be a live hotfix nobody committed, or a stray `git add` |
| `Queueserver … NOT READY — worker environment CLOSED` | the unit is up but nothing opened the RE worker environment (every `geecs-qserver` restart recreates this until #793's readiness step lands); the console gets "Plan … is not in the list of allowed plans" | `qserver environment open` from any client env (GeecsBluesky's), then re-run; the plan name in the console's error is a red herring |
| `Queueserver … readiness UNKNOWN (port only)` | no local env with `bluesky-queueserver` to ask the manager | `/env-doctor` for GeecsBluesky; the port is listening but that proves nothing about plans |
| `Queueserver … readiness UNKNOWN — plans_allowed unanswered` | the manager answered `status` but not the plan list (a wedged or overloaded manager, or a refused user group) — readiness is an assertion about plans, so none is made | re-run; if it repeats, read the manager's journal on the host before touching the environment |

Any unit whose clone is on a branch other than `master` is a fact to
surface, not a fault — feature-branch deploys for live checks are
normal here. Say which branch, and whether the merge has since landed
(stage 3 tells you).

**Stage 3 — local cross-reference.** Fetches origin, lists this clone's
worktrees, then names each deployed sha: the remote branch it is the tip
of (or the branches containing it), its distance from `origin/master`,
and the local worktree holding it. `commit unknown to this repo` means
the host runs something never pushed, or a branch this clone never
fetched — say so; do not guess which.

## Reporting

Lead with the drift, not the table. The user wants: which services are
on a non-master checkout, which have a pending restart or install, which
PVA boxes lag, and what is unknown because of reachability or missing
ssh. Then the one-line-per-service summary (host, unit state, branch @
sha, package version). Quote versions and shas exactly as printed.

When the observed picture contradicts `docs/platform/fleet_map.md`
(a service on a different host, a checkout path the table doesn't
name, a unit the map lists that is not listening),
say so explicitly and offer to update the fleet map in the same PR as
whatever change the user makes next — the map's own rule is "update
this page in the same PR".

## Boundaries

- The script never restarts, pulls, or installs. Those are the user's
  actions on the host (sudo, service account); hand them the exact
  command and unit name, referencing the service's runbook.
- Stage 2 needs key-based ssh as the service account. A failed ssh is
  reported per host with the `--ssh` override hint; it is not a fleet
  finding.
- "Gateway version X" says what package is running, not whether the
  device roster is healthy — that is `/lab-status --hardware` and the
  gateway DB audit.
- Tiled has no checkout (pip install); its version is the whole story.
- Port-owner discovery needs `ss -p` to see the pid, which it does only
  for processes owned by the ssh user (or root). A fleet port that shows
  as listening in stage 1 but has no stage-2 record was started by a
  different account.
