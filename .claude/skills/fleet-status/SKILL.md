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
| The normal case | `scripts/fleet_status.sh` |
| Off the lab network / just want the local worktree picture | `scripts/fleet_status.sh --local-only` |
| No key-based ssh to the hosts from this machine | `scripts/fleet_status.sh --no-ssh` |
| ssh alias not discoverable from `~/.ssh/config` | `scripts/fleet_status.sh --ssh <ip>=<alias>` |
| config.ini has no `[Experiment]` | add `--experiment <Name>` (needed for the CA gateway PVs) |
| Avoid the GitHub round-trip | `--no-fetch` (distances then use the last-fetched `origin/master`) |

`$ARGUMENTS`, if given, passes straight through. Budget: ~10 s offline,
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
(`ok`, catalog probe, package version), MCP port liveness (no version
endpoint; stage 2 reads its venv), the CA gateway's heartbeat /
`devices_connected` / `version` PVs (reused from `lab_status.sh
--hardware`, read-only), and every PVA image gateway's `version` +
`heartbeat` PVs (roster = the checked-in Phoebus fleet screen; a
DB-driven roster is still owed). A `[WARN] PVA fleet runs mixed
versions` line means a rollout is incomplete or a box's
pull-on-restart no-oped — the GeecsPvaGateway runbook's known failure.

**Stage 2 — host checkouts (ssh).** On each host derived from
config.ini, every `geecs-*` (+ `tiled`) systemd unit is mapped to the
clone it runs from (`WorkingDirectory`, or the baked venv's recorded
source path for the MCP pattern), then: branch, short sha, commit date,
dirty count, pyproject version vs the version installed in the unit's
venv, and whether HEAD moved after the unit started. The three warnings
to act on:

| Warning | Meaning | Remedy (on the host, by the user or with their go) |
|---|---|---|
| `pyproject says X but the venv has Y installed` | a bump landed in the checkout without reinstall | `poetry install` in that package dir, then restart |
| `checkout moved <time> after the service started` | the running process predates the code on disk | `systemctl restart <unit>` — after confirming that checkout is the intended one |
| `DIRTY: N modified tracked file(s)` | someone edited the deployed tree in place | look before touching: it may be a live hotfix nobody committed |

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
name, a service the table calls "pending deploy" that is listening),
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
