---
name: package-tidy
description: >
  Make one GEECS-Plugins package look organized and professional WITHOUT
  changing behaviour: layout moves, cross-package duplicate removal and a
  docstring trim, each in its own reviewed PR, after a repo-wide
  duplication map. Use for "clean up <package>", "tidy <package>",
  "reorganize the layout", "this package looks vibe coded", "too many
  top-level modules", "the docstrings are essays", or when a package's
  doc_audit advisory counts keep growing. Not for splitting functions or
  modules, or deleting features — that is its own arc with hardware
  gates. PR mechanics are /land; env repair is /env-doctor.
---

# /package-tidy <Package> — tidy one package, behaviour unchanged

Pilot: GeecsBluesky #1048/#1051/#1052. PR ritual: `/land`. Envs:
`/env-doctor`. Prose rules and the gate: `scripts/doc_audit.py`. Branch
topology: `CONTRIBUTING.md`. `$ARGUMENTS` — the package directory name;
`<import_name>` below is its import name (`geecs_bluesky`).

## Scope

In: moving or renaming modules, removing a duplicate of a helper that
has a home elsewhere, trimming docstrings and comments, fixing
references. Out: splitting large functions or modules (a 200-line plan
generator, a 2,000-line FastAPI `app.py`). A split changes control flow,
so "no test assertion changed" is no longer a sufficient gate; that work
is its own arc with hardware verification of the scan paths. List such
modules in the survey as candidates for that arc and move on. Also out:
dead-code hunting by tool — a periodic repo-wide vulture sweep (every
package's source dir plus every `tests/` dir as one input,
`--min-confidence 80`,
`--ignore-names cls,model_config,pytestmark,pytest_*,connection_made,connection_lost,datagram_received,error_received,__getattr__`,
`--ignore-decorators '@model_validator,@field_validator,@*.validator,@*.route,@*.get,@*.post,@*.put,@*.delete,@*.websocket,@*.fixture'`)
is a cleanup-day activity outside this skill, and its findings are
always questions for Sam, never verdicts.

## 0. Inputs — ask Sam first (one question round)

- The goal in his words, and what struck him (layout? prose? duplication?).
- Constraints. Default: no logic change; docs track every move; no new
  comments or docs.
- Anything he plans to use that looks unused today. Never infer "dead"
  from zero callers (actions are config-driven and written on the fly).

## 1. The cross-package map (repo-wide, before any one-package proposal)

A one-package audit cannot see a duplicate: both pilot fixes
(`scanner_configs` → `geecs_core.configs_repo`; the `config.ini` reader)
were visible only repo-wide. Findings live in ONE tracking issue with
a checklist (#1061), titled with the `Cross-package:` prefix; add new
findings to it as checklist items, and tick the item a tidy PR resolves.
Read it first:

    gh issue list --state open --search 'in:title "Cross-package:"'

If the list is empty, this run produces the map (main session, read-only):

1. Importer map, repo-wide, per top-level module of every package —
   code, tests, docs, skills, scripts, units, CI:
   `git grep -nP "<import_name>[./]<mod>(?![a-z_])|from \.+<mod>\b|from (<import_name>|\.+) import .*\b<mod>\b"`
   (the dotted form, `from .mod`, and `from pkg import mod`). Prove the
   grep hits one importer you already know before trusting it.
2. Duplication sweep by category: `config.ini` readers; DB query unions
   and the gateway served set; path translation; exception trees;
   schema knowledge; FastAPI glue (forwarded-prefix middleware, templates
   factory, theme mount). grep the other packages for the same helper;
   read docstrings that announce a planned move.
3. Show Sam the map; add each finding he keeps as a checklist item in
   the tracking issue (or open one if none exists), each naming
   the places (`file:line`), the candidate home, and whether failure
   semantics differ between the copies (diff them before merging them).

Known open examples at the time of writing — verify, not a fixed list:
the CA served set (`get='yes'` ∪ settables of enabled devices) computed
in GeecsCAGateway `config.py`, GeecsArchiver `archive_set.py` and
GeecsBluesky `db_runtime.py` (candidate `geecs_core.db.served_set`;
GeecsCAGateway `audit.py` shares the two DB queries, not the union rule;
Bluesky reads a DB failure as "unknown", the others do not); further
`config.ini` readers in GEECS-Data-Utils
`config_roots.py` and `scripts/qserver_probe.py` (the latter overlaps
`qs_client.read_qserver_config`).

## 2. Survey the package (main session, read-only, no agents)

1. Freshness: `git fetch && git log -1 origin/master`; verify every fact
   from a handoff or memory file before acting on it.
2. Inventory: top-level modules and subpackages with LOC
   (`git ls-files <Package>/<import_name> | xargs wc -l`).
3. Importers per module, in and out of the package, from the map above.
4. Every `__init__.py` (eager vs lazy) and each module's import block:
   this decides where a "light" module may live. Light-import contracts
   are pinned by tests (`GeecsBluesky/tests/test_lazy_package_import.py`
   is one); find the package's.
5. Deploy footprint: console-script targets, unit files, `python -m`
   paths, logger names (`getLogger(__name__)`), GEECS-LogTriage
   fingerprints. A package that runs inside the gateways and on the
   Windows camera servers (GEECS-Core) is redeployed fleet-wide by any
   rename — prefer prose and duplicate removal there.
6. Baseline: `python3 scripts/doc_audit.py --advisory -p <Package>`;
   `--strict` must already exit 0. Do not invent your own prose metrics;
   they overstate the problem.

## 3. Proposal → STOP for Sam

One table, one row per move: old → new path, LOC, importers in/out of
the package, value (high/med/low), deploy impact. A second list,
"considered, recommend NOT", with one-line reasons: junk-drawer folders
(`runtime/`, `contracts/`), moves that make a light import heavy (a
module under a package whose `__init__` loads the engine), deleting a
config-driven feature with zero callers. Sam picks; nothing moves before.

## 4. Execute

- One worktree `.claude/worktrees/<package>-tidy`, branch off
  `origin/master`. One mover (an agent, or you): one commit per move;
  delete the old path — no shims or re-exports; update every reference
  repo-wide; rewrite paths in existing docs only.
- Testing from the worktree: install its envs per `/env-doctor` (a
  worktree never shares the main checkout's; `poetry -C <main checkout>
  run` imports the MAIN checkout, whose `__pycache__`-only namespace
  dirs can hide CI failures). Per move, `./scripts/check.sh <Package>`;
  the fast shortcut is the main env's python with the worktree packages
  first on `PYTHONPATH` — print `<import_name>.__file__` first and
  confirm it is inside the worktree.
- Gates per move: targeted tests pass; `scripts/doc_audit.py --strict
  --only dangling-ref,dangling-path` adds nothing; `git grep` for the old
  dotted name AND the old file path is empty outside `CHANGELOG.md`, and
  every hit on the bare `\b<mod>\b` is triaged (`from pkg import mod`
  carries neither).
- No version bump (per `/land`, bumps happen at deploy/tag). The PR body
  records old → new paths, every visible side effect (logger names,
  fingerprints) and the host deploy step; the release that ships it is a
  minor bump for the package (its import paths changed) and lifts those
  into the CHANGELOG entry.
- Full `./scripts/check.sh --all` once; `doc_audit.py --strict` exits 0.

## 5. Gates before the PR

- NO test assertion changed: test edits are import paths only. An
  assertion change is a behaviour change — stop and explain.
- Light-import contracts still pinned; add an assertion if a light
  module now sits behind a package `__init__`, and prove it bites by
  breaking the code first.
- Optional guard: a test pinning the set of top-level modules, so a new
  one is a deliberate allowlist edit.
- Prose trim is a separate PR after the layout PR merges: the AST of
  every `.py` is identical with docstrings stripped; load-bearing rules
  are shortened, never deleted; history is deleted, not copied (the
  CHANGELOG has it); at most one bare `#NNN` per docstring.

## 6. Review and land

Follow `/land`. Spawn the reviewer with the Agent tool, not inside a
Workflow, so the same reviewer confirms each fix commit. Add this fourth
lens to the `/land` brief, verbatim:

> 4. **Dead code, by reading the diff** — symbols whose last caller this
>    PR removed, re-exports nothing imports any more, shims or old paths
>    left behind, tests that now pin nothing. Report each as a finding
>    phrased as a question for the owner ("unused today — planned?"),
>    never as a verdict to delete: features with zero callers today are
>    planned (actions). Do not run vulture on the diff: per package it
>    is ~2% signal (a sibling package's use of the public API and
>    framework callbacks read as "unused").

Post findings and dispositions as a PR comment. Master merges are
Sam's; stacked PRs follow `/land` step 9.

## 7. Deploy

`/lab-status` first; the ritual is `docs/platform/fleet_map.md`'s. The
tidy-specific step: `poetry install` every service env that contains
the package BEFORE the restart (see Traps). Then `/fleet-status`, and a
hardware check if scan paths or devices were touched.

## Agent shape

The main session surveys and proposes (no agents) and gets Sam's
decisions. Then one mover (one worktree, one commit per move, targeted
tests per move, the full check once); one prose trimmer after the moves
merge, two above ~15k lines, split by disjoint file groups; one
reviewer per PR, spawned with the Agent tool. Reason: the pilot spent
~1.8M tokens on 18 agents, five of them 1–3-file renames that each ran
the whole 859-test suite in a worktree of their own.

## Traps (all hit in the pilot)

- `git grep -E` ignores `\b` → silent zero hits. Use `-P`.
- zsh does not word-split `set -- $var` → run loops with `bash -c`.
- Handoffs and memory carry stale facts (the pilot's called two
  dependent modules stdlib-only) — verify each before acting on it.
- Moving a module renames its logger → GEECS-LogTriage fingerprints
  shift; note it in the CHANGELOG.
- Moving a console-script target breaks the installed wrapper until
  reinstall.
- Merging two docstrings pushes a module docstring over doc_audit's
  25-line long-doc limit.
- Cherry-picked or rebased branches never show as merged: before
  deleting one, compare its changed lines against master's commit.
