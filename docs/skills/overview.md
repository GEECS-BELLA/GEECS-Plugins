# Skills

A **skill** in this repository is a markdown instruction file at
`.claude/skills/<name>/SKILL.md`, invoked as a slash command from a
[Claude Code](https://claude.ai/code) session. Each `SKILL.md` starts with
YAML frontmatter declaring the skill's name and a trigger description —
that description is what tells the agent when to reach for the skill even
if you don't type the slash command yourself.

Skills come in two shapes. The first wraps a deterministic tool: a Python
CLI or shell script does the structured, reproducible work — parsing logs,
running the test suites, probing the network — and the agent handles the
parts that benefit from language understanding: summarizing findings,
locating relevant source code, drafting GitHub issues, asking clarifying
questions. Each layer is independently testable and independently useful.
The CLI can be run by a script or a human without an agent; the agent can
be swapped out without changing the CLI. The same separation that makes the
scan engine's typed event stream robust applies here: a well-defined
contract between a deterministic core and a flexible consumer. The second
shape is instruction-only: a workflow ritual or diagnostic playbook (how to
land a PR, how to repair a Poetry environment) where the value is the
codified procedure itself rather than a wrapped tool.

## Shipped skills

| Skill | What it does |
|---|---|
| `/land` | Land a change as a PR the GEECS-Plugins way: branch off the right base, scope check, version bump + changelog, tests the way CI runs them, adversarial review, CI watch, merge |
| `/check` | Run repo lint + unit tests the way CI does, scoped to what changed; wraps `scripts/check.sh` |
| `/env-doctor` | Diagnose and fix a package's Poetry environment when poetry, pytest, or an import fails for setup-shaped reasons |
| `/lab-status` | Probe lab-network and hardware reachability with bounded timeouts before doing anything that needs them; wraps `scripts/lab_status.sh` |
| `/scan-audit` | Scan timing and cadence audit for the Bluesky path: "why was the scan slow", "did every shot land", per-shot cadence analysis of a scan folder |
| `/triage` | Generate a structured error report from scan logs, then analyze bug candidates against the codebase and draft GitHub issues |
| `/get-started` | Onboard a new developer in guide mode: environment check, orientation, a small first win, with the repo's guardrails applied on the user's behalf |

`/land`, `/check`, `/env-doctor`, and `/get-started` are development
workflow skills — they operate on the repository itself. `/lab-status`,
`/scan-audit`, and `/triage` are lab operations skills — they operate on
the experiment: the network, the hardware, and the data a scan left
behind. `/triage` is the reference implementation of the CLI-backed
pattern and is documented in depth below;
[Writing a skill](writing_a_skill.md) uses it as the template for new
skills.

### /triage — diagnose scan failures and draft bug reports

`/triage` walks one or more scan logs, groups errors into stable fingerprints, classifies each fingerprint as a bug candidate, hardware issue, config issue, or operator error, and then — for each bug candidate — locates the relevant source code, reasons about why the failure happens, and drafts a GitHub issue for your review before filing anything.

**When to use it:**

- A scan just aborted and you want to know whether the cause was hardware, misconfiguration, or a code bug.
- You want a weekly sweep across a date range to find recurring patterns before they accumulate into a backlog.

**Typical invocations:**

```
/triage --date 2026-05-08 --experiment HTU
```

```
/triage --date 2026-05-08 --experiment HTU --scan 42
```

```
/triage --date-range 2026-05-01:2026-05-08 --experiment HTU
```

```
/triage --scan-folder /path/to/Scan037
```

**What happens:**

The agent runs the underlying `geecs-log-triage` CLI twice: once as JSON (for its own analysis) and once as markdown (written as `triage.md` next to the scan data for human reference). It prints a one-paragraph summary — scans examined, total errors, count per classification — and then for each `bug_candidate` fingerprint it reads the relevant source code, writes a draft issue body with a root-cause hypothesis and a suggested fix, and shows all drafts before asking which ones to file. Fingerprints that already have an open GitHub issue are skipped automatically.

Hardware issues, config issues, and operator errors appear in the summary but do not produce issues unless you ask explicitly.

**Underlying CLI:**

If you want the triage report without the agent layer:

```bash
cd GEECS-LogTriage
poetry run geecs-log-triage --date 2026-05-08 --experiment HTU
# writes triage.md next to the scan data

poetry run geecs-log-triage --date 2026-05-08 --experiment HTU --format json
# emits TriageReport JSON to stdout

poetry run geecs-log-triage --scan-folder /path/to/Scan037
# single-scan mode; prints markdown to stdout
```

The `TriageReport` JSON can be piped into any downstream tool — a notebook, a dashboard, another script — without involving the agent at all.

## Installation

The skill files ship with the repository under `.claude/skills/`. If Claude Code is configured to read project-level skills (the default when you open the monorepo root), every skill above is available immediately after cloning. No extra installation step is needed for the slash commands themselves.

The CLI underlying `/triage` requires a one-time setup:

```bash
cd GEECS-LogTriage
poetry install
```

Run this once from the monorepo root. After that, the agent and the CLI both work from the same poetry environment.
