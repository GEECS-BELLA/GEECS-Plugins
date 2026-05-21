# Skills

A **skill** in this repository is a markdown command file under `.claude/commands/` that wraps a CLI tool and is invoked as a slash command from a [Claude Code](https://claude.ai/code) session. The pattern works because of a clean layering: a deterministic Python CLI does the structured, reproducible work — parsing logs, querying databases, walking scan folders — and the agent handles the parts that benefit from language understanding: summarizing findings, locating relevant source code, drafting GitHub issues, asking clarifying questions. Each layer is independently testable and independently useful. The CLI can be run by a script or a human without an agent; the agent can be swapped out without changing the CLI. The same separation that makes the scan engine's typed event stream robust applies here: a well-defined contract between a deterministic core and a flexible consumer.

## Shipped skills

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

The skill file ships with the repository at `.claude/commands/triage.md`. If Claude Code is configured to read project-level commands (the default when you open the monorepo root), `/triage` is available immediately after cloning. No extra installation step is needed for the slash command itself.

The underlying CLI requires a one-time setup:

```bash
cd GEECS-LogTriage
poetry install
```

Run this once from the monorepo root. After that, the agent and the CLI both work from the same poetry environment.
