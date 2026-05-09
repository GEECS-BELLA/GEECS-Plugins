# Writing a Skill

This page is for contributors who want to add a new skill to the repository. Read the [Skills Overview](overview.md) first if you haven't — it covers what a skill is and why the pattern exists.

## The two-layer pattern

Every skill has exactly two parts:

**A Python CLI tool** that does the deterministic work. It accepts well-typed arguments, produces structured output (JSON to stdout, markdown to disk), and has no dependency on any agent or LLM. It can be run by a human, a script, or a CI job. It can be tested with pytest against fixtures. It has a version and a changelog.

**A markdown slash command** in `.claude/commands/<name>.md` that wraps invocation of the CLI and describes to the agent how to interpret and act on the output. The markdown file is the UX layer for agentic use; it documents the argument forms, the expected output shape, and what the agent should do at each stage.

The separation matters for the same reason the scanner engine's typed event stream matters: the CLI is the stable contract. The agent is a consumer of that contract, not part of its implementation. When you upgrade the LLM, the CLI doesn't change. When you want to run the CLI in a notebook, you don't need the agent. When you want to test the skill end-to-end, you test the CLI against real or synthetic inputs and separately test that the command file's instructions produce sensible agent behavior on known CLI output.

## How /triage is built

`/triage` is the reference implementation. Walking through it concretely is more useful than abstract guidelines.

**The CLI** lives in `GEECS-LogTriage/`. Its entry point is `geecs_log_triage.cli:main`, registered in `pyproject.toml` as `geecs-log-triage`. The CLI walks one or more scan folders, parses each `scan.log` file, normalizes errors into stable fingerprints via `ErrorFingerprint` (exception type + file + function, hashed to a short string), classifies each fingerprint as `bug_candidate`, `hardware_issue`, `config_issue`, or `operator_error`, and assembles a `TriageReport` Pydantic model.

**Output shape.** The CLI has two output modes controlled by `--format`:

- Default (no flag): writes a human-readable markdown summary as `triage.md` alongside the scan data. Good for a lab operator reading the file directly.
- `--format json`: serializes the `TriageReport` to stdout as JSON. Good for the agent — structured, typed, and parseable without screen-scraping.

The agent uses both. It runs the CLI twice: once with `--format json` to get the structured data it reasons over, and once without to produce the human-readable file that persists on disk. This is the standard contract: **JSON to stdout for the agent, markdown to disk for the human**.

**The slash command** lives at `.claude/commands/triage.md`. It defines five stages:

1. Environment check — verify the poetry env is ready before running anything.
2. Generate the report — run both CLI invocations described above.
3. Summarise — print a brief count of errors and classifications.
4. Analyze bug candidates — for each `bug_candidate` fingerprint, locate the relevant source file in the monorepo, read the failing function, form a root-cause hypothesis, and draft a GitHub issue.
5. Confirm and file — show all drafts, ask which to file, run `gh issue create` for each confirmed one.

The markdown file is instruction, not code. It describes what the agent should do with the CLI output. It does not contain any Python. It does not hard-code paths that could drift.

## The contract worth standardizing

For any new skill, follow this convention:

| Layer | What it does | Output |
|---|---|---|
| CLI | Deterministic work: parse, query, classify, compute | JSON to stdout (`--format json`), markdown file to disk (default) |
| Slash command | Agentic UX: invoke CLI, interpret output, reason, act | Agent conversation; optional side effects (issues filed, files written) |

Keeping these two forms of output separate — machine-readable JSON for the agent, human-readable markdown for the operator — means both audiences get what they need without either compromising the other. A triage report that's easy to read in a terminal is not necessarily easy to parse reliably; a report that's easy to parse is not necessarily readable. Emit both.

## Adding a new skill

**Step 1: build the CLI tool.** Create a new package in the monorepo following the existing pattern (`GEECS-LogTriage/` is the template). The CLI should:

- Accept well-typed arguments via `argparse` or `click`.
- Return exit code 0 on success, non-zero on failure.
- Write `--format json` output to stdout with a stable schema (a Pydantic model is the right choice).
- Write the human-readable default output to a predictable location on disk (next to the data it analyzed).
- Have tests that run against synthetic fixtures without network access.

**Step 2: write the slash command.** Create `.claude/commands/<name>.md`. The file should cover:

- What arguments the underlying CLI accepts, with examples for each common form.
- How to check that the environment is ready before running.
- What the agent should do with each part of the CLI output.
- What the agent should show the user and when to ask for confirmation before taking irreversible actions (filing issues, moving files, etc.).

Keep the instruction file honest about what the agent cannot do reliably — hallucinating bug fixes without reading the source code, for example. Name those boundaries explicitly so the skill degrades gracefully when it hits them.

**Step 3: document it** in [Skills — Overview](overview.md). Add a section following the `/triage` pattern: what problem it solves, when to use it, a worked invocation, and the CLI surface for users who want to skip the agent layer.

## Candidate next skills

These are worth building when you've heard the same question twice from different people — not speculatively:

**`/scan-info`** — given a scan number (and optionally an experiment and date), summarize what the scan was: scan type, devices, variable range, shot count, any errors in the log. The CLI would wrap `geecs_data_utils.ScanPaths` and the log parser from `GEECS-LogTriage`. The agent would add context: "this device had a timeout on step 3, and the next scan that day worked fine."

**`/find-scans`** — query the scan database by date range, experiment, scan variable, or device name. Returns matching scan numbers. The CLI is a thin wrapper around `geecs_data_utils`; the agent surfaces the results in a way that's useful for deciding which scan to load into a notebook.

**`/diff-scans`** — compare two scans by scan number. Difference in scalar summary statistics, device lists, log error counts. Useful for "scan 42 worked, scan 43 didn't — what changed?" The output is a structured diff; the agent interprets what's significant.

**`/issue-from-log`** — given a scan log path, draft a GitHub issue with the full context pre-filled: scan tag, exception, traceback, save element YAML, config version. Overlaps with `/triage` for the single-scan case, but without the classification step — useful when you already know something is a bug and just want the issue written.

**`/save-element-check`** — validate a save element YAML against the live experiment database. The CLI queries the database, checks that each device and variable exists, and reports missing entries. The agent explains what each missing entry probably means and suggests corrections.
