# /triage — GEECS scan log triage

Stage 1: generate a structured triage report from scan logs.
Stage 2: analyze bug candidates against the codebase and draft GitHub issues.

## Arguments

`$ARGUMENTS` is passed verbatim to `geecs-log-triage`. Valid forms:

```
--date 2026-05-08 --experiment HTU
--date 2026-05-08 --experiment HTU --scan 42
--date-range 2026-05-01:2026-05-08 --experiment HTU
--scan-folder /path/to/Scan037
```

If `$ARGUMENTS` is empty, print usage and stop.

---

## Step 1 — Environment check

Before running anything, verify the poetry env is ready:

```bash
cd GEECS-LogTriage && poetry run python -c "import geecs_log_triage" 2>&1
```

If this fails with `ModuleNotFoundError`, run `poetry install` in `GEECS-LogTriage/`
first and explain to the user what you did.

---

## Step 2 — Generate the triage report

Run two commands from `GEECS-LogTriage/`:

**JSON** (for your analysis — always goes to stdout):
```bash
cd GEECS-LogTriage && poetry run geecs-log-triage $ARGUMENTS --format json
```
Capture the stdout JSON into memory. If the command fails, show the error and stop.

**Markdown** (for operators — auto-written to the data folder):
```bash
cd GEECS-LogTriage && poetry run geecs-log-triage $ARGUMENTS
```
With `--date` + `--experiment`, this silently writes `triage.md` alongside the
scan data. With `--scan-folder`, it prints to stdout; write it next to the scan
folder yourself. No need to capture this output.

---

## Step 3 — Summarise the report

Print a one-paragraph summary: how many scans examined, total errors, count per
classification. If there are zero `bug_candidate` entries, say so and stop — no
issues to file.

---

## Step 4 — Analyze each bug candidate

For each unique fingerprint with `classification = "bug_candidate"`:

1. **Locate the code.** The `signature` field contains
   `(exception_type, file, function)`. Use `grep` or `find` to locate that file
   in the monorepo. Read the relevant function and surrounding context (±30 lines).

2. **Understand the error.** Read `sample_traceback` and `normalized_message`.
   Cross-reference with the source to understand *why* this path fails, not just
   *that* it fails.

3. **Draft a GitHub issue.** Use this template:

   ```
   Title: [Package] ExceptionType in module.function — short symptom

   ## Summary
   <1-2 sentences: what fails and what the user experiences>

   ## Occurrences
   - Scans: <scan IDs>
   - Fingerprint: `<hash>`
   - First seen: <timestamp from earliest occurrence>

   ## Traceback
   ```
   <sample_traceback, truncated to 30 lines>
   ```

   ## Root cause hypothesis
   <Your analysis of why this happens based on the source code>

   ## Suggested fix
   <Concrete suggestion: guard clause, exception catch, type check, etc.
    Include the file path and approximate line number.>

   ## Labels
   bug, <package-name>
   ```

4. Show the draft to the user before filing anything.

---

## Step 5 — Confirm and file

After presenting all drafts, ask: **"File issues for which of these? (all / none /
list numbers)"**

For each confirmed issue, run:

```bash
gh issue create \
  --title "<title>" \
  --body "<body>" \
  --label "bug"
```

Print each created issue URL. If `gh` is not authenticated or the command fails,
show the draft body so the user can file manually.

---

## Notes

- Only file for `bug_candidate`. `hardware_issue`, `config_issue`, and
  `operator_error` are noted in the summary but do not produce issues unless the
  user explicitly asks.
- Deduplicate: if the same fingerprint hash already appears in an open GitHub
  issue (check with `gh issue list --search "<fingerprint_hash>"`), skip it and
  note the existing issue URL instead.
- The monorepo root is the parent of `GEECS-LogTriage/`. All file paths in
  tracebacks are relative to the scan machine's working directory — map them to
  the monorepo layout by matching the filename and function name, not the full
  absolute path.
