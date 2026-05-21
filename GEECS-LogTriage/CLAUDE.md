# GEECS-LogTriage — Developer Context for Claude

Stage 1 of an auto-debugger pipeline. This package is the deterministic,
testable floor: it parses scan execution logs and produces structured error
reports. It does **not** call any LLM, file any GitHub issues, or modify any
other code. Stage 2 (LLM triage) and Stage 3 (autonomous fix loop) consume
this package's output but live elsewhere.

## Package Layout

```
geecs_log_triage/
  __init__.py             # Public API
  schemas.py              # Pydantic models: LogEntry, ErrorFingerprint, etc.
  parser.py               # Regex parser for scan.log format
  fingerprint.py          # Normalize + hash error signatures
  classifier.py           # Map exception types -> Classification enum
  harvester.py            # Walk scans for a date, aggregate into TriageReport
  cli.py                  # `geecs-log-triage` console entry point
```

## Log Format Source of Truth

The parser targets the per-scan log format defined in
`GEECS-Scanner-GUI/geecs_scanner/logging_setup.py::attach_scan_log`:

```
%(asctime)s.%(msecs)03d %(levelname)s %(name)s [%(threadName)s] shot=%(shot_id)s - %(message)s
```

Where `asctime` uses `datefmt="%Y-%m-%d %H:%M:%S"`. If the format string changes
in `logging_setup.py`, update the regex in `parser.HEADER_RE` and bump a minor
version.

Multi-line tracebacks: when a log record contains an exception, the formatter
appends the traceback as additional lines that do not start with a header
matching `HEADER_RE`. The parser groups orphan lines into the previous
record's `traceback` field.

## Classification Conventions

- `bug_candidate` — should likely produce a bug report and a code fix:
  `KeyError`, `AttributeError`, `TypeError`, `ValueError`, `IndexError`,
  uncaught exceptions logged via `geecs_scanner._wrap_excepthook`,
  any unhandled exception originating in geecs_* modules.
- `config_issue` — user/operator misconfig, not a bug:
  `ActionError`, `ConflictingScanElements`, validation errors.
- `hardware_issue` — physical / network condition, not a code defect:
  `GeecsDeviceInstantiationError`, `ConnectionRefusedError`, `TimeoutError`,
  `OSError` with errno EINVAL/EBADF (the SafeFileHandler ignored class).
- `operator_error` — recoverable runtime conditions: missing data files,
  invalid scan parameters caught at scan start.
- `unknown` — fallback when the exception type or message can't be mapped.

When the GEECS exception taxonomy grows, edit `classifier.CLASSIFICATION_MAP`.
The map is the single source of truth.

## Fingerprint Algorithm

In order of preference (first that succeeds wins):

1. If a traceback is present: `(exception_type, top_user_frame_file,
   top_user_frame_function)` — strips stdlib frames.
2. Else: `(logger_name, level, normalized_message)` where `normalize()`:
   - replaces all digit runs with `N`
   - replaces absolute paths with their basename
   - replaces hex/uuid-like tokens with `X`
   - strips trailing punctuation and whitespace.

Hash is `sha1(signature_string).hexdigest()[:12]`.

The fingerprint stability guarantee: two errors with cosmetic-only differences
(scan numbers, timestamps, file paths) must hash to the same value. Tests in
`test_fingerprint.py` enforce this — do not relax those tests without bumping
the version, since downstream issue dedup depends on fingerprint stability.

## Output

The CLI emits a `TriageReport` JSON document. Schema is documented in
`schemas.py` docstrings. Stage 2 consumes this directly without further
transformation.

## Tests

- All synthetic — no live GEECS data dependencies in the unit tests.
- Run: `poetry run pytest`
- Markers: this package does not currently use the `integration`/`data`
  markers from the root `pyproject.toml`; add them if/when integration
  tests against real scan folders are introduced.

## Dependencies

- `geecs-data-utils` (path develop) — for `ScanPaths`, `ScanTag`, and
  `load_scan_log()`.
- `pydantic >=2.0` — all data models.

No optional dependencies. No PyQt. No matplotlib. This is a pure-Python utility
intended to run in CI, on the lab control machine, or in a notebook with
equal ease.
