# GeecsScanLog — Developer Context for Claude

The scan logbook. Successor to `LogMaker4GoogleDocs`, which this package
will eventually replace outright.

## The one idea

The log is a **rendered view over (derived scan data) + (a small commentary
store)**. Those two halves have completely different properties and must not
share a store:

- **The record** — scan number, parameters, shot counts, status. Derived from
  `ScanInfoScanNNN.ini` and the folder listing on every request. The logbook
  stores *none* of it, so it can never drift from the data and an upstream
  change needs no migration here.
- **The commentary** — what people wrote. Irreplaceable, free-form, and the
  only thing that genuinely needs storage. Arrives in phase 02.

Conflating them is what made LogMaker4GoogleDocs unmaintainable: machine
content was injected into a human document by string-matching headings.

## A day is a query, not a document

`/log/day/2026-09-11` lists whatever `ScanNNN` folders exist at request time.
Nothing creates a log; no template is stamped; no daily job runs. A scan
appears because its folder does.

This is why the old `createGdoc.py` + bash polling loop has no successor —
along with its failure modes (the doc nobody made, the scan appended twice,
the entry that landed in yesterday's log past midnight).

A UI poll for *"what scans exist now"* is a different animal and is fine: it
is read-only and creates nothing. Do **not** subscribe to the RunEngine doc
stream — that would couple the logbook to the engine for no gain.

## Scan-folder invariant

This package is analysis-side code under the repository invariant (root
`CLAUDE.md`). It is a **consumer** of scan folders, never a producer:

- never construct `ScanPaths(read_mode=False)`
- never call `Path.mkdir` — not even `exist_ok=True`, in this phase
- a missing day or scan folder is *reported as absent*, never repaired

## Borrow the parsing, own the view

The primitives live one layer down, in `geecs_data_utils`, which already
owns scan folders — one surface to fix when a format changes:

| Need | Use | Never |
|---|---|---|
| Parse `ScanInfoScanNNN.ini` | `scan_paths.read_scan_info_file(path)` | a local `ConfigParser` |
| When a scan ran | `scan_log_loader.first_log_timestamp(path)` | the folder's `st_mtime` |

The folder's mtime is **not** a start time: any later pass that writes into
the folder (the analysis task queue's `analysis_status/`, for one) moves it,
and it has been measured over an hour off the real start.

What this package owns is the *logbook's* reading of those facts: the status
classification, campaign shaping, and the day document.

Pinned by `tests/test_scan_reader.py::TestScanFolderCreationInvariant`, which
monkeypatches `Path.mkdir` to explode.

When phase 02 adds writes, commentary goes to **`logbook/`, a sibling of
`scans/` and `analysis/`** — never inside a scan folder. That keeps the raw
data tree pristine, gives the day intro a home, and means the writer never
traverses `scans/ScanNNN/` at all.

## Status is reported, not inferred

`scan_status` classifies from `ScanEndInfo` only:

| Status | Meaning |
|---|---|
| `success` | `ScanEndInfo = "success"` |
| `failed` | starts with `fail` — the reason is surfaced verbatim |
| `aborted` | starts with `abort` — `RE.abort()`, Ctrl-C, or the queueserver stop the console and GEECS-MCP expose; reason surfaced the same way |
| `incomplete` | no `ScanInfo`, **or** `ScanEndInfo` still empty |
| `unknown` | a non-empty `ScanEndInfo` we do not recognise |

The chip must not contradict the card. `incomplete` covers two different
things, so the view splits them on `has_scan_info`: **"no scan info"** for a
bare folder, **"not finalised"** for one whose ScanInfo parsed fine and
whose facts are on screen. Rendering "no scan info" above a provenance line
naming the file it just read is the kind of small lie that costs trust in
everything else on the page.

The empty case is the one to get right. The scanner writes
`ScanEndInfo = ""` when it claims the folder and fills it in at the stop
document, so empty means *not finalised* — not *unrecognised*. It is the
most common state on the real share (37 of 49 ScanInfo files across four
sampled days), and classifying it as `unknown` painted most of a day amber.

`incomplete` therefore covers a scan still running, one that died before its
stop document, and development churn alike. An earlier version of this file
claimed those are "not separable from the folder". **That was wrong**:
`scan.log` is present in every such folder and this repo already parses it
(`geecs_data_utils.scan_log_loader`, `GEECS-LogTriage`). Separating them is
open, not impossible — it is simply not done yet, and the status vocabulary
should grow a `running` member when it is. Until then, report what the files
say and do not guess.

Surfacing `failure_reason` matters: it is the single most useful auto-filled
fact on a card, and it is the thing nobody remembers three weeks later.

## Templates seed, they never enforce

(Phase 03.) A template supplies the *initial text* of an entry body. The body
is one opaque markdown string; first save is copy-on-write and the text is
then entirely the author's. Changing a template never alters an existing
entry.

Do **not** store entries as structured fields keyed by template headings.
That is the trap: it makes a template edit retroactively change or hide
historical content, and turns every "can we add a field" into a migration.

## Deployment

No service, port, or unit of its own. GEECS-DataPortal mounts it:

```python
app.include_router(create_log_router(experiment), prefix="/log")
```

behind the portal's `log` extra and its `--scan-log` flag. It rides the
portal's existing checkout and systemd unit.

`create_log_router` takes the experiment explicitly — this package carries no
facility default, per the "facility values have one home" invariant.
