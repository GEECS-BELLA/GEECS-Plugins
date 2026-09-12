# GeecsLogbook — Developer Context for Claude

The logbook: two books in one store. The **scans** book is the curated
campaign record — a day document over scan folders, the successor to
`LogMaker4GoogleDocs`, which this package will eventually replace
outright. The **ops** book is routine operations, read by month. They
share every mechanism below and differ only in which page you write from.

## The one idea

The log is a **rendered view over (derived scan data) + (a small commentary
store)**. Those two halves have completely different properties and must not
share a store:

- **The record** — scan number, parameters, shot counts, status. Derived from
  `ScanInfoScanNNN.ini` and the folder listing on every request. The logbook
  stores *none* of it, so it can never drift from the data and an upstream
  change needs no migration here.
- **The commentary** — what people wrote. Irreplaceable, free-form, and the
  only thing that genuinely needs storage. See "The commentary store".

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
- the reader never calls `Path.mkdir`; the mirror calls it only inside the
  logbook's own tree (below), and refuses any path with `scans` in it
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

Commentary is mirrored into **a tree the logbook owns**,
`{experiment}/logbook/Y2026/09-Sep/26_0911/…` — the data tree's date
shape, outside the data tree. It is always writable (a note on a day with
no scans has a home), backs up and syncs as one folder, and never enters
`scans/` at all: `mirror._assert_own_tree` refuses any path with `scans`
in it, pinned in `tests/test_mirror.py`. The first cut put `logbook/`
inside each day folder and could not create the day; that stranded the
ops book, which is why it moved (owner ruling 2026-09-11).

## The commentary store

`store.NotesStore` (SQLite, WAL) is authoritative; `mirror` writes each
entry as a front-matter markdown file into the logbook tree, second. The
store is written first so that a save never fails because the share is
slow or unmounted — `mirrored_at` stays null and `mirror.sync` (throttled,
on day views) pays the debt later. Files-as-truth was considered and
rejected for exactly that reason: it puts an SMB write in the save path.
Git-as-truth was considered too (history and sync for free) and rejected
because the attachment load — figures published from the portal — is a
filesystem's job, not a content store's.

**Bytes follow the same rule.** `attachments.AttachmentStore` keeps
uploads under `attachments/` beside the database file; the page serves
them from there and the mirror copies them beside the markdown. So
everything irreplaceable is one directory — the service's state directory
— and that is the whole backup story.

The entry's shape is `geecs_schemas.log_entry.LogEntry`, defined there so
GEECS-MCP and a mirror reader can agree on it without importing this
package. Rules the store enforces, each pinned in `tests/test_store.py`:

- **Two books.** `book` is `scans` or `ops`, chosen by the page the
  author writes from — never inferred from the timestamp. An ops entry is
  day-level only; the scans book takes the anchors below.
- **Three anchors.** `scan=`, `after=` (interscan), or neither — a
  day-level entry. Both is refused. There is no `scan=0`; scans start at 1.
- **Tags come out of the body.** `#laser` in the text is the tag;
  `tags.parse_tags` reads them at every save into an indexed column. A
  type button inserts a template whose prefill carries its tag, so a
  button press and a typed tag are the same thing. Nobody sets `tags`
  directly, and there is no category column to drift from the text.
- **History.** Every edit, keep/un-keep, upload and delete first snapshots
  the entry as it was into `entry_history`, in the same transaction.
  `GET /api/entries/{id}/history` serves it; undo is a new edit with an
  old body. Nothing rewrites history.
- **One query.** `NotesStore.query(day range, book, tag, kind, status,
  author, include_scan_anchored)` is the month page, its filter chips, a
  search, and a synchroniser's "everything since" — one method, not four.
- **An agent's entry is born a draft.** `kind` other than `note` with
  `status="kept"` is refused at creation. A person keeps it via the status
  route; an agent has no route to promote itself. A *promoted* agent entry
  is a valid stored state, which is why the schema does not carry this
  rule.
- **`updated_at` moves on every change; `edited_at` only when the text
  does.** The reader is told about edits; a synchroniser asks for
  `updated_at` — a promotion or an upload would otherwise be invisible to
  "everything since".
- **Delete is a tombstone.** `deleted_at` is set, listings hide the row,
  writes to it fail as if it were missing, and the mirror sync removes the
  file. The row stays so a downstream copy can learn it went and an
  accidental delete is a field to clear.
- **Optimistic locking.** Every edit carries the `version` it read; a
  mismatch is a 409 with the current entry, never a silently eaten
  paragraph.
- **Migration is additive columns only** (`_ADDED_COLUMNS`): the table
  ships `CREATE TABLE IF NOT EXISTS`, and a column added later arrives via
  `ALTER TABLE` with a fill expression. Nothing else, deliberately.

`body_md` is opaque: `render.render_markdown` (markdown-it + nh3) is the
only thing that reads it, and only to draw it.

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

## Deferred decisions — do not re-litigate, do not lose

Reviewed and deferred deliberately during phase 01 (owner ruling,
2026-09-11). Each is recorded here so a later phase does not re-open it by
accident, and so the ones with a due date actually get done.

### `scan_reader` stays in this package

An adversarial review argued it is pure logic that belongs in
`geecs_data_utils`, next to `ScanPaths`, `scans_database/` and
`scan_log_loader`. The argument is good and the owner agreed with its
*direction* — but not its conclusion, for a specific reason:

> the data-utils package in this context is quite weak — fairly convoluted
> and confusing. A simple 'scan info' reader that doesn't care about paths
> seems like the right thing.

So the shape this package wants is a *pure* reader with no path-object
ceremony, and moving it under `ScanPaths` would bind it to the very design
that needs fixing. The parse itself **is** shared
(`read_scan_info_file`, `first_log_timestamp`) — that was the half worth
doing now. A little duplication in the day-walking and the summary model is
accepted in exchange.

**OWED, at the arc's merge to `master`:** file an issue to review
`ScanPaths` / `ScanData` and extract their pure parts into path-free
utilities. This package's `scan_reader` is a sketch of what that looks
like. Do not let the arc land without filing it — the whole point of
accepting duplication now is that someone later removes it.

### Also deferred, no due date

| Item | Why |
|---|---|
| `ScanSummary` vs `scans_database.entries.ScanMetadata` overlap | Same reasoning: consolidating means adopting the model layer under review. Revisit with the ScanPaths issue above. |
| Two day views — the portal's `/day/` (Tiled runs) and `/log/day/` (scan folders) — can disagree | A scan Tiled never received appears in one; a folder predating the catalog appears in the other. Needs an owner ruling on which is canonical, not an implementation choice. |
| Package name vs `geecs_data_utils.scan_log_loader` and `GEECS-LogTriage`, which read `scan.log` | This package is about the *logbook*, not `scan.log`, and `scan_reader` now imports `scan_log_loader`. Renaming costs one commit today and more later. |
| Separating "running" from "aborted" from churn, for folders with no ScanInfo | Open, not impossible — `scan.log` is in every such folder. Add a `running` status when it is done. |
| Off-site reading | The mirror tree is one folder, so a text-only `git push` of it to a private repository is cheap whenever wanted; the Google Doc exporter (blocked on credential rotation) is the route that carries images. Neither is needed for a functional logbook. |
| The scan index | A month-partitioned redevelopment of `geecs_data_utils.scans_database` with an `update(day)` entry point, the portal as its writer. Parked by the owner (2026-09-11) until the two books are live. |
| `EntryCreate` (the write shape) lives in the router, `LogEntry` (the stored shape) in `geecs_schemas` | An agent posts the create shape, so it belongs beside `LogEntry` for GEECS-MCP to validate. Moves with the agent-verbs phase, which is its first second consumer. |
| A third private atomic-write helper (`_fs.replace_with`; `scan_analysis.config_store` and `task_queue` have their own) and `logbook_root` re-deriving the daily folder | Fold into the `ScanPaths`/`ScanData` review owed at master-merge — same home, same issue. |

## Deployment

No service, port, or unit of its own. GEECS-DataPortal mounts it:

```python
app.include_router(create_log_router(experiment), prefix="/log")
```

behind the portal's `log` extra and its `--scan-log` flag. It rides the
portal's existing checkout and systemd unit. `--notes-db` names the SQLite
file (the portal defaults it to systemd's `StateDirectory`); uploads go
to `attachments/` beside it. Without a store the router has no write
routes at all. The routes live in `routes/` — `day`, `entries`,
`attachments`, one module per concern — and `router.create_log_router`
only assembles them.

`create_log_router` takes the experiment explicitly — this package carries no
facility default, per the "facility values have one home" invariant.
