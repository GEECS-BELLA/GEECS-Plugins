# GeecsLogbook — Developer Context for Claude

The logbook: two books in one store. The **scans** book is the curated
per-scan record — a day document over scan folders, the successor to
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

`/day/2026-09-11` lists whatever `ScanNNN` folders exist at request time.
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
classification, day shaping, and the day document.

Pinned by `tests/test_scan_reader.py::TestScanFolderCreationInvariant`, which
monkeypatches `Path.mkdir` to explode.

Commentary is mirrored into **a tree the logbook owns**,
`{experiment}/logbook/Y2026/09-Sep/26_0911/…` — the data tree's date
shape, outside the data tree. It is always writable (a note on a day with
no scans has a home), backs up and syncs as one folder, and never enters
`scans/` at all: `mirror._assert_own_tree` refuses a day root whose own
four segments are not `logbook/Y/M/D` (the site's share root above them
is not inspected — `/mnt/scans/data` is a valid site), and a test
monkeypatches `Path.mkdir` across both the markdown and the attachment
copy to assert every directory made is under `logbook/`. One sentinel
stands before any `mkdir`: the **experiment directory** must exist,
because its absence means the share is not mounted, and a tree built on
the bare mount point would be hidden when the share came back. The first cut put `logbook/`
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
  author, include_scan_anchored)` is the month page, its filter chips and
  a search — one method, not three. A synchroniser asks a different
  question ("what changed"), answered by `changed_since` below.
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
  file. The row stays so a downstream copy can learn it went (through the
  change feed, the one listing that shows it) and an accidental delete is
  a field to clear.
- **Optimistic locking.** Every edit carries the `version` it read; a
  mismatch is a 409 with the current entry, never a silently eaten
  paragraph.
- **Migration is additive columns only** (`_ADDED_COLUMNS`): the table
  ships `CREATE TABLE IF NOT EXISTS`, and a column added later arrives via
  `ALTER TABLE` with a fill expression. Nothing else, deliberately.

`body_md` is opaque: `render.render_markdown` (markdown-it + nh3) is the
only thing that reads it, and only to draw it — plus the tag scan.

## The ops book reads by month, from the store alone

`/month/2026-09` is the other book: every `ops` entry in the month,
grouped by day, newest day first, with the tag chips as URL filters
(`?tag=laser`) and one composer that takes a date. It reads **only the
database** — `NotesStore.query`, one call — and never the share, which is
the point: on a day the share is slow, this page is not (pinned in
`tests/test_month.py`, which makes the share reader explode). The two
books point at each other: the day page carries "N ops notes today →"
into the month's day heading, and every day group on the month page
links to its day document. The pages share one set of Jinja macros
(`templates/_entries.html`): an entry and a composer look the same in
either book, and the editor script meets one shape of form.

## Navigation

`static/nav.js` is the other script both pages load: the rail calendar,
keyboard stepping and hover prefetch. Like the editor it reads its facts
off `<main id="logbook">` (`data-api`, `data-day` or `data-month`,
`data-prev`, `data-next`, `data-today`) and templates nothing.

The **calendar** is a `<details>` the script fills when opened. Its marks
come from `GET /api/month/{m}/days` — the store's per-day counts (one
grouped query) plus which day folders exist. The share side of that is
**one listing of the month folder** (`scan_reader.days_with_folders`),
never thirty per-day walks, and it is a lazy fetch by the *open*
calendar, so the month page keeps its promise: the page itself still
reads the store alone (pinned in `tests/test_nav.py`, which makes the
listing explode under the month page). A missing experiment directory
comes back as `share: false` — said, not hidden — and the store's marks
stand on their own.

"Faster day switching" is keyboard stepping (`←` `→` `t` `c`; refused
while any composer holds unsaved text — a shortcut must never discard a
note) and a `<link rel="prefetch">` added when the viewer **rests** on a
day or month link for 250 ms — the page they are about to open is served
warm. Nothing is prefetched on load: every day page is a share read, and
the rail offers fifteen of them; the dwell is what keeps a pass over the
rail from prefetching them all. The trade-off, stated: the pages send no
freshness headers, and Chrome may reuse a prefetched document for a few
minutes without asking, so a scan that landed between the hover and the
click appears on the next reload. `/today` and `/month/today`
are the bookmarkable names.

## The change feed

`GET /api/entries?since=<aware ISO 8601>` (`NotesStore.changed_since`) is
the synchroniser's listing — ARIEL's, first: everything whose
`updated_at` moved after `since`, oldest change first, **tombstones
included**. It is the one listing that returns a deleted row, because a
deletion is a change a downstream copy has to learn, and this is the
only place it can. Rows are ordered `(updated_at, rowid)` and a full
page carries a `next_cursor` that resumes after its last row, so two
entries sharing an `updated_at` across a page boundary cannot lose one.
`since` must carry a timezone — the columns are UTC `isoformat()`
strings and a naive value would be compared in a zone nobody stated.
Scans are not in this feed; a since-filtered scan list waits for a
caller that finds walking the day endpoints too slow.

## The editor

`static/editor.js` is the whole write path in the browser, one
implementation for every composer (per scan, per gap, the day, in-place
edits, and the month page). It is deliberately **not** a WYSIWYG
editor: the toolbar writes markdown around the selection and the body
stays the plain text the mirror holds. The two things people actually
need — paste a screenshot, paste a spreadsheet — are events on the
textarea: a pasted or dropped file goes to the upload endpoint and its
relative link lands at the cursor; a tab-separated or HTML-table paste
becomes a markdown table (a text-only paste must have a consistent
column count and a non-empty first header cell — tab-indented prose is
not a table, and a cross-tab with a blank corner from a text-only source
is pasted as text; spreadsheet apps supply the HTML form, which has no
such rule). A pasted **permalink** is the third:
it becomes a labelled reference to that note (see "References between
notes"). A brand-new entry has no id until saved, so
the first attachment saves it first ("autosaved", with Discard); Save is
then an edit. The page hands the script its facts through
`<main id="logbook" data-api data-day data-book>` and the type prefills
through one JSON block; nothing is templated into the script. A form on
a page with no single day (the month page) carries its own `.when` date
input. It sets no colours (the theme guard's rule).

**Every composer folds, in both books, on one contract** — three
attributes `editor.js` implements once:

| Attribute | On |
|---|---|
| `data-open-composer="X"` | anything that opens composer X |
| `data-compose-host="X"` | the hidden element holding it |
| `data-insert="X"` | an affordance row that folds away while X is open — optional |

Close folds without discarding (the form stays in the DOM with its text),
which is what makes Esc safe to wire to it, and it *refuses* while the
entry is already in the store — attaching a file autosaves one, so folding
the composer away would take its Discard button with it and silently
publish. The ops book had none of this until 0.11.0: its single composer
was wedged open at the top of the month, so there was nothing for Close to
fold and the button was never drawn.

The two books' affordances differ, deliberately. On the day page the rule
row **is** the position — the note lands between those two scans. The ops
book's composer has no position; it takes a date, which is what makes one
composer enough for a whole month, so its opener sits in the "New note"
heading beside the other actions and a day heading's "+ note" points it at
that day. Revealing, dating, focusing and scrolling all happen in
`editor.js` so they cannot come apart: they were two listeners on one
button, and the scroll ran while the composer was still hidden.
Pinned by `tests/test_composer_fold.py`.

## References between notes

A logbook whose notes cannot cite each other makes the reader carry the
connection — "the clock change is written up somewhere in last Saturday".
Three things have to hold, and they are separate.

**One name.** `GET /entry/{id}` redirects to whichever page draws the
entry — the day page for the scans book, the month page for the ops book —
anchored at `#entry-<id>`. A reader citing a note should not have to know
which book it is in, and the name survives a change of page shape. A
tombstone is a 404 like any other missing entry; the history endpoint is
where a deleted entry is still readable.

**A relative stored form.** A body holds `entry/<id>` and nothing else;
`render.render_markdown(entry_base=…)` swaps in the serving route at draw
time and marks the link `class="entryref"`. Same rule as an attachment,
for the same reason: the mount prefix and the host are deployment facts
and must never reach a stored body. Unlike an attachment link it does
*not* resolve in the mirrored markdown — it names a row, and only the
service can turn that into a page. Off-site it reads as a dead relative
link rather than as a wrong one; that is the honest failure, and it is the
price of not baking a URL into the record.

The rewrite is keyed on the **id's alphabet**, not on the word `entry`, so
an ordinary relative link that happens to sit under that prefix is left
exactly as the author wrote it.

**Arrival.** Everything on these pages folds — an entry is a `<details>`,
so is the scan block around it, and Collapse All is a stored per-viewer
preference — so a permalink routinely points *into* something shut, where
the browser scrolls to nothing and the reader sees the top of a day.
`nav.js` opens the target's ancestors, aligns it under the sticky topbar
(whose height it measures — the bar wraps on a narrow window) and lets
`:target` mark it. It runs on load and on `hashchange`, and beats the
Collapse All preference because that is applied by an inline script while
this file is deferred.

The composer's half is a paste: a pasted permalink is fetched and becomes
`[author · 12 Sep](entry/<id>)`. The label's date is spelled out from the
`YYYY-MM-DD` rather than handed to `toLocaleDateString`, which answers in
the viewer's locale ("Sep 12" beside the page's own "12 Sep") and in some
ICU versions abbreviates September to "Sept" — every other date on these
pages is `%-d %b`, rendered server-side. The pasted URL's **host is not
checked**: the same logbook is reached as a bare IP, as a name, and
through the front door's prefix, and a link copied on one is pasted on
another all the time. What validates the reference is the fetch that
follows — an id that is not here comes back 404 and the text is pasted
unchanged.

Copying runs on plain `http://`, which is the lab's own address and not a
secure context, so `navigator.clipboard` does not exist there; the Link
tool falls back to a selection copy rather than failing silently (giving
focus back afterwards — the tools are `opacity:0` until `:focus-within`,
so a keyboard user would otherwise watch the flash at opacity zero and
lose their place in the tab order). It is an `<a>` carrying the real URL,
so right-click "copy link address" and ⌘-click keep working whatever the
script does.

Two hazards the review of #890 found, both fixed there and both worth not
reintroducing:

- **The reference goes in synchronously; only its label is fetched.** An
  earlier cut fetched first and wrote nothing until it returned, so a ⌘↩
  in that gap — paste the link, save, the obvious motion — read the body
  without the citation and reloaded the pending write away, silently. The
  first version of the *fix* deferred the write behind a latch, which
  closed the loss but left the reference writing at the caret offsets
  remembered at paste time: type while the fetch is in flight and the link
  splices into the middle of it. Writing immediately removes both, and the
  label upgrade is applied by finding its own placeholder text rather than
  by position, so it is safe to lose. **The worst case is a reference that
  reads `[note]`.** Do not reintroduce a deferred write here; the only
  thing that defers a textarea write is an upload, which is why one
  `form._uploading` slot is enough.
- **The match is on the trailing `entry/<id>` pair, not on `ENTRY_BASE`.**
  Anchoring on this page's own prefix meant a link copied at
  `:8400/entry/<id>` and pasted into a page served under `/log` did not
  match — and the fallback then wrote the absolute URL, host and all, into
  the stored body. That is the one thing this whole design exists to
  prevent. It also means the bare stored form pasted out of one note's raw
  markdown is recognised.

A permalink can also name a note the **scans book cannot draw**: an entry
anchored to a scan whose folder is not on the share is stored and counted
but has no block to hang on. `reveal()` says so rather than leaving the
reader at the top of an apparently ordinary day.
Pinned by `tests/test_crosslinks.py`.

## Status is reported, not inferred

`scan_status` classifies from `ScanEndInfo` only:

| Status | Meaning |
|---|---|
| `success` | `ScanEndInfo = "success"` |
| `failed` | starts with `fail` — the reason is surfaced verbatim |
| `aborted` | starts with `abort` — `RE.abort()`, Ctrl-C, or the queueserver stop the scanner and GEECS-MCP expose; reason surfaced the same way |
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

A template supplies the *initial text* of an entry body. The body is one
opaque markdown string; first save is copy-on-write and the text is then
entirely the author's. Changing a template never alters an existing
entry — the entry keeps only the template's *name* (`template`), as
provenance.

Do **not** store entries as structured fields keyed by template headings.
That is the trap: it makes a template edit retroactively change or hide
historical content, and turns every "can we add a field" into a migration.

**Templates are data, not code** (`seed_templates.py`): `*.md` files in
`logbook_templates/` at the top of the configs checkout the portal
already reads — the parent of its `--processing-configs` tree. Each file
is a `key: value` header between `---` lines (`label`, `colour`, `book`,
`order`, all optional) and a body that is the prefill, carrying the
type's `#tag` so the button and a typed tag are the same thing. Adding a
file adds a button on the composers of the book(s) it names; no code
change, no restart (the set is re-read in the background when stale, and
a failed re-read keeps the last set — the share is never on a page's
critical path). `examples/logbook_templates/` is the documented set to
copy into the configs repo.

`colour` is a **theme token name** from the closed vocabulary
`seed_templates.TONES` (`accent`, `ok`, `warn`, `crit`, `agent`,
`muted`, `trace-1`…`trace-4`), never a literal: the page turns it into a
`.tone-<name>` class whose value is `var(--<name>)`, so a type follows
whichever palette the viewer picked and GeecsWebTheme's literal-colour
guard stays true for a value it cannot see. `tests/test_seed_templates.py`
pins the vocabulary against the stylesheet in both directions; an unknown
name falls back to `accent` with a warning rather than dropping the
button.

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

**Filed as #839** (2026-09-11): review `ScanPaths` / `ScanData` and
extract their pure parts into path-free utilities. This package's
`scan_reader` is a sketch of what that looks like. The whole point of
accepting duplication now is that someone later removes it — the issue
is where that is tracked.

### Also deferred, no due date

| Item | Why |
|---|---|
| `ScanSummary` vs `scans_database.entries.ScanMetadata` overlap | Same reasoning: consolidating means adopting the model layer under review. Revisit with the ScanPaths issue above. |
| Two day views — the portal's `/day/` (Tiled runs) and the logbook's `/day/` (scan folders) — can disagree | A scan Tiled never received appears in one; a folder predating the catalog appears in the other. **Owner ruling (2026-09-13): the scan folders are canonical for now** — that matches the LabVIEW Master Control implementation the lab runs today. The folders are not downstream of Tiled: the s-file, `ScanInfo` and the per-device files are written from the same RunEngine document stream Tiled records (and by the device servers), which is exactly why the two views can differ. Tiled is expected to become canonical later; when it does, this row is where the logbook's reader changes. Until then a disagreement is resolved in the folders' favour, never by writing one. |
| Package name vs `geecs_data_utils.scan_log_loader` and `GEECS-LogTriage`, which read `scan.log` | This package is about the *logbook*, not `scan.log`, and `scan_reader` now imports `scan_log_loader`. Renaming costs one commit today and more later. |
| Separating "running" from "aborted" from churn, for folders with no ScanInfo | Open, not impossible — `scan.log` is in every such folder. Add a `running` status when it is done. |
| Off-site reading | The mirror tree is one folder, so a text-only `git push` of it to a private repository is cheap whenever wanted; the Google Doc exporter (blocked on credential rotation) is the route that carries images. Neither is needed for a functional logbook. |
| The scan index | A month-partitioned redevelopment of `geecs_data_utils.scans_database` with an `update(day)` entry point, the portal as its writer. Parked by the owner (2026-09-11) until the two books are live. |
| `EntryCreate` (the write shape) lives in `routes/entries.py`, `LogEntry` (the stored shape) in `geecs_schemas` | An agent posts the create shape, so it belongs beside `LogEntry` for GEECS-MCP to validate. Moves with the agent-verbs phase, which is its first second consumer. |
| A third private atomic-write helper (`_fs.replace_with`; `scan_analysis.config_store` and `task_queue` have their own) and `logbook_root` re-deriving the daily folder | Fold into the `ScanPaths`/`ScanData` review, #839 — same home, same issue. |
| One home for the entry-id alphabet | The store mints `uuid4().hex[:12]`; `render._ENTRY_REF` and `editor.js`'s `ENTRY_ID` each re-declare it, and `LogEntry.entry_id` carries no `pattern` (unlike `day`). A pattern in `geecs_schemas` is the real home, but it is a cross-package change for a UX fix. Until then `tests/test_crosslinks.py::TestTheIdAlphabetHasNotDrifted` ties both matchers to a freshly minted id, so a change to the minting fails loudly instead of silently breaking paste-recognition and reference-marking. Raised in the review of #890. |
| Copy-on-plain-http, twice | `editor.js`'s `copyText` and the portal's `run.html` `copyPlotImage` both work around the absent `navigator.clipboard` on an http host. They are not mergeable as they stand (text into a control's own label vs an image blob into a corner toast), and `GeecsWebTheme/CLAUDE.md` makes the **third** surface the forcing function. When the web console needs a copy button, `kit.js` is the home — not a third copy. Raised in the review of #890. |
| Whether the topbar measurement must wait for the pickers | `reveal()` measures `.topbar` inline. The review of #890 argued it must wait, because kit.js and theme.js build the theme and density pickers on `DOMContentLoaded` — correct about the ordering. Measured cold at a fragment, at 1380px and at 560px (two rows), inline and deferred place the target identically, so the deferral was deleted. The reviewer's narrower point stands unsettled: the pickers add ~200px of width, so there is in principle a band between those two widths where the static bar fits one row and the full bar needs two, and inside it an inline measure would be a row short. Settle it with a width sweep on the month page (both pickers unbuilt there) comparing the bar's height with and without them, not by adding the deferral back on reasoning. Raised in the review of #890. |
| Backlinks — "notes that link here" | A reference is a link inside an opaque body, so the reverse direction needs an index of what points where, maintained at every save and edit. Worth it once people are citing enough to lose track; not for the handful the feature starts with. The forward link is the half that carries the value. |
| Which template "started" an entry when several buttons were pressed | The last one pressed is recorded. Provenance only; nothing reads it back but the chip. |
| `scan_reader.month_folder` / `days_with_folders` are a third copy of the share-layout walk (`.parent` chains up from `get_daily_scan_folder`; a `YY_MMDD` parser beside `ScanPaths.get_scan_tag`'s and `scans_database.builder`'s `strptime("%y_%m%d")`), after `mirror.logbook_root` and `read_day` | The layout has one builder in `geecs_data_utils.scan_paths` and should have one reader there (`day_folder_date(name)`, `list_day_folders(month)`), which is exactly #839's brief. Recorded on #839 at the review of #844; not lifted here so the logbook keeps depending on `ScanPaths` alone. |
| `seed_templates.parse_template` is the package's first front-matter reader, while `mirror.render` is its writer | One reader, one writer, different shapes today (the template header has no lists). When the mirror *reader* lands (off-site/rebuild, deferred above), extract one `parse_front_matter` beside `mirror` and point both at it — not before there is a second caller. Waived in the review of #842. |

## Deployment

Its own service since 0.10.0: `geecs-logbook` (the console script →
`__main__.main`) serves one experiment on port **8400** behind the unit
template `deploy/geecs-logbook.service`, with the entries in systemd's
`StateDirectory` (`/var/lib/geecs-logbook`). `deploy/DEPLOYMENT.md` is
the runbook — install, the one-time move of the entries out of the
portal's state directory, the proxy prefix, troubleshooting. Before
0.10.0 it was a router the Data Portal mounted at `/log`; that mount and
the portal's `log` extra are gone, and the portal now reaches this
service over HTTP alone (`--logbook-url`) — it links to a scan's card,
and since portal 0.28.0 it also **writes**: the Plot tab's "send plot to
this scan's log entry" creates an entry, uploads the PNG and patches the
image into the body, through the public verbs in `routes/entries.py` and
`routes/attachments.py` (`GEECS-DataPortal/geecs_portal/logbook_send.py`,
`tests/test_logbook_send.py` there). It is an ordinary API client — no
import edge, no special casing — and two of this package's own rules are
what shape it: an attachment needs an entry to exist first, and
`render.py`'s `_IMAGE_RUN` grids only *consecutive* image paragraphs, so
the sender appends and keeps its link above them. Changing either is a
change to that caller.

`app.create_app(experiment, *, base_directory, notes_db, templates_dir,
root_path)` is the one entry point. It takes the shared web glue from
`geecs_web_theme.web` — the forwarded-prefix middleware, the `/theme`
mount, the templates factory that puts `root` in every context — never a
copy; the routes live in `routes/` (`day`, `month`, `entries`,
`attachments`, one module per concern) and register on one router the
app includes at its root. The templates address every asset and link
root-relatively (`{{ root }}/static/…`, `url_for(...).path`), so the
service works at root and under `/log` at the front door alike.

`--notes-db` names the SQLite file (default `$STATE_DIRECTORY/logbook.db`
under systemd); uploads go to `attachments/` beside it. Without a store
the app has no write routes at all. `--templates-dir` names the
seed-template directory (the unit points it at `logbook_templates/` at
the top of the configs checkout).

`create_app` takes the experiment explicitly — this package carries no
facility default, per the "facility values have one home" invariant.
