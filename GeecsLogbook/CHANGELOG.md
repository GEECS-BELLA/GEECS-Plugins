# Changelog

All notable changes to `geecs-logbook` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to semantic versioning.

## [0.5.0] - 2026-09-11

The ops book: the second book gets its page, and entries get types.

### Added

- `GET /log/month/{YYYY-MM}` — the ops book: every `ops` entry in the
  month grouped by day, newest day first, one composer with a date
  picker, prev/next month and a month picker, and tag chips that filter
  through the URL (`?tag=laser`; chips count the whole month while the
  list shows the filtered part). Reads only the notes store — never the
  share — so it stays fast when the share is slow (pinned).
  `GET /log/api/month/{YYYY-MM}/entries?book=&tag=` is its JSON peer.
- **Seed templates as type buttons** (`seed_templates.py`): `*.md` files
  in a directory the host names (`create_log_router(templates_dir=)`;
  the portal points it at `logbook_templates/` in the configs checkout).
  Front-matter `label` / `colour` / `book` / `order`, body = the prefill
  with its `#tag`. A button press inserts the prefill and the entry
  records the template's name as provenance; a stored entry shows its
  template as a chip in the file's tone. `colour` is a theme token name
  from a closed vocabulary (`TONES`, pinned against the stylesheet), never
  a literal. Read once at start, refreshed in the background when stale;
  a failed refresh keeps the last set. `examples/logbook_templates/` is
  the documented starter set.
- Cross-links between the books: the day page's "N ops notes today →"
  strip (into the month page's day heading) and an "Ops book" link in its
  topbar; each day on the month page links to its day document.

### Changed

- The entry and composer markup moved into shared macros
  (`templates/_entries.html`) so both pages draw them identically.
- `editor.js` takes the day per form (a `.when` date input or `data-day`,
  falling back to the page's), reads the prefills from one JSON block,
  and sends `template` with a create.
- `CLAUDE.md`: the owed ScanPaths review is filed as #839.

## [0.4.0] - 2026-09-11

The editor: what makes people use it.

### Added

- `static/editor.js`, one implementation for every composer on the page
  (per scan, per gap, the day, and in-place edits): a toolbar that writes
  markdown around the selection (bold, italic, code, list, task list,
  link, table, callout, image, preview); **paste or drop a screenshot**
  into the composer — the file goes to the upload endpoint and a relative
  image link lands at the cursor; a brand-new entry is saved first so the
  upload has somewhere to go ("autosaved"), with Discard to delete the
  stub; **paste a spreadsheet range** (tab-separated or an HTML table) as a
  markdown table; a server-rendered Preview; ⌘/Ctrl+Enter saves.
- `POST /api/preview` — the page's renderer, for the composer.
- Two or more consecutive images render as one grid (`figgrid`): the
  plot table, replacing LogMaker's `gdoc_slot` numbering — no slot
  assignment, no ceiling at four.

### Changed

- The day page's inline write-path script is gone; the page hands the
  editor its facts through `<main id="logbook" data-api data-day
  data-book data-accept>` and loads `editor.js`.

### Fixed (review of #837)

- An uploaded image link was left selected, so the next keystroke — or
  the second file of a multi-file drop — replaced it. Attachment links
  now insert with the caret after them; only the table and callout
  skeletons stay selected for overtyping.
- Duplicate entries from a held ⌘↵ (key repeat), a double submit, or two
  quick pastes on a new composer: one create per form (the pending POST
  is shared), key-repeat ignored, Save/Discard exclusive with the buttons
  disabled meanwhile.
- The version is re-read after uploads even when a later file in the
  batch failed, and written back to the article for an in-place edit so
  Cancel-then-Edit does not start stale.
- Attaching before entering a name no longer creates an entry attributed
  to "unknown" forever: the name is required first.
- Paste precedence: an HTML table on the clipboard wins over a bitmap of
  the same cells (Excel); a file pasted alongside text keeps the text and
  uploads the file.
- Tab-indented prose is not turned into a table (consistent column count
  and a non-empty header cell are required); list/task prefixing at the
  start of the text and on a triple-clicked line; a failed preview leaves
  nothing visible; a dropped file of a type the server does not take says
  so instead of vanishing. Accepted types come from the page
  (`data-accept`), one list not two. The theme guard walks `editor.js`.
- Between-scan blocks collapse as well as expand (owner's hand-test):
  they are `<details>` like the scans, with a count in the header, and
  Expand all / Collapse all reaches them.

## [0.3.0] - 2026-09-11

The foundation for the operations book. Owner rulings 2026-09-11.

### Added

- **Two books.** `book` (`scans` | `ops`) on every entry, chosen by the
  page the author writes from. An ops entry is day-level only. The day
  document shows the scans book; `GET /api/day/{day}/entries?book=`
  filters either way.
- **Tags from the body.** `#laser` in the text is the tag;
  `tags.parse_tags` reads them at every save into an indexed `tags`
  column (narrowly: not `#1`, not headings, not inside code). Rendered as
  chips.
- **History.** `entry_history` keeps the entry as it was before every
  edit, keep/un-keep, upload and delete, in the same transaction;
  `GET /api/entries/{id}/history` serves it. `GET /api/entries/{id}`.
- **`NotesStore.query`** — a day range with book, tag, kind, status,
  author and scan-anchored filters: the one method the month page, its
  filter chips and a synchroniser will share.
- **Attachments are store-first.** `attachments.AttachmentStore` keeps
  uploads under `attachments/` beside the database; the page serves them
  from `/log/attachments/{entry_id}/{filename}` and the mirror copies them
  beside the markdown. A screenshot pasted with the share unmounted lands.
- Additive column migration for `book` and `tags`.

### Changed

- **The mirror owns its tree.** Entries mirror into
  `{experiment}/logbook/Y2026/09-Sep/26_0911/…` — the data tree's date
  shape, outside it — instead of `logbook/` inside each day folder. The
  tree is always writable, so a note on a day with no scans has a home
  and the "deferred forever" state is gone; `mirror._assert_own_tree`
  refuses any path with `scans` in it. `write_attachment` is replaced by
  `mirror_attachments` (from the store to the share).
- The router is assembled from `routes/day`, `routes/entries` and
  `routes/attachments`; `router.create_log_router` only wires them.
- Front matter carries `book` and `tags`.

### Fixed (review of #835)

- **`/log/attachments/../logbook.db` served the database.** The
  attachment store checked that a file sat inside the entry directory
  but not that the entry directory sat inside the root; `..` as an entry
  id resolved to the state directory. Both levels are now checked, and
  the test's decoys sit where an escape would land.
- A share root containing a `scans` component (`/mnt/scans/data`) made
  every save 500 after the row was written: the invariant assert now
  inspects only the mirror's own `logbook/Y/M/D` segments and raises
  `MirrorUnavailable`; anything else the mirror raises after the row
  landed is logged and deferred, never surfaced as a failed save.
- A dropped share no longer gets a logbook tree built on the bare mount
  point: the experiment directory must exist before any `mkdir`.
- Deleting a tombstone (or a missing id) no longer records a spurious
  history snapshot; an edit racing a delete is a 404, not a 409 with a
  body that no longer exists.
- `query(limit=-1)` is bounded.
- Tags: a markdown anchor link `[see](#results)` and `?#top` are not
  tags; a trailing `-` is dropped; the tail is Unicode-aware (`#eé` is
  not the tag `e`).
- `update` uses the shared transaction helper; one `attachments`
  constant; `?book=` typed as `Book`; the unused per-view ops count is
  gone until the month page; `geecs_schemas` exports `Book`; docs that
  still described the day-folder mirror and an unparsed body corrected.

## [0.2.0] - 2026-09-11

### Added

- **Commentary.** `geecs_logbook.store.NotesStore` — SQLite (WAL) with
  optimistic locking (`version`, `ConflictError` carrying the current
  entry) — and `geecs_logbook.mirror`, which writes each entry as
  front-matter markdown into the day's `logbook/` folder on the share
  (a sibling of `scans/`; never inside a scan folder, never creating the
  day). The store is written first; `mirror.sync` pays the debt when the
  share is back.
- Entry routes on the router when `notes_db` is given: create, edit
  (409 on a stale version), keep/un-keep, delete, attachment upload
  (20 MiB; png/jpeg/gif/webp/pdf), plus `GET /api/day/{day}/entries`.
  The day page grows a composer per scan, per gap and for the day.
- `geecs_logbook.render.render_markdown` — markdown-it (commonmark +
  tables + strikethrough + task lists) sanitised by nh3, with `> [!NOTE]`
  callouts and attachment links rewritten to the serving route.
- Day-level entries: neither `scan` nor `after` — a note about the day.
- `updated_at` on every entry (moves on any change; `edited_at` only on
  text) and `deleted_at` tombstones instead of row removal.
- An agent's entry (`kind` other than `note`) cannot be created `kept`;
  the store refuses it and the route answers 422.
- Additive column migration for an existing database file.

### Changed

- **Renamed from `GeecsScanLog` / `geecs_scan_log`.** The scan logger is
  the archetype for a general logbook, so the package is named for what it
  is becoming. Distribution name `geecs-logbook`; the portal's `log` extra
  follows.
- The day intro (`scan=0`) is gone; `scan` starts at 1 and the intro card
  holds the day-level entries. Their mirror files sit at `logbook/` root
  under the same stamped name as every other entry (no `day.md`).
- `mirror.logbook_root` raises `MirrorUnavailable` when the share cannot
  be resolved at all (no configuration, unmounted), so a save on such a
  host still returns 201 with the file owed rather than a 500 after the
  row was written.
- The page takes its colours from `geecs_web_theme`; no palette of its
  own.

### Fixed (review of #832)

- Editing an entry whose text held `'`, `"`, `<`, `>` or `&` fed the
  HTML-escaped form back into the editor and saved it. The raw body now
  travels as JSON.
- The mirror queue rotates: a failed attempt records `mirror_attempted_at`
  and never-tried entries go first, so an entry whose day folder never
  appears cannot starve the ones behind it.
- Mirror filenames and the page's time stamps are the host's local time,
  the clock the day and its scans are named by, not UTC.
- Two uploads with the same name no longer overwrite each other
  (`image-2.png`, …); each attachment has its own id; the manifest append
  is one SQL statement, so concurrent uploads both land.
- The upload route runs in the threadpool rather than blocking the event
  loop on a share write; an unresolvable share is a 503 on upload and on
  attachment serving, not a 500.
- Task-list checkboxes survive sanitising.
- Mirroring is serialised per process (`mirror.WRITE_LOCK`) and reads
  the entry afresh under the lock, so the periodic sync can never write
  an older body over a file a request just mirrored; the mark is pinned
  to the version written. Temp files carry unique names and the mode a
  plain write would have had (mkstemp's 0600 is not for a mirror people
  read).
- An edit no longer changes the entry's `author` (which is part of the
  mirror file's stable name — the old behaviour left a stale file behind
  and re-attributed the entry to whoever fixed a typo). The editor is
  recorded as `edited_by` and shown as "edited by …". `PATCH` takes
  `editor`, not `author`.
- Same-name uploads are numbered by claiming the name on disk
  (`O_EXCL`), so pastes in flight at once cannot collide.
- New tests for the renderer (sanitiser, callouts, link rewrite, task
  lists) and the attachment routes (upload, serve, size and type limits,
  traversal, unresolvable share).

## [0.1.0] - 2026-09-11

### Added

- Initial package: a read-only day-document view over scan folders.
- `geecs_logbook.models` — `ScanSummary` and `DaySummary`, the derived
  view of a scan folder. Nothing here is stored by the logbook.
- `geecs_logbook.scan_reader.read_day` — lists a day's `ScanNNN` folders
  and parses each `ScanInfoScanNNN.ini` into a `ScanSummary`. Read-only by
  construction: it never constructs `ScanPaths(read_mode=False)` and never
  calls `mkdir`.
- Scan status derived from `ScanEndInfo`: `success`, `failed` (with the
  failure reason surfaced), `incomplete` (folder exists, no ScanInfo), or
  `unknown`.
- `Campaign` — consecutive scans sharing a parameter and purpose group into
  one run. Derived from what the scanner already wrote, so nobody declares a
  campaign and nobody can forget to. A day above 20 scans renders campaigns
  (one rail row each, collapsed) instead of a flat list; a failure inside a
  collapsed campaign still surfaces on its header.
- Fewer round trips per scan: one `os.scandir` of a scan folder yields the
  ScanInfo path, that file's own stat (the cache key), the `scan.log` path
  and the device list — replacing a `glob`, an `iterdir` and a `stat`.
  Measured against the original reader at 1.85x on cold days (403 -> 218 ms
  per scan, A/B across untouched August dates). That figure predates the
  `scan.log` read added for start times, which costs one more open per
  uncached scan; and the listing itself is not cached, so a warm 108-scan
  day is ~300 ms rather than the single-digit milliseconds an earlier draft
  of this entry claimed.
- Concurrent folder reads (16 workers) and a cache of the per-scan file
  reads, keyed on the **ScanInfo file's** own mtime and size. A 108-scan day
  over VPN went from 27.3 s to 5.6 s on a cold share and milliseconds once
  cached. The key is the file's, never the folder's: the scanner finalises a
  scan by rewriting ScanInfo in place, which changes no directory entry, so
  a folder-keyed cache served a running scan's empty `ScanEndInfo` forever —
  losing exactly the failure reason this view exists to surface.
- Parsing borrowed rather than reimplemented: `ScanInfo` through
  `geecs_data_utils.scan_paths.read_scan_info_file` (shared with
  `ScanPaths.load_scan_info`) and the scan's start time through
  `scan_log_loader.first_log_timestamp`. One surface to fix per format.
- `abort` is its own outcome with its reason surfaced, rather than falling
  through to `unknown`: `RE.abort()`, Ctrl-C and the queueserver stop the
  console and GEECS-MCP both expose all reach the scanner as
  `exit_status="abort"`.
- The `incomplete` chip reads **"not finalised"** when ScanInfo parsed and
  **"no scan info"** only when there is none — it no longer claims a card
  full of parsed ScanInfo facts has no ScanInfo.
- Scan start falls back to the ScanInfo file's mtime, marked approximate,
  for archive scans with no `scan.log`; dropping the fallback entirely made
  a whole 2025 day render every time as an em dash.
- `ScanEndInfo = ""` classifies as `incomplete`, not `unknown`. The scanner
  writes it empty at claim time and fills it at the stop document, so empty
  means *not finalised*; it is the most common state on the real share (37
  of 49 ScanInfo files across four sampled days) and reporting it as
  "unrecognised" painted most of a day amber.
- Scan start time comes from `scan.log`, not the folder's modification time.
  Any later pass that writes into a scan folder moves that timestamp — it
  was measured over an hour off the real start.
- `/log/static/{name}` is a plain route, not `router.mount(StaticFiles(...))`.
  A `Mount` is a `BaseRoute`, not a `Route`, and `APIRouter.include_router`
  drops it silently on FastAPI versions this package's floor allows, 404ing
  the stylesheet and 500ing the page while CI stays green on a newer pin.
- Unpadded scan folders (`Scan42`) are no longer invisible, matching
  `geecs_log_triage.harvester`.
- A malformed numeric field (`inf`, `nan`) reads as absent instead of
  escaping `int()` and 503-ing the whole day.
- Expand/collapse reaches scans inside campaigns, not just the campaigns.
- Day navigation: a date picker, previous/next-day steps, a "back to today"
  link, and quick links centred on the shown date so stepping forward is as
  easy as stepping back.
- The page's palette comes from `geecs_web_theme`; `scanlog.css` defines no
  tokens of its own, and the picker in the top bar switches every surface.
- `CLAUDE.md` records the deferred decisions from phase 01's review,
  including the owed issue to review `ScanPaths`/`ScanData` and extract
  their pure parts — the price of the duplication accepted here.
- `geecs_logbook.router.create_log_router` — an `APIRouter` the Data
  Portal mounts at `/log`, serving `/log/day/{date}` and a JSON peer at
  `/log/api/day/{date}`.
