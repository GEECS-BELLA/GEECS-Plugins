# Changelog

All notable changes to `geecs-logbook` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to semantic versioning.

## [0.10.2] - 2026-09-13

### Changed

- `deploy/DEPLOYMENT.md` corrected from the live deploy on the reference
  host: the backup steps use Python's `sqlite3` module (the `sqlite3` CLI
  is not installed there and `sudo sqlite3 …` failed as one quiet line
  while the move went on), the install step reads `GEECS_POETRY` from
  `site.env` (`poetry` is not on the login `PATH` either), and the
  verification counts rows through the change feed, tombstones included
  — a day whose entries were all deleted shows none on its page by
  design. Records the 2026-09-13 deploy.

## [0.10.1] - 2026-09-13

### Changed

- `tests/test_seed_templates.py`'s tone guard reads `scanlog.css` through
  the theme's parser-based `token_indirection_map` (the theme's `testing`
  extra, taken in the dev group) instead of a formatting-shaped regex —
  the #875 lesson applied to the last regex CSS scanner in this package.
  Mutation-proved: renaming a `.tone-` rule fails it.
- `CLAUDE.md` records the owner ruling on the two day views: the scan
  folders are canonical for now (they match the LabVIEW Master Control
  implementation the lab runs), Tiled expected to become canonical later.

## [0.10.0] - 2026-09-13

### Changed

- **The logbook is its own service.** `geecs-logbook` (console script,
  `python -m geecs_logbook`) serves one experiment on port 8400 —
  `--experiment`, `--notes-db` (default `$STATE_DIRECTORY/logbook.db`),
  `--templates-dir`, `--root-path` — behind its own unit template
  (`deploy/geecs-logbook.service`, `StateDirectory=geecs-logbook`) with
  a runbook (`deploy/DEPLOYMENT.md`, including the one-time move of the
  entries out of `/var/lib/geecs-data-portal`). Until now it was a router
  inside the Data Portal's process at `/log` — the portal's `MemoryMax=`
  is meant to kill that process when it runs away, and the write path
  for what people wrote should not be in it.
- `app.create_app(experiment, *, base_directory, notes_db, templates_dir,
  root_path)` replaces `router.create_log_router`; the routes serve at
  the app's root (`/day/…`, `/month/…`, `/api/…`), with `/health`
  (version, experiment, writable — never touches the share) for the
  fleet probe.
- The shared web glue comes from `geecs_web_theme.web` (the forwarded-
  prefix middleware, the `/theme` mount, the templates factory that puts
  `root` in every context); the templates drop their own `root`
  computation and address the page's assets through a named `/static`
  mount instead of the `_static` route. `geecs-web-theme` (`web` extra)
  and `uvicorn` are runtime dependencies now.

## [0.9.0] - 2026-09-12

### Added

- `render.summarize()` — the one-line stand-in a collapsed entry shows,
  beside the markdown parser it uses. Derived, never asked for: a title
  field would make the writer name a thing before typing it and would be
  empty for every entry already written.
- **Every entry is collapsible, in both books** — the scan log and the ops
  book. `<details>` was added for scan blocks and never generalised, so the
  scan log could fold a scan but not a note, and the ops book could fold
  nothing at all. Open by default (the notes are what you came to read);
  Collapse All shuts them. A shut entry shows a one-line summary derived from its body.
- **The ops book has a Collapse All button**, which it never had — so
  carrying this across meant building the control, not porting one. It
  shares the scan log's stored preference: how dense a logbook reads is one
  preference, not two.
- **One way to add a note, everywhere.** Each anchor — a scan, a gap, the
  day — renders its entries, then a `+ note` affordance with the composer
  folded behind it. Close or `Esc` folds it away and the affordance returns;
  nothing typed is lost, because the form stays in the DOM.
- The **density control** is on both pages. It had been built, tested and
  pinned across three files, and wired into no template at all — reachable
  from nowhere since it was written.

### Fixed (from adversarial review, before merge)

- **The ops book's filter blanked the whole page on any keystroke.** Its
  handler still selected `article.entry`; an entry became a `<details>` in
  this release. So it hid no entry, every day group then saw zero visible
  entries and hid *itself*. That is the same failure this release deletes
  the scan filter for, reintroduced in the other book by the same change.
- The ops book's **search box lost its styling** — `.searchbox` went out
  with the scan filter, leaving a bare label, a chromeless input and the
  magnifier on its own line. Its filter reads note *bodies*, which is why
  it survived; the CSS should have too.
- **Edit did nothing on a collapsed entry.** The tools moved into the
  `<summary>` and the toggle is suppressed there on purpose, so the edit
  form was inserted into a subtree the browser does not render — and a
  second click returned early on a form it could not show.
- **A failed Delete threw instead of showing the error.** The tools are no
  longer inside `.entry-main`, so `closest(".entry-main")` returned null:
  deleting an entry someone else already deleted produced a silent
  TypeError rather than "no such entry".
- **Spurious 409s against your own edit.** `closest()` includes the element
  it starts from, and an in-place edit form carries its own `data-entry` —
  so the version was written back to the form, never to the entry, and the
  next Edit read a stale one.
- The summary **mangled lab notation**: a regex stripping `` ` * _ ~ ``
  turned `~20 mJ, jitter ~3%` into `20 mJ, jitter 3%`. A single `~` is not
  markdown and a single `*` is not emphasis. It now reads markdown-it's
  token stream instead of guessing, so emphasis loses its markers and
  arithmetic keeps its characters.
- **Close silently published an autosaved entry.** Attaching a file saves a
  real entry immediately; folding the composer away took its Discard button
  with it, so the affordance returned, the author concluded nothing was
  written, and the note was in the log. Close and `Esc` now refuse while the
  entry is in the store, and name both ways out — refusing without an exit
  is the Discard bug over again.
- The summary returned a **table's first cell** (`"Parameter"`, `"Date"` —
  the toolbar's own table skeleton, and a pasted spreadsheet), which is
  content-free while looking like a summary; and **nothing at all** for a
  note that is one pasted screenshot, the commonest attachment shape here.
  A test pinning the table case existed and was dropped when the function
  moved packages.
- **The refusal message rendered into the 26px avatar column.** A composer
  is a two-column grid and `fail()` appends to the form, so the message
  landed under the avatar — pre-existing, but load-bearing the moment Close
  started refusing, since that message is the only thing separating a
  refusal from a button that appears to do nothing.
- The summary returned **nothing** for a table followed immediately by
  prose — the shape the toolbar's table button and the spreadsheet paste
  both produce, since they leave the cursor one newline below the block and
  markdown-it absorbs that line as another row. It now falls back to the
  first cell: thin, but true.
- Guard holes, each found by mutation: the `<details>`-display rule matched
  only the bare `.entry{…}` form (missing `.panel.scan`, `#logbook .entry`,
  `:not([open])`, descendant and `@media` shapes) and read only one
  stylesheet; the variant-ordering rule never blanked Jinja, so the very
  pairs it was rewritten for — `.entry-agent`/`.entry` and
  `.avatar-agent`/`.avatar` — were invisible to it. It also recorded each
  selector's **first** declaration while the cascade is decided by its
  **last**, and eight selectors in this sheet are declared twice — so a
  variant placed *between* two copies of its base was still overridden and
  still passed. And a statement at-rule (`@import …;`, `@layer base;`)
  erased every rule between it and the next `{`, because the prelude match
  was not bounded at `;`.

### Removed, after three review rounds

- **Two hand-rolled CSS scanners** — the `<details>`-display guard and the
  single-class-variant guard. Six regex CSS scanners were written this
  session and every one had a silent coverage gap: a consumed anchor
  (three times), a filter testing the empty string, a match starting
  inside a Jinja comment, and a first-declaration lookup where the cascade
  reads the last — the last of which made a mutation test pass for the
  wrong reason. Across three adversarial rounds the shipped surface
  converged (7 defects → 2 → 2) while the guards stayed flat at 2 per
  round, each one a hole in the guard written to close the previous hole.

  What survived untouched is the guard that **does not parse anything
  itself**: `test_inline_scripts_parse` shells out to `node --check`. So
  does the rule — the same one this release already found on the Python
  side, where `summarize()` stopped generating findings the moment it
  read markdown-it's token stream instead of stripping characters. Ask a
  real parser. For CSS that means `tinycss2` in a follow-up, not more
  regex here.

  Neither invariant is lost: `.entry{display:grid}` is fixed and confirmed
  on the deployed branch, and `.tag-retired` was a single historical bug.

### Known, deliberately not fixed here

- `summarize()` is a **second full markdown parse** per entry: 0.73 ms
  against `render_markdown`'s 1.18 ms on a 12.6 KB body, so ~60% more
  markdown work per entry and ~145 ms on a 200-entry month page. One
  `_md.parse()` feeding both a render and a summarize removes it — a change
  to `render_markdown`'s shape, which does not belong in a PR that has
  already grown twice.

### Removed

- **The scan filter.** It matched on scan label, parameter, purpose and
  devices, so typing anything from a *note* hid the whole page, and it
  caused three regressions in one day. Ctrl-F searches note bodies too,
  which is what a reader actually wants. Gone with it: the search box, the
  handler, the `data-hay` plumbing on scans, `data-filtered`, and the
  `scan_hay` macro.

  The **ops book's filter stays** — it searches note *bodies*, which is why
  it works and the scan one did not. They were never the same feature.

### Fixed

- **A composer that could not be closed.** A gap with no notes hid its
  composer behind "+ note"; a gap that already had notes rendered one
  **permanently** — twenty-two open text boxes on a busy day, with no
  affordance and no host, so the Close button added for the first case
  correctly did nothing in the second. The asymmetry is gone rather than
  patched, which also collapsed `between_block` to two lines.
- **Notes were uncollapsible**: `scanlog.css` carried `.entry{display:grid}`
  from when an entry was an `<article>` laid out as avatar-plus-body.
  Setting `display` on a `<details>` makes the browser lay out every child
  regardless of `open` — no error, it simply stops being a disclosure.
  There were two `.entry` rules in that file and only the second was
  updated.
- **The page's inline script was a syntax error.** Deleting the filter left
  its closing `});` behind, so the browser refused to execute the block and
  Collapse All *and* the rail jump died together. 217 tests were green.

### Changed

- New guards, each verified by mutation: every inline `<script>` in every
  template parses (`node --check`, Jinja blanked); no `<details>` is given a
  `display`; and the single-class-variant ordering rule now only considers
  classes that actually land on the **same element**, after flagging
  `.entry-body` as a variant of `.entry` — a guard that cries wolf gets
  switched off.

## [0.8.0] - 2026-09-12

### Fixed (from adversarial review, before merge)

- **The ops book's filter blanked the whole page on any keystroke.** Its
  handler still selected `article.entry`; an entry became a `<details>` in
  this release. So it hid no entry, every day group then saw zero visible
  entries and hid *itself*. That is the same failure this release deletes
  the scan filter for, reintroduced in the other book by the same change.
- The ops book's **search box lost its styling** — `.searchbox` went out
  with the scan filter, leaving a bare label, a chromeless input and the
  magnifier on its own line. Its filter reads note *bodies*, which is why
  it survived; the CSS should have too.
- **Edit did nothing on a collapsed entry.** The tools moved into the
  `<summary>` and the toggle is suppressed there on purpose, so the edit
  form was inserted into a subtree the browser does not render — and a
  second click returned early on a form it could not show.
- **A failed Delete threw instead of showing the error.** The tools are no
  longer inside `.entry-main`, so `closest(".entry-main")` returned null:
  deleting an entry someone else already deleted produced a silent
  TypeError rather than "no such entry".
- **Spurious 409s against your own edit.** `closest()` includes the element
  it starts from, and an in-place edit form carries its own `data-entry` —
  so the version was written back to the form, never to the entry, and the
  next Edit read a stale one.
- The summary **mangled lab notation**: a regex stripping `` ` * _ ~ ``
  turned `~20 mJ, jitter ~3%` into `20 mJ, jitter 3%`. A single `~` is not
  markdown and a single `*` is not emphasis. It now reads markdown-it's
  token stream instead of guessing, so emphasis loses its markers and
  arithmetic keeps its characters.
- Guard holes, each found by mutation: the `<details>`-display rule matched
  only the bare `.entry{…}` form (missing `.panel.scan`, `#logbook .entry`,
  `:not([open])`, descendant and `@media` shapes) and read only one
  stylesheet; the variant-ordering rule never blanked Jinja, so the very
  pairs it was rewritten for — `.entry-agent`/`.entry` and
  `.avatar-agent`/`.avatar` — were invisible to it.

### Removed

- **The `Campaign` concept, entirely.** It grouped consecutive scans that
  shared a `Scan Parameter` and a `ScanStartInfo`, and rendered them in a
  second collapsible above the scans themselves. Two things were wrong with
  it. It **inferred** — nothing in `ScanInfo` says those scans belong
  together — which is the one rule this package already holds itself to
  ("Status is reported, not inferred": *report what the files say and do
  not guess*). And it borrowed a word the lab uses for something else: a
  campaign here is weeks of work, so a day page announcing "Campaigns · 15"
  was telling an operator it held fifteen multi-week efforts.

  It also did not pay. Measured across four real days, grouping collapsed
  108 scans into 11 on a sweep day — but on an acceptance run of 21 scans
  it produced **15 groups, 11 of them wrapping a single scan**, because the
  threshold asked "is the day long?" when the question was "does grouping
  help?". Gone: the model, the property, seven CSS classes, the grouped
  template branch, and the filter and jump logic that had to reach two
  nesting levels.

  What survives is the honest part: **a long day opens collapsed**, which
  is a fact about volume, not a claim about meaning.

### Changed

- **A note between two scans is just a note.** It had worn a
  `<details class="between">` announcing "Between scans" and the range it
  fell in; that was ceremony, and it made identical content look like a
  different kind of thing from the same note in the ops book. Now it
  renders as an ordinary entry at document level, carrying its own
  timestamp — usually out of step with the scans either side, which is the
  point. The day reads as what happened, in the order it happened.
- Notes between scans render **between** scan blocks rather than inside the
  following one. Under the old grouping a note "after Scan005" was emitted
  inside Scan006's group.

### Added

- `scripts/seed_demo_notes.py` — worked-example entries for a **separate**
  database. The store is authoritative (what people wrote exists nowhere
  else), so invented content must never be seeded into it; but a day with
  no notes shows none of what the page is for, which is how an unstyled
  composer shipped unnoticed. Entries say in their own body that they are
  seeded, and the script refuses to write to a file that already exists —
  or, on import, to a database that already holds entries for the day.

  One entry is anchored to the day rather than a scan, because it is the
  only shape that renders whether or not the share is reachable: an anchor
  naming a scan the day folder does not contain is stored and counted in
  "Notes N" but never drawn, so seeding on a checkout with no share
  mounted otherwise produced exactly the empty page the script exists to
  prevent.

### Fixed

- **A dangling selector left `.insert[hidden]` visible.** Removing
  `.between[hidden]` from `.insert[hidden],.between[hidden]{display:none}`
  took the declaration block with it, so `.insert[hidden],` merged into the
  *next* rule and inherited `display:flex`. Clicking "+ note after …" hid
  nothing: the row stayed on screen above the composer it had just opened,
  mis-spaced, accumulating until a save reloaded the page. There is no
  global `[hidden]{display:none}` to fall back on.
- **The filter left notes between scans floating unlabelled.** Deleting the
  wrapper removed the only thing that said which gap a note sat in, and the
  `#q` handler hid scans only — so filtering to one scan left a note from
  a different gap sitting directly above it, reading as commentary on it.
  Notes and insert rows now carry the bracketing scan labels as a
  haystack and hide in the same pass.
- `editor.js` no longer sets `.open` on the reveal target, which stopped
  being a `<details>` in this release.
- **The filter and the editor were writing the same property.** `editor.js`
  uses `hidden` on an insert row to mean "this one has been used"; the new
  filter wrote `hidden` too, so clearing the box un-hid every used row and
  put the affordance back above the composer it had just opened. Filtering
  moved to its own `data-filtered` channel.
- **Filtering by a scan parameter hid every note on the page.** The
  bracketing-label haystack held only two labels, while a scan's held its
  parameter, purpose and devices — so typing the string the rail prints in
  every row left all the scans and hid all the notes, and nothing brought
  them back. Notes now borrow the haystacks of the scans that bracket them.
- Dead after the deletion: `.scanrow .count` (the grouped rail row was its
  only emitter), and the package `CLAUDE.md`'s "campaign shaping" and
  "curated campaign record", which named the concept this release removes.

## [0.7.0] - 2026-09-12

### Changed

- **The logbook adopts the surface kit.** `<body class="kit">` plus
  `kit.css`, and the page's own copies of the shell, the topbar, the rail,
  buttons, the selectable lists, the card and the status chips are
  deleted rather than overridden — `scanlog.css` goes from 359 to 311
  lines. The rail is the kit's 216 px with one breakpoint at 900 px,
  replacing the logbook's 228 px and 860 px; scan blocks are `.panel`;
  the day list and the scan list are both `.picklist`.
- **One status vocabulary.** The nine `chip-*` / `dot-*` classes are gone.
  `KIT_STATE` in `models.py` maps each `ScanStatus` onto a kit state, and
  the chip keeps its own word. The mapping is deliberately not the
  identity: `incomplete` (an empty `ScanEndInfo`, the most common state on
  the real share) stays neutral, because painting it amber painted most of
  a day amber; `unknown` (something was written and we cannot read it) is
  the case that earns amber. Colour carries severity, text carries which.
- A **tag is no longer a `.chip`.** It was borrowing the status chip,
  whose leading dot means *state*, which a tag is not; it gets `.tag`.
- The active day in the rail is marked with `aria-current="page"` rather
  than an `is-active` class — the kit keys on the accessibility attribute,
  which also closes a screen-reader gap the class never covered.

### Fixed

- **Five `url_for(...)` calls had no `.path`**, including both `editor.js`
  script tags. Starlette returns an ABSOLUTE url built from the request
  the app saw, so behind TLS termination that is `http://` — a
  mixed-content block for a `<script src>`, meaning the editor's script
  silently never loads and the composer stops working, with every test
  green. Both templates have carried a comment saying to use `.path`
  since they were written. Pinned by `tests/test_templates.py`.

### Added

- `geecs-web-theme` as a dependency (it has none of its own, so the edge
  is one-way): the logbook needs its status vocabulary to map onto, and a
  host mounting this package standalone can now serve the theme from here
  rather than relying on the portal's mount.
- `tests/test_models.py` pins `KIT_STATE` — every `ScanStatus` mapped,
  every target a real kit state, the colour each status had is the colour
  it keeps, and no failure ever reading as success.

### Fixed (from adversarial review, before merge)

- **`month.html` had not been converted at all.** Its `.card` composer and
  every day group were left orphaned by the CSS deletion — no background,
  border, radius or shadow — and their own overrides had been renamed to
  `.panel.*`, matching nothing. A `git checkout` I used to revert a test
  mutation had silently discarded the file's edits, and no test asserts the
  month page's container class.
- **The Save button in every composer** rendered as a plain neutral button:
  `_entries.html` and `editor.js` emit `btn btn-sm btn-primary`, and the
  three-class form was not covered by the rename. `editor.js` had not been
  touched at all.
- **`incomplete` and `unknown` had their severities swapped.** This package's
  CLAUDE.md records that empty `ScanEndInfo` is the most common state on the
  real share (37 of 49 across four sampled days) and that painting it amber
  "painted most of a day amber" — yet `incomplete` was mapped to `degraded`,
  while `unknown`, the genuinely unreadable case, went neutral. Both are now
  the colour they had, and the test pins the colours rather than merely
  asserting they are not `ok`.
- The "today" chip no longer claims `running`, which in the kit carries a
  permanent pulse — and that pulse had **no reduced-motion escape at all**
  (GeecsWebTheme 0.2.1 adds one; the logbook's own reduced-motion rule only
  kills `transition`).
- A retired template name is no longer a `.chip`: it is not a status, and
  the kit's chip leads with a dot that means *state* — the same rule this
  PR applied to tags.
- `.rail section + section` outlived the shell it belonged to and was
  double-spacing rail sections against the kit's `gap`.
- Dead markup dropped: `class="wrap shell"`, and the `daylist` / `scanlist`
  hooks that existed only for the deleted rule.
- **`.tag-retired` was declared above `.tag`**, so the base won on source
  order and the retired-template name rendered accent-coloured — pixel
  identical to the real tag beside it, which is the exact confusion the
  class was added to remove. Caught by re-review; a markup assertion cannot
  see it, so `tests/test_templates.py` now fails when any single-class
  variant is declared before the rule it varies.
- Three templates write `data-state` literally rather than through
  `KIT_STATE`; a typo there renders an uncoloured chip that looks plausible
  and passes any markup test. Every literal is now pinned against
  `geecs_web_theme.STATES`.
- `geecs-web-theme` moved to **dev** dependencies — no module under
  `geecs_logbook/` imports it, and the templates reach the theme through the
  host's mount. The root `CLAUDE.md` dependency graph, which claims to be
  verified against each `pyproject.toml`, now names this edge and the
  pre-existing `GEECS-Schemas` one it had also been missing.

## [0.6.0] - 2026-09-12

Navigation polish, and the synchroniser's feed.

### Added

- **A calendar in the rail** of both pages (`static/nav.js`, a
  `<details class="cal">` drawn on open): a month grid whose days are
  marked when they have notes (the store's per-day counts) and when a
  day folder exists on the share. The marks come from a new
  `GET /log/api/month/{YYYY-MM}/days`, fetched lazily by the open
  calendar — the month page itself still never touches the share, and
  the share side is **one** listing of the month folder
  (`scan_reader.days_with_folders`), never a walk of the days. When the
  experiment directory is missing the payload says `share: false` and
  the store's marks stand alone.
- **Keyboard stepping**: `←` / `→` go to the previous / next day (day
  page) or month (month page), `t` to today, `c` toggles the calendar;
  ignored while typing. The pages hand the targets to the script as
  `data-prev` / `data-next` / `data-today` on `<main>`.
- **Hover prefetch**: resting on a day or month link adds a
  `<link rel="prefetch">` for it, so the click that follows is served
  from the browser's cache. Only links the viewer is about to follow —
  nothing is fetched speculatively on load.
- **Today, by name**: `GET /log/today` (the day page) and
  `GET /log/month/today` (this month, at today's day group). The month
  rail's Today control is now always present, not only as a way back
  from another month.
- **The change feed** — `GET /log/api/entries?since=<aware ISO 8601>`:
  every entry whose `updated_at` moved after `since`, oldest change
  first, **tombstones included** (the one listing that returns them —
  a delete is a change a downstream copy must learn). `until=`,
  `book=`, `include_deleted=false` narrow it; `limit=` pages it with a
  `next_cursor` that resumes after the last row even when rows share an
  `updated_at`. `NotesStore.changed_since` and `NotesStore.count_by_day`
  are the store methods behind the feed and the calendar.

### Changed

- `NotesStore.create` takes its timestamp under the write lock, as every
  other writer already did, so two concurrent creates cannot commit out
  of stamp order and slip past a synchroniser's high-water mark (review
  of #844). The feed's cursor is opaque and URL-safe (a pasted, unencoded
  cursor no longer re-sends the boundary row) and a corrupted one is a
  422, not a quiet "caught up". `/api/month/{m}/days` turns a share I/O
  error into `share: false` and anything else into a 503, like the day
  page. Keyboard stepping is refused while a composer holds unsaved text;
  the prefetch dwell is 250 ms so a pass over the rail prefetches nothing.

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

### Fixed (review of #842)

- The month page no longer runs the mirror sync on its request thread —
  that put a share write on the one page whose promise is that the share
  is never on its path; the day page pays the mirror debt (pinned).
- An attachment autosave on the month composer pins the entry's day, so
  the date picker locks once the entry exists instead of a later change
  being silently ignored on Save; "+ note" respects the lock.
- A template file named `blank`, `scan_note` or `day_intro` is refused
  (`RESERVED_NAMES`): it would have put a chip on every hand-typed entry.
  The names the pages render quietly come from the same constant.
- `/log/month/9999-12` and `0001-01` are 400s, not 500s (their neighbour
  month cannot exist).
- The filter haystack (`data-hay`) is emitted only on the month page; the
  day page filters scans and was carrying a third copy of every body.
- `TEMPLATES_DIRNAME` lives in `seed_templates` and the portal imports it;
  `PageSeeds` is a model rather than a dict.
- After an attachment autosave the type buttons lock along with the date
  picker: an edit carries no `template`, so a type pressed after the
  entry existed would show on screen and not in the store (Codex review).
- An entry no longer shows a tag chip for a tag its type's prefill
  carries ("Laser" beside "#laser" said one thing twice — owner's
  hand-test). Tags the author added, or typed with no type, keep theirs.

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
