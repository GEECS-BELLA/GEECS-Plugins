# One app: GeecsPlotter folds into the Data Portal — the brief

For the sessions that do the merge and for Sam, who reviews it at the
promotion PR from `users/ajgonsalves` to `master`. This file is
scaffolding (`Planning/README.md`): the PR that deletes `GeecsPlotter/`
deletes it too, after moving what still binds into the portal's
`CLAUDE.md`.

## Goal

The Data Portal's Plot and Images tabs and GeecsPlotter do the same job.
Make the portal the one app: everything GeecsPlotter does, in the
GeecsWebTheme kit look, then delete GeecsPlotter.

## Rulings this reverses (the author's override, 2026-10-28)

The author (who wrote GEECSplotter in LabVIEW and built GeecsPlotter)
overrides three of Sam's rulings. Each PR in this arc names the ones it
touches, so Sam can confirm or send it back at the promotion PR.

1. **"Portal/plotter integration is a later effort"** and "the maintainer
   still wants a separate package" (`Planning/geecs_plotter.md`,
   `GeecsPlotter/CLAUDE.md`). Integration starts now.
2. **"Portal onto the kit is deliberately LAST"**, as the kit's acceptance
   test, once HTU is quiet (`GeecsWebTheme/CLAUDE.md`). The whole portal
   moves onto the kit now, in one PR, not tab by tab. It still grows the
   kit (tabs, toasts) through `kit.html`, as that ruling intended.
3. **"All view state is URL-carried; a link IS the analysis"**
   (`GEECS-DataPortal/CLAUDE.md`). View state moves into named settings
   sets, files only. A link carries the day, the scan and the set name
   (`set=`); it opens the set as it is now, so a set edited since draws a
   different plot. The author ruled that enough for sharing. Old links
   with `y=`, `x=`, `filters=`, `bincfg=`, `display=`, `gridcfg=` … open
   the scan but no longer reproduce the plot. That includes the links
   already stored in logbook entries by "send plot to log"
   (`logbook_send.py` relies on the URL restoring the analysis), and new
   entries link the live set, which may draw differently later. The
   author accepted both (2026-10-28) over freezing a copy per send.

## The author's answers (2026-10-28)

| Question | Answer |
|---|---|
| Start when? | Now, on `users/ajgonsalves`; Sam reviews at promotion |
| Where does view state live? | Named settings sets, files only |
| Kit adoption | The whole portal at once |
| Sharing a plot | `set=` in the URL is enough |
| A URL with no `set=` (the day list, the logbook's link, MCP-built links, old bookmarks) | Opens **your own set**: the page asks your name once (kept in the browser, like the logbook author box) and opens and saves into the set of that name, so one person's browsing never rewrites another's plots. `default` is just a name |
| Logbook links | The entry links the live set; no frozen copies |
| Images tab | The plotter's strip is the overview; clicking a tile opens the portal's large single-shot view (steppers, processing, traces) under it |
| How many Y lists? | Three (Y1–Y3), as GEECSplotter; the portal's fourth goes |
| Front page `/` | Today's scan list, as now. `/?set=<name>` opens that set's last scan |
| Settings folder | `{config.ini [Paths] geecs_data}/DataPortal/settings/`, else `<main checkout>/user data/DataPortal/settings/`; `--settings-dir` overrides. Existing sets in `…/GeecsPlotter/settings/` are copied over once (the copy never deletes or overwrites) |
| `--browser` | Opens a browser by default, as the plotter does; the service unit passes `--browser none` |
| GeecsPlotter | Kept until the portal does everything it does, then deleted in the last PR |

## What stays exactly as it is

- **Every `/api/...` route and its payload.** GEECS-MCP calls
  `/api/run/jump/{iso}`, `/api/run/{uid}/analysis`, `/api/analysis/catalog`,
  `/api/run/{uid}/binned`, `/api/run/{uid}/logbook` and
  `/run/{uid}/artifact`, and its contract tests mount the portal in
  process. OSPREY's htu-assistant skill (`skills/geecs-data-portal`,
  another repository) mirrors every payload. New routes are added
  (settings, image strip); none is changed or removed. If one must
  change, GEECS-MCP moves in the same PR and the skill is named in it.
  In particular `/frame` and `/binned` keep accepting **four** columns
  (`plot_api._MAX_Y_COLUMNS`; GEECS-MCP's `get_scan_scalars` sends an
  agent's list straight to `/binned`): the page offers three Y lists,
  and the figure code still draws a fourth series on its own axis for
  an API caller.
- The read-only charter and the scan-folder invariant. Settings writes
  become a fourth exception, written into the portal's `CLAUDE.md`: they
  go to the settings folder only, never the data share's scan tree.
- The ScanCatalog seam (Tiled and folder scans), the Overview, Grid and
  Analysis tabs, the config editor, the logbook send, the processing
  selector, traces, the cache policy and the prefix-agnostic URLs.
- **Deployment.** The portal runs on port 8200 on the worker host and
  others use it. Nothing deploys from `users/ajgonsalves`; each PR body
  says so. The `--browser` default changes the service unit; that file
  moves in the same PR, and the deploy itself waits for promotion.
- **Where the deployed portal keeps its sets.** The folder rule in the
  table is a laptop rule. The service unit says today "the portal writes
  nothing durable" and has no `StateDirectory`; a fallback into the
  deployed checkout could be unwritable or wiped by a redeploy. PR 3
  gives the unit `--settings-dir`, rendered from `site.env` (a facility
  value with one home), defaulting to a `StateDirectory`, rewrites that
  comment, and names it in `DEPLOYMENT.md` and `site.env.example`.

## The pull requests, in order

Each goes through `/land` against `users/ajgonsalves`: `/check`, the
portal's full suite with `--extras analysis`, GEECS-MCP's contract tests,
an adversarial review subagent, a browser check of every tab, and a merge
only on the author's yes.

1. **This brief.**
2. **The portal onto the kit.** `base.html`, `day.html`, `run.html` and
   `grid_script.html` on `body class="kit"`: the kit's top bar, rail,
   panels, status chips, pane states and overlay ladder (the destructive
   confirmation is already a `<dialog>`; the display popup and filter
   editor become a drawer). The inline style block goes; what the kit
   lacks (tabs, toasts, a plot frame) is added to `kit.css` with a
   specimen on `kit.html`, never as inline style. The page script moves
   to `static/portal.js` (no build chain: a plain file). The glue copy
   goes in the same change: `_ForwardedPrefixMiddleware`, `_clean_prefix`
   and the `root` processor are replaced by `geecs_web_theme.web`
   (`ForwardedPrefixMiddleware`, `mount_theme`, `make_templates`); the
   portal moves into `test_no_literal_colours.py`'s kit surfaces and the
   adoption table in `GeecsWebTheme/CLAUDE.md` says yes. Behaviour and
   URLs unchanged; the reverse-proxy tests keep passing.
3. **Settings sets replace the URL state.** `settings.py` moves from the
   plotter into the portal as `PortalSettings` (the plotter's fields plus
   the portal's: per-shot/binned, row filters, binning, display, the
   Grid's `gridcfg`/`gridbin`/`imagebin`, the image device and shot, the
   processing pick), `extra="allow"` kept so the plotter and an older
   portal still load a newer set. The rail gets the set picker, Save as,
   and the replace `<dialog>`. Every control saves into the open set;
   the page reads the set, not the query string. `/api/settings`,
   `/api/settings/{name}` added. `CLAUDE.md`'s URL-state rule is
   rewritten (ruling 3 above), with `logbook_send`'s docstring and tests,
   and the settings write added to the charter. A URL without `set=`
   opens the viewer's own set (the table above). The deployed settings
   folder (above). The one-time copy from `GeecsPlotter/settings/`.
4. **The XY plot.** The plotter's rail: X and Y1–Y3 pick lists, each
   with its colour swatch, filter box (double-click or Enter picks the
   one left), name | alias toggle and Fit (Gaussian, SuperGaussian,
   NoOffset twins, polynomial of an order); "Y plots 1 | 2 | 3"; X =
   scan variable (auto); the day and scan-number boxes as the scan
   stepper. `figures.py` takes the plotter's three-slot axis ladder
   (titled, colour-coded axes, `autoshift`) and the fits, replacing the
   portal's four-column ladder; `/api/run/{uid}/frame`'s payload is
   unchanged, the fit numbers come from a new `/api/run/{uid}/fits`.
   The portal's binned view, row filters, display settings, timestamps
   and send-to-log stay.
5. **The Images tab and `--browser`.** The plotter's strip (a tile per
   bin or per shot, grid or one row, tiles across, colour maps, a z scale
   per tile / shared / fixed, a per-camera ROI drawn in a drawer) as the
   overview; clicking a tile opens the portal's large view of that shot
   or bin under the strip, with its steppers, processing selector and
   trace view. The strip's routes join `routes/images.py` under
   `/api/run/{uid}/images/…`. `--browser` (default on; `none` in
   `deploy/geecs-data-portal.service`, `DEPLOYMENT.md` and the fleet
   map's row as needed).
6. **Delete GeecsPlotter.** The package, and every registration: root
   `CLAUDE.md` (map, graph, changelog list), `README.md`,
   `CONTRIBUTING.md`, `.importlinter`, `scripts/check.sh`,
   `scripts/ci_select.py`, `GeecsWebTheme`'s guards and adoption table,
   the doc-audit lists, the large-file `exclude` in
   `.pre-commit-config.yaml`, the GeecsPlotter comment in `.gitignore`
   (the `/user data/` rule itself stays: the portal's laptop fallback
   uses it), the "GeecsPlotter #1172" citation in the portal's
   `figures.py`, and `Planning/epics_estate.md`'s mentions
   (`.claude/launch.json` is untracked; each person edits their own
   copy). This brief and
   `Planning/geecs_plotter.md` go too; their live items (the still-open
   LabVIEW settings import, the fits and images still to come) move to
   the portal's `CLAUDE.md` or to issues.

## Open, for Sam

- The three reversed rulings above.
- **`Planning/epics_estate.md` (provisional, 2026-10-25) plans this
  merge differently:** its phase 4 moves portal, plotter and editor into
  a React `geecs-web` app, and its phase 5 deletes the kit layer this
  arc invests in (PR 2). Neither Planning doc binds. This one is what
  the author is building now; if Sam keeps the epics_estate route, PR 2's
  kit work is thrown away at its phase 5.
- Whether `PortalSettings` stays in the portal or moves to GEECS-Schemas
  (the plotter brief asked the same of `PlotterSettings`).
- Settings sets are shared files with no revisions: two people editing
  one set overwrite each other, as with LabVIEW's settings files. A
  deployed portal, used by more people than the plotter was, makes this
  more likely.
