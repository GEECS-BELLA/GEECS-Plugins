# GeecsLogbook — deployment

The logbook is its own service since 0.10.0: one process per experiment,
port **8400**, unit `geecs-logbook`, state in `/var/lib/geecs-logbook`.
Before that it was a router inside the Data Portal's process at `/log`
(portal ≤ 0.26, `--scan-log`); that mount is gone, and the portal now
only *links* to the logbook (`--logbook-url`). The split is deliberate:
the portal's `MemoryMax=` is meant to kill it when it runs away, and the
write path for what people wrote should not be in that process.

The fleet-map page (`docs/platform/fleet_map.md`) carries the service's
row — host, port, health check — and must be updated in the same PR when
this deployment moves or changes.

## What it serves

- `/day/YYYY-MM-DD` — the **scans book**: a day document over scan
  *folders*. Whatever `ScanNNN` directories exist for that date, each
  `ScanInfoScanNNN.ini` read at request time. There is no daily job and
  nothing to create; a scan appears because its folder does. `/today`
  names today.
- `/month/YYYY-MM` — the **ops book**: day-level notes read by month,
  from the database alone (it never touches the share, so it stays fast
  when the share is slow). `/month/today` names this month.
- `/api/…` — the same as JSON, plus the entry write verbs and
  `GET /api/entries?since=`, the change feed for synchronisers.
- `/health` — `{"ok", "version", "experiment", "writable"}`; it never
  touches the share, so a hung mount cannot make the service look down.
- `/theme/…` — the shared GEECS look, mounted by this process (the kit's
  reference page at `/theme/kit.html`).

The scan *record* is rendered from the folders and stored nowhere. What
people **write** — notes on a scan, between scans, or about the day;
agent drafts; pasted screenshots — goes to the SQLite file named by
`--notes-db` and an `attachments/` directory beside it, and each entry is
mirrored as a markdown file (plus its attachments) into
`{experiment}/logbook/Y2026/09-Sep/26_0911/…` on the share — a tree the
logbook owns, with the data tree's date shape but outside it. It never
enters `scans/` and never creates a scan folder (pinned in
`tests/test_mirror.py` and
`tests/test_scan_reader.py::TestScanFolderCreationInvariant`). The
database is written first, so a save never fails because the share is
slow or unmounted; the files follow when they can (a sync runs on day
views, at most once a minute). Deleting an entry leaves a tombstone row,
keeps its history, and removes the mirrored file.

The store uses SQLite's JSON functions (`json_insert`), present in the
interpreter's bundled SQLite from 3.31 on — any Python 3.11 build, and
the system library on Ubuntu 22.04 or later.

## Install

The logbook rides the **portal's clone** (`<root>/portal-checkout`) with
its own poetry environment inside `GeecsLogbook/` — one clone for the two
web viewers. The consequence: a `git pull` there is a deploy of both;
restart both units after it (`fleet_status.sh` reports a running process
whose code on disk has moved on). `deploy/bootstrap_host.sh` does the
clone and the install; by hand:

```bash
cd <root>/portal-checkout
( . deploy/site_env_lib.sh; load_site_env /etc/geecs/site.env; cd GeecsLogbook && "$GEECS_POETRY" install )
```

`site.env` carries the absolute poetry path the units use
(`GEECS_POETRY`), read here through the repo's own `site.env` parser
(quotes and whitespace handled). On the reference host that is the only
form that works **as the service account**: `poetry` is not on its
`PATH` in a non-login shell, and its login shell (`bash -lc`) does not
add it either (observed as that account, 2026-09-13). The bare `poetry
install` in the unit header and the sibling runbooks assumes a shell
where it resolves; when yours does not, this is the fallback.

Render and install the unit from the host's `site.env`:

```bash
deploy/render_units.sh /etc/geecs/site.env ~/deploy-staging GeecsLogbook/deploy/geecs-logbook.service
sudo install -m 0644 ~/deploy-staging/geecs-logbook.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now geecs-logbook
curl -s http://localhost:8400/health
```

The unit takes `GEECS_EXPERIMENT` and `GEECS_CONFIGS_ROOT` from `site.env`
(the seed templates are `logbook_templates/*.md` at the top of the configs
checkout — see below) and optional flags from `GEECS_LOGBOOK_EXTRA_ARGS`.

### The entries live in the state directory

The unit sets `StateDirectory=geecs-logbook`: systemd creates
`/var/lib/geecs-logbook`, owns it to the service user, and the logbook
defaults `--notes-db` to `logbook.db` there — no path in `site.env`.
**That directory is everything irreplaceable** — the database, its
history, and every uploaded file; back it up as one unit (a nightly
WAL-aware copy of the database plus an rsync of `attachments/`). The
WAL-aware copy is `sqlite3 logbook.db ".backup …"` where the `sqlite3`
CLI exists — **the reference host does not have it**, and a missing
binary under `sudo` fails as one quiet line in a run of commands. The
form that always works is the interpreter every host has — run **as the
service user**, never as root: the store opens a connection per
operation and closes it, so between requests there is no `-shm` file,
and a root-run backup would create one owned by root that the next save
cannot open read-write. One-time setup of a destination the service user
owns, then the nightly line:

```bash
sudo install -d -o <service user> -g <service user> /var/backups/geecs-logbook
sudo -u <service user> python3 -c "import sqlite3; s=sqlite3.connect('/var/lib/geecs-logbook/logbook.db'); d=sqlite3.connect('/var/backups/geecs-logbook/logbook.db.$(date +%F)'); s.backup(d); d.close()"
```

The
markdown mirror on the share is the second copy of what people wrote,
legible without any of this running. Running the logbook by hand (no
systemd) gives a read-only logbook unless you pass `--notes-db`
explicitly; its directory must already exist.

Without a `--notes-db` the logbook is the read-only day view and no entry
route exists.

### Moving the entries from the portal's state directory (one-time)

A host that ran the logbook inside the portal (portal 0.22–0.26,
`--scan-log`) has the entries in `/var/lib/geecs-data-portal/`. They
move once, with both services stopped — the logbook too, if the Install
section above already started it on an empty state directory:

```bash
sudo systemctl stop geecs-data-portal
sudo systemctl stop geecs-logbook 2>/dev/null || true   # not yet installed is fine
# back up first — the database and the uploads are the only copies on the host.
# Outside both state directories, so the glob below cannot sweep it along.
# python, not the sqlite3 CLI: the CLI is not installed on the reference host
# and `sudo sqlite3 …` then fails quietly while the mv below still runs.
# (root is fine HERE: both services are stopped and chown -R follows below)
sudo python3 -c "import sqlite3; s=sqlite3.connect('/var/lib/geecs-data-portal/logbook.db'); d=sqlite3.connect('/var/backups/logbook.db.pre-split'); s.backup(d); d.close(); print('backup written;', s.execute('select count(*) from entries').fetchone()[0], 'rows in the source')"
sudo cp -a /var/lib/geecs-data-portal/attachments /var/backups/logbook-attachments.pre-split
ls -la /var/backups/logbook*                    # BOTH exist before going on
sudo install -d -o <service user> -g <service user> /var/lib/geecs-logbook
# logbook.db* — the glob matters: the store runs SQLite in WAL mode, and a
# portal that last died by MemoryMax= (the case this split exists for) has
# committed entries only in logbook.db-wal until the next checkpoint. The
# .backup above is WAL-aware; a bare mv of logbook.db is not.
sudo mv /var/lib/geecs-data-portal/logbook.db* /var/lib/geecs-logbook/
sudo mv /var/lib/geecs-data-portal/attachments /var/lib/geecs-logbook/
sudo chown -R <service user>: /var/lib/geecs-logbook
sudo systemctl enable --now geecs-logbook      # StateDirectory= adopts the existing directory
curl -s http://localhost:8400/health            # "writable": true
# the count that proves the move: the change feed lists every row, tombstones
# included, in pages of 500 — it must equal the source count printed above,
# with next_cursor null (else page on). A day whose entries were all deleted
# shows NONE on its page — by design — so the page is not the check.
curl -s 'http://localhost:8400/api/entries?since=2000-01-01T00:00:00%2B00:00' | python3 -c 'import json,sys; d=json.load(sys.stdin); print(len(d["entries"]), "rows on the new port; next_cursor =", d["next_cursor"])'
# in a browser: /day/<a day with live entries> shows them; an attachment link serves
sudo systemctl start geecs-data-portal          # on its re-rendered unit (no StateDirectory)
```

Done on the reference host 2026-09-13 (portal 0.27.1, logbook 0.10.1):
6 rows in the source, 6 on the new port, all of them tombstones, so the
day page was rightly empty and the feed count was the check that
mattered.

The backups in `/var/backups/` and the now-empty
`/var/lib/geecs-data-portal` (`sudo rmdir` — it refuses if anything is
still there) can go once the entries are seen on the new port. The mirror on the
share is untouched by the move — the sync on the next day view finds
nothing owed.

## The portal's link

The portal's run page links each scan to its card in the logbook
(`<logbook>/day/YYYY-MM-DD#ScanNNN`). It builds the link from its
`--logbook-url` — the logbook's base URL, set in `site.env` through
`GEECS_PORTAL_EXTRA_ARGS` (`--logbook-url http://<host>:8400` until the
fleet's front door routes `/log` to this service, then `--logbook-url
/log`). Without the flag the portal shows no link.

## Type buttons (seed templates)

Every composer's type buttons are seed templates: `*.md` files in
`logbook_templates/` at the top of the configs checkout, named by the
unit's `--templates-dir "${GEECS_CONFIGS_ROOT}/logbook_templates"`. Copy
`examples/logbook_templates/` there to start (commit it to the configs
repo); each file's header names its label, colour (a theme token name),
book and order, and its body is the prefill. Adding a file adds a button
within a minute, no restart. Without the directory the composers are
plain.

## Behind a reverse proxy

Two proxy shapes, one flag each — they are not interchangeable:

- **A prefix-stripping proxy** (Caddy `handle_path /log/*`, nginx
  `location /log/ { proxy_pass http://…:8400/; }`) forwards `/day/…` and
  must send `X-Forwarded-Prefix: /log`. The middleware from
  `geecs_web_theme.web` adopts the header per request, and every link,
  form, fetch base and asset URL on the page carries `/log`. No flag.
- **A prefix-preserving proxy** (one that forwards `/log/day/…` as is and
  sends no header) needs `--root-path /log`: the page then links
  `/log/static/…` and `/log/theme/…`, and the **mounted assets** answer at
  the prefixed path **only** — `/static/nav.js` on the upstream port is a
  404 with the flag set. The page routes (`/day/…`, `/health`) answer
  either way, which is why the flag is not a fallback for a stripping
  proxy that omits the header: that combination serves the HTML and
  loses every stylesheet and script, rather than failing loudly.

When the header is present it wins over the flag. Nothing in the page is
absolute.

## Health and troubleshooting

| Symptom | Look at |
|---|---|
| `/health` answers, `"writable": false` under systemd | the unit lost `StateDirectory=` (re-render) or `GEECS_LOGBOOK_EXTRA_ARGS` passes a `--notes-db` whose directory does not exist |
| A day page 503s "data share unavailable" | the share is not mounted at the service account's `config.ini` path; the month page keeps working (store only) |
| Composers have no type buttons | `logbook_templates/` missing at the top of `GEECS_CONFIGS_ROOT`, or the unit's `--templates-dir` points elsewhere |
| Entries save but no markdown appears on the share | the mirror is deferred (share slow/unmounted); `journalctl -u geecs-logbook` shows "mirror deferred"; the next day view retries |
| The portal's run page has no "log" link | its `--logbook-url` is unset (`GEECS_PORTAL_EXTRA_ARGS`) or the run belongs to another experiment |

Reading a day of ~100 scans off a VPN-mounted share takes a few seconds
cold and milliseconds thereafter — per-scan summaries are cached on the
ScanInfo file's own mtime and size. A slow first load on a cold share is
expected, not a fault.
