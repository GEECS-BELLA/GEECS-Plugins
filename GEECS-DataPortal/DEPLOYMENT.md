# GEECS Data Portal — deployment runbook

One service, one host: the portal runs on the worker host — the fleet
map (`docs/platform/fleet_map.md`) is the authority on which machine
that is at any given time. On the interim box it happens to share a
machine with Tiled and GEECS-MCP, so the ports read `:8000` Tiled,
`:8100` MCP, **`:8200` portal** — but only the portal's own port is
load-bearing; nothing below assumes Tiled is local. Anyone on the lab
network browses to `http://<host>:8200/`.

The portal is **read-only except explicit analysis runs** (see
`CLAUDE.md`): it renders the Tiled catalog and reads per-shot files off
the data share, never creates anything on the scans path, and — only
when started with `--processing-configs` and the `analysis` extra —
runs a ScanAnalysis analyzer on a scan when a user clicks Run on the
Analysis tab. That run writes what ScanAnalysis writes (figures under
`analysis/ScanNNN/`, s-file columns, some analyzers' derived subfolders
inside the scan folder), so **where analysis runs are enabled the share
must be mounted read-write**; a read-only mount makes every run fail —
or, for analyzers that swallow write errors, finish `done` with missing
outputs. Without `--processing-configs` the portal never writes, and a
read-only mount is the right choice on a dedicated viewer host.

A run in flight cannot be interrupted: on `systemctl stop/restart` the
portal refuses new runs, logs the in-flight one, and the process exits
when that run finishes. systemd's default `TimeoutStopSec` (90 s)
would then SIGKILL a long analysis mid-write (s-file merge, HDF5) — set
`TimeoutStopSec=` in the unit to the longest analysis you expect, or
restart between runs.

## Prerequisites

- Ubuntu with **Python 3.11** and **Poetry** on the service account
  (same account and tooling as the other services on the box).
- `~/.config/geecs_python_api/config.ini` with a `[tiled]` section
  (`uri`, `api_key`) and a `[Paths]` section whose
  `geecs_data_local_base_path` points at the mounted data share —
  the same file every GEECS-Plugins package reads.
- The data share mounted at that path (e.g. `/mnt/<share>/data/`) —
  read-write if analysis runs are enabled (above).
- For the Images tab's processing selector and the Analysis tab's
  runs: `poetry install -E analysis` (ImageAnalysis + ScanAnalysis and
  their closure, incl. the Google client libs ScanAnalysis lists) and
  `--processing-configs <scan_analysis_configs tree>` on the command
  line. Omit both for a read-only viewer.
- Port **8200** free (`ss -tlnp | grep 8200`).

## Install

The portal runs from its own repo checkout so it can be upgraded
without touching the checkouts other services run from (the
queueserver-worker precedent). Give the checkout a portal-specific
name — on a box that also runs the CA gateway, a checkout named plain
`~/GEECS-Plugins` is likely the *gateway's* running checkout, and this
runbook's Upgrade step must never `git pull` that one. The site
profile fixes the name: `<root>/portal-checkout`, where `<root>` is
`GEECS_CHECKOUT_ROOT` from the host's `site.env`
(`docs/platform/site_profile.md`); paths below use that.

**Run every command in this section as the service account** (the
`User=` of the unit): Poetry keys the project venv under the invoking
user's cache, so an env installed by an admin account is invisible to
the service and the unit crash-loops on an empty env while admin-side
checks pass.

```bash
cd <root>/portal-checkout/GEECS-DataPortal
poetry env use python3.11
poetry install --extras analysis
```

The `analysis` extra installs ImageAnalysis for the Images tab's
**processing selector** (0.13.0+) and the Analysis tab. The feature
needs both the extra and the scan-analysis configs tree on the command
line; the rendered unit supplies the latter from the site profile —

```
--processing-configs "${GEECS_CONFIGS_ROOT}/scan_analysis_configs"
```

(`GEECS_CONFIGS_ROOT` is the configs repo on the data share, quoted
because the lab's share paths contain spaces; systemd substitutes the
quoted `${VAR}` as one argument). The argument is unconditional in the
rendered unit and the bootstrap always installs the extra: there is no
raw-images-only configuration to maintain by hand (a rendered unit is
never edited — a re-render is part of every deploy). A missing or
misconfigured tree logs a startup WARNING naming the path and the
selector hides itself; nothing else changes.

Smoke-test in the foreground before installing the unit:

```bash
poetry run geecs-data-portal --experiment Undulator
# in another shell:
curl -s http://localhost:8200/health
```

`/health` returns the catalog probe — `ok` requires the Tiled server
reachable with the configured key. Then load a real day page in a
browser and open one run's image gallery (exercises the share mount).

## systemd unit

`deploy/geecs-data-portal.service` is a **template**: render it from the
host's `site.env` with `deploy/render_units.sh` (or let
`deploy/bootstrap_host.sh` do the whole host), then install the rendered
unit and `enable --now` it — see the
[Site Profile](../docs/platform/site_profile.md). The account, checkout
root, poetry path, experiment, timezone, and the memory ceiling all come
from `site.env`; nothing site-specific is typed into the unit by hand.

**Memory ceiling.** The rendered unit carries `MemoryHigh=` and
`MemoryMax=` from `GEECS_PORTAL_MEMORY_HIGH` / `GEECS_PORTAL_MEMORY_MAX`
(both required; 3G / 4G in the example for a 16 GB host shared with
Tiled, MySQL and the queueserver). Above HIGH systemd throttles and
reclaims the portal; at MAX it kills it and `Restart=on-failure` brings
it back — so a runaway portal evicts itself before the kernel picks a
victim, and the victim is the portal rather than Tiled (#834). To change
the numbers: edit `site.env`, re-render, `daemon-reload`, restart. For a
quick change without a re-render, `sudo systemctl set-property
geecs-data-portal MemoryMax=6G` writes a persistent drop-in;
`systemctl revert geecs-data-portal` removes it. `systemctl status`
shows the current `Memory:` line against the cap.

Verify:

```bash
systemctl status geecs-data-portal
curl -s http://localhost:8200/health
journalctl -u geecs-data-portal -n 20
```

## Behind a reverse proxy (OSPREY panels, etc.)

The portal is prefix-agnostic: mount it under any path and every link,
form, image, `/api` fetch, and redirect carries the prefix. Configure
the proxy to **strip the prefix and name it** in `X-Forwarded-Prefix`
(the Grafana/JupyterHub convention) — nginx example:

```
location /portal/ {
    proxy_pass http://<worker-host>:8200/;
    proxy_set_header X-Forwarded-Prefix /portal;
}
```

The header is per-request and needs no portal-side config; any mount
name works, including ones that collide with portal route heads
(`/run`, `/api`, …). Malformed header values (not root-absolute, `//`,
whitespace, query/fragment characters) are ignored rather than
propagated into page links.

Two proxy shapes, one setting each — they are **not interchangeable**
(the rule lives in `geecs_web_theme.web`'s middleware docstring):

- **Prefix-stripping** (the nginx block above; Caddy `handle_path`):
  forwards `/run/…` and sends the header. No flag.
- **Prefix-preserving** (forwards `/portal/run/…` as is, sends no
  header): start the service with `geecs-data-portal --root-path
  /portal`. The page then links `/portal/static/…`, and the mounted
  assets (`/static`, `/theme`) answer at the prefixed path **only** —
  plain routes answer either way, which is why the wrong pairing fails
  quietly: a stripping proxy without the header serves the HTML and
  loses every stylesheet and script. In this mode also avoid mount names
  that collide with a portal route head, and know that trailing-slash
  redirects drop the prefix (Starlette builds them from the un-prefixed
  path; the header mode re-prefixes the path and has neither
  limitation).

The header, when present, wins over the flag. A proxy that does **not**
strip the prefix must not send it (the symptom of that misconfiguration
is loud — every page 404s).

For a panel health LED, probe `GET /health` — 200 always (the JSON
`ok` field reports the catalog probe, so a down Tiled shows as a
degraded catalog, not a dead portal).

## Upgrade

```bash
cd <root>/portal-checkout && git pull      # the portal's clone only — never another service's
cd GEECS-DataPortal && poetry install --extras analysis
sudo systemctl restart geecs-data-portal
```

(Drop `--extras analysis` only on a deployment that deliberately runs
without the processing selector.)

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `/health` reports a catalog error | Tiled down, or `[tiled]` uri/api_key wrong — `curl http://<tiled-host>:8000/api/v1/` |
| Day pages load, images 404 | share not mounted (or moved) at `geecs_data_local_base_path`; a 404 on one shot with others fine is the exact-match rule working (that device missed the shot) |
| Slow day listings | measure `list_runs` against the catalog first — the fix is a portal-side cache, not a schema change |
| Unit crash-loops at start | `status` shows **217/USER** — the installed unit is a pre-profile file (or a copy from an old staging run) with the generic `User=`; the clone it came from predates the templated units — pull it forward, re-render with `deploy/render_units.sh`, reinstall (site profile page). Wrong absolute Poetry path in `ExecStart` (`status` shows 203/EXEC); env installed by a different account than `User=` (empty venv — reinstall as the service account); or port 8200 already taken. A down Tiled does **not** exit the service — that shows up as the `/health` row above |
| Plot tab: **Copy plot to clipboard downloads a PNG instead** | The portal is served over plain `http://`, and every browser gates clipboard *image* writes on a secure context, so `navigator.clipboard` does not exist on the page — the button says so in its tooltip and its note, and degrades to the 2× download. No page-side workaround exists (the legacy `execCommand` copy carries text only). To get a real bitmap copy, reach the portal from a secure origin — see **Clipboard copy and the secure-context rule** below |
| Evening scans 404 (or resolve oddly) while daytime scans work | host timezone differs from the scanner hosts' — daily folders are named by the scanner's local date. `site.env` sets `TZ`; keep it matching the lab's zone |

The fleet-map page (`docs/platform/fleet_map.md`) carries the
service's row — host, port, health check — and must be updated in the
same PR when this deployment moves or changes.

## Getting a plot out of the Plot tab

Two buttons on the plot's modebar, and they are not interchangeable.

**Send plot to this scan's log entry** is the one that works on the
deployed portal. The browser hands the PNG to the portal and the portal
calls the logbook's own API server-to-server, so nothing depends on the
page's origin and the logbook needs no CORS headers. It requires an
**absolute** `--logbook-url` (`http://<host>:8400`): the portal has to
dial the logbook itself, and a path-shaped base (`/log`) describes the
browser's front door, not an address this process can reach. With a path
base the run page still links to the logbook, but the send button is
hidden — `GET /api/run/{uid}` reports this as `logbook_send`.

That one URL serves **both** audiences, so it must be reachable from the
portal process *and* from operators' browsers. `http://localhost:8400`
satisfies the send gate on a single-host deployment and passes every
test, then hands every operator a dead scan-card link and a dead "open
the entry" button after a successful send. Name the host.

**Copy plot to clipboard** only works when the page is a *secure
context* — `https://`, or a `localhost` host. Browsers expose the
clipboard-image API nowhere else, so on the plain-HTTP deployment
`navigator.clipboard` does not exist, the button's tooltip says it will
download, and it downloads the 2× PNG. There is no page-side workaround:
the legacy `execCommand("copy")` path carries text only.

## The logbook link (`--logbook-url`)

The logbook is **its own service** (GeecsLogbook 0.10.0: unit
`geecs-logbook`, port 8400, entries in `/var/lib/geecs-logbook`; runbook
`GeecsLogbook/deploy/DEPLOYMENT.md`). The portal no longer mounts it —
`/log` on the portal's port is nobody's route — and knows it only as a
URL: `--logbook-url` is the logbook's base, and the run page links each
scan to its card there (`<base>/day/YYYY-MM-DD#ScanNNN`; `GET
/api/run/{uid}` carries the same URL as `logbook`). An absolute URL
(`http://<host>:8400`) is used as given; a path (`/log`, once the fleet's
front door routes it to the logbook) is same-origin and carries the
portal's own proxy prefix. Without the flag there is no link. Set it on
the host through `GEECS_PORTAL_EXTRA_ARGS` in `site.env`. The link is
built for runs of `--experiment` alone: the logbook serves one
experiment's share and scan numbers restart per experiment.

Since portal 0.28.0 the same flag also enables **sending a plot** to a
scan's entry (`POST /api/run/{uid}/logbook`, the Plot tab's modebar) —
the portal's third write verb, and the only one that leaves this
process. Sending needs the absolute form of the URL, for the reason in
§ Getting a plot out of the Plot tab. The portal writes as whoever the
browser named, and the logbook's 20 MiB attachment cap applies (the
portal refuses over 8 MiB first, with its own message).

**Upgrading a host that ran `--scan-log`** (portal 0.22–0.26): the flag
and the `log` extra are gone, so the old `GEECS_PORTAL_EXTRA_ARGS` value
fails argument parsing at start — replace `--scan-log` with `--logbook-url
…` in `site.env`, re-render the unit (it no longer declares a
`StateDirectory`), and move the entries out of
`/var/lib/geecs-data-portal` per the logbook runbook **before** starting
the portal on the new unit.

## The config editor (`--config-editor`)

Off by default in the CLI. The unit template appends
`$GEECS_PORTAL_EXTRA_ARGS` from `site.env` to the command line; setting it
to `--config-editor` (as `deploy/site.env.example`, the reference profile,
does) mounts the analysis config editor at `/configs` and puts an **edit**
button on the Analysis tab. A read-only viewer leaves the key empty — a
site.env choice, never a hand-edit of the rendered unit. It is a write verb: saves
land in the `--processing-configs` tree — the share copy of the configs
repo — uncommitted, exactly like the scanner's preset writes; someone
still commits them. It needs the `analysis` extra (ScanAnalysis brings the
editor router; its `editor` extra — fastapi, jinja2 — is satisfied by the
portal's own dependencies). Turn it on where operators are expected to
tune diagnostics from the browser; leave it off on a read-only mount. It
carries no authentication beyond the lab network, like the scanner's
preset writes — `DELETE` removes a file from the share checkout (git
restores it, nothing else does).
