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
root, poetry path, experiment, and timezone all come from `site.env`;
nothing site-specific is typed into the unit by hand.

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
(`/run`, `/api`, …). For a proxy that cannot send it, start the
service with a static prefix instead: `geecs-data-portal --root-path
/portal` (the header, when present, wins) — in that fallback mode
avoid mount names that collide with a portal route head, and know that
trailing-slash redirects drop the prefix (Starlette builds them from
the un-prefixed path; the header mode re-prefixes the path and has
neither limitation). Malformed header values (not root-absolute, `//`,
whitespace, query/fragment characters) are ignored rather than
propagated into page links. The header is a strip-style contract: a
proxy that does **not** strip the prefix must not send it (the symptom
of that misconfiguration is loud — every page 404s).

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
| Slow day listings | measure `list_runs` against the catalog first — the fix is a portal-side cache, not a schema change (scope doc, open questions) |
| Unit crash-loops at start | `status` shows **217/USER** — the installed unit is a pre-profile file (or a copy from an old staging run) with the generic `User=`; the clone it came from predates the templated units — pull it forward, re-render with `deploy/render_units.sh`, reinstall (site profile page). Wrong absolute Poetry path in `ExecStart` (`status` shows 203/EXEC); env installed by a different account than `User=` (empty venv — reinstall as the service account); or port 8200 already taken. A down Tiled does **not** exit the service — that shows up as the `/health` row above |
| Evening scans 404 (or resolve oddly) while daytime scans work | host timezone differs from the scanner hosts' — daily folders are named by the scanner's local date. `site.env` sets `TZ`; keep it matching the lab's zone |

The fleet-map page (`docs/platform/fleet_map.md`) carries the
service's row — host, port, health check — and must be updated in the
same PR when this deployment moves or changes.
