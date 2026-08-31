# GEECS Data Portal — deployment runbook

One service, one host: the portal runs on the worker host — the fleet
map (`docs/platform/fleet_map.md`) is the authority on which machine
that is at any given time. On the interim box it happens to share a
machine with Tiled and GEECS-MCP, so the ports read `:8000` Tiled,
`:8100` MCP, **`:8200` portal** — but only the portal's own port is
load-bearing; nothing below assumes Tiled is local. Anyone on the lab
network browses to `http://<host>:8200/`.

The portal is **read-only by doctrine** (see `CLAUDE.md`): it renders
the Tiled catalog and reads per-shot files off the data share, and must
never create or modify anything on the scans path. On a dedicated
portal host, mount the share read-only; on a shared host where other
services (the queueserver worker) legitimately write, the guarantee is
the code path itself — pinned by the package's tree-untouched tests.

## Prerequisites

- Ubuntu with **Python 3.11** and **Poetry** on the service account
  (same account and tooling as the other services on the box).
- `~/.config/geecs_python_api/config.ini` with a `[tiled]` section
  (`uri`, `api_key`) and a `[Paths]` section whose
  `geecs_data_local_base_path` points at the mounted data share —
  the same file every GEECS-Plugins package reads.
- The data share mounted at that path (e.g. `/mnt/<share>/data/`).
- Port **8200** free (`ss -tlnp | grep 8200`).

## Install

The portal runs from its own repo checkout so it can be upgraded
without touching the checkouts other services run from (the
queueserver-worker precedent). Give the checkout a portal-specific
name — on a box that also runs the CA gateway, a checkout named plain
`~/GEECS-Plugins` is likely the *gateway's* running checkout, and this
runbook's Upgrade step must never `git pull` that one. Paths below
assume `~/GEECS-Plugins-portal` — substitute yours.

**Run every command in this section as the service account** (the
`User=` of the unit): Poetry keys the project venv under the invoking
user's cache, so an env installed by an admin account is invisible to
the service and the unit crash-loops on an empty env while admin-side
checks pass.

```bash
cd ~/GEECS-Plugins-portal/GEECS-DataPortal
poetry env use python3.11
poetry install
```

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

`deploy/geecs-data-portal.service` — install per its header comments
(copy, `sudoedit` the account/paths, `daemon-reload`,
`enable --now`). Site specifics (the real account, checkout path,
`--experiment`) live in the `/etc/systemd/system` copy, not in the
repo file.

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

The header is per-request and needs no portal-side config. For a proxy
that cannot send it, start the service with a static prefix instead:
`geecs-data-portal --root-path /portal` (the header, when present,
wins). Malformed header values (not root-absolute, `//`, whitespace)
are ignored rather than propagated into page links.

For a panel health LED, probe `GET /health` — 200 always (the JSON
`ok` field reports the catalog probe, so a down Tiled shows as a
degraded catalog, not a dead portal).

## Upgrade

```bash
cd ~/GEECS-Plugins-portal && git pull
cd GEECS-DataPortal && poetry install
sudo systemctl restart geecs-data-portal
```

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `/health` reports a catalog error | Tiled down, or `[tiled]` uri/api_key wrong — `curl http://<tiled-host>:8000/api/v1/` |
| Day pages load, images 404 | share not mounted (or moved) at `geecs_data_local_base_path`; a 404 on one shot with others fine is the exact-match rule working (that device missed the shot) |
| Slow day listings | measure `list_runs` against the catalog first — the fix is a portal-side cache, not a schema change (scope doc, open questions) |
| Unit crash-loops at start | wrong absolute Poetry path in `ExecStart` (`status` shows 203/EXEC); env installed by a different account than `User=` (empty venv — reinstall as the service account); or port 8200 already taken. A down Tiled does **not** exit the service — that shows up as the `/health` row above |
| Evening scans 404 (or resolve oddly) while daytime scans work | host timezone differs from the scanner hosts' — daily folders are named by the scanner's local date. The unit pins `TZ`; keep it matching the lab's zone |

The fleet-map page (`docs/platform/fleet_map.md`) carries the
service's row — host, port, health check — and must be updated in the
same PR when this deployment moves or changes.
