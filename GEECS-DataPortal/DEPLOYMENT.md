# GEECS Data Portal — deployment runbook

One service, one host: the portal runs on the services host (interim:
the queueserver worker box — the same machine as Tiled and GEECS-MCP,
so ports read `:8000` Tiled, `:8100` MCP, **`:8200` portal**). Anyone
on the lab network then browses to `http://<host>:8200/`.

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
queueserver-worker precedent; paths below assume the checkout is
`~/GEECS-Plugins` — substitute yours):

```bash
cd ~/GEECS-Plugins/GEECS-DataPortal
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

## Upgrade

```bash
cd ~/GEECS-Plugins && git pull
cd GEECS-DataPortal && poetry install
sudo systemctl restart geecs-data-portal
```

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `/health` reports a catalog error | Tiled down, or `[tiled]` uri/api_key wrong — `curl http://<tiled-host>:8000/api/v1/` |
| Day pages load, images 404 | share not mounted (or moved) at `geecs_data_local_base_path`; a 404 on one shot with others fine is the exact-match rule working (that device missed the shot) |
| Slow day listings | measure `list_runs` against the catalog first — the fix is a portal-side cache, not a schema change (scope doc, open questions) |
| Unit exits immediately at boot | Tiled not up yet — `Restart=on-failure` retries; check `journalctl -u geecs-data-portal` |

The fleet-map page (`docs/platform/fleet_map.md`) carries the
service's row — host, port, health check — and must be updated in the
same PR when this deployment moves or changes.
