# Deploying the GEECS Scanner

The scanner is one systemd unit on the worker host, its own process on
port **8300**, behind the same front door as the Data Portal and the
logbook (arc PR 5: one origin, `/portal`, `/log`, `/scan`). It is a client
of the queueserver — same standing as the Qt console had, same dependency
cone as GEECS-MCP — and it runs from the **worker's checkout**
(`<root>/qs-checkout`), so the plan surface it submits against is the one
the worker defines.

## Install

On the worker host, as the service user:

```bash
cd <root>/qs-checkout/GeecsScanner
poetry install
poetry run geecs-scanner --experiment <Experiment> --port 8300   # try it by hand first
```

`--experiment` is the site value `GEECS_EXPERIMENT` from `site.env`; the
manager address comes from `~/.config/geecs_python_api/config.ini`
`[qserver]` (host, and optionally the console-stream and document-stream
addresses), exactly as for every other queue client. Without a `[qserver]`
section the scanner starts and reports the manager unreachable — it never
guesses an address.

## The unit

Render and install through the fleet's one path (`docs/platform/site_profile.md`):

```bash
deploy/render_units.sh /etc/geecs/site.env ~/deploy-staging
sudo install -m 644 ~/deploy-staging/geecs-scanner.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now geecs-scanner
```

`site.env` keys the template reads: `GEECS_EXPERIMENT` (required),
`GEECS_SCANNER_EXTRA_ARGS` (optional; `--root-path /scan` when the front
door mounts the scanner under a prefix and does not send
`X-Forwarded-Prefix`; `--portal-url <the portal's base URL>` so the rail's
scan links open the Data Portal's run pages — behind the front door that
is the `/portal` prefix).

The scan.log tail reads `<scan_folder>/scan.log` as the worker's start
document names the folder, so the scanner runs on the worker host (same
data-share mount); elsewhere the tail says the file is not readable and
nothing else changes.

## Verify

```bash
curl -s localhost:8300/health | jq            # ok, manager, readiness, version, tiled_writer
curl -s localhost:8300/api/status | jq        # re_state, readiness word
curl -N localhost:8300/api/events             # status / progress / console events
```

`readiness` must read `ready`; anything else names the recovery gesture
(`environment_closed` → the `geecs-qserver-ready` unit; `plans_empty` →
`qserver permissions reload lists`). `tiled_writer.state` is the Tiled
writer's word (`ok` / `degraded` / `failed`, from its heartbeat under
`GEECS_TILED_WRITER_STATE` — the unit sets it to
`/var/lib/geecs-tiled-writer`, the directory the `geecs-qserver` and
`geecs-tiled-writer` units own; the qserver runbook § The Tiled writer
says what each word means) — shown on the page as the "tiled writer"
chip, **never a gate on submit**. `scripts/fleet_status.sh` reads
`/health` here for the fleet picture (ok only when `readiness` is `ready`)
and shows the unit as "GEECS Scanner", and reads the writer's word from
the same answer for its "Tiled writer" row; a host without the scanner
shows the role as absent, not down.

## Behind the front door

The scanner honours `X-Forwarded-Prefix` per request (the portal's
convention). The SSE route needs the proxy not to buffer: Caddy does not
by default; for nginx the response already carries `X-Accel-Buffering: no`.

## Upgrade

```bash
git -C <root>/qs-checkout pull
cd <root>/qs-checkout/GeecsScanner && poetry install
sudo systemctl restart geecs-scanner
```

A restart interrupts nothing on the worker — a running scan continues; the
page reconnects to the event stream by itself.

## Demo mode

`poetry run geecs-scanner --demo` serves the same API over an in-memory
manager that runs submitted scans by itself. It contacts nothing and is how
the API and the page are exercised without a worker — on a laptop, or on
the host beside the real unit on another port.
