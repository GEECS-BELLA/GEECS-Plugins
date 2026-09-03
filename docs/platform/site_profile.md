# Site Profile

What changes when this fleet is deployed somewhere else, and where each
of those values lives. The [Fleet Map](fleet_map.md) is the *intended*
picture of one deployment; this page is the contract that lets the same
fleet be stood up on a new box, on a different network, or at a
different GEECS facility without forking any file in the repository.

!!! note "Reference deployment"
    HTU/Undulator on the central lab server is the worked example
    throughout the docs. Its values appear in runbooks and templates only
    as **examples** — the rule below is what makes that safe.

## The rule: one home per facility value

A facility value — experiment name, gateway address, serving interface,
timezone, data-share mount, configs-repo path, service account, checkout
root — lives in exactly one of two places. Committed files may carry it
only as an example or a placeholder.

| Side | Home | Who reads it |
|---|---|---|
| **Client** | `~/.config/geecs_python_api/config.ini` — `[Experiment]`, `[Paths]`, `[epics] ca_addr_list`, `[tiled]`, `[qserver]`, `[mcp]` (reference: [Getting started](../tutorials/getting_started.md)) | every Python client and every service process; `scripts/lab_status.sh`, `scripts/fleet_status.sh` |
| **Host** | `/etc/geecs/site.env` — one file per service host, from [`deploy/site.env.example`](https://github.com/GEECS-BELLA/GEECS-Plugins/blob/master/deploy/site.env.example) | every systemd unit (`EnvironmentFile=`), `deploy/render_units.sh`, `deploy/bootstrap_host.sh` |

On a service host `site.env` is the root: the bootstrap renders the
service account's `config.ini` *from* it (only when absent — hand edits
win afterwards), so the two never disagree. Client machines keep their
own `config.ini` exactly as before.

What is **not** a site value: the fleet's port numbers. The fleet map
fixes them (CA 5064, Tiled 8000, portal 8200, MCP 8100, queueserver
60615/60625/5568, PVA 5075/5076) and every client assumes them.

## What `site.env` carries

Two kinds of keys, documented line by line in the example file:

- **Runtime values** — exported into every service process by systemd.
  The variable *names* are the ones the services and EPICS actually
  read (`GEECS_EXPERIMENT`, `QS_EXPERIMENT`, `EPICS_CA_ADDR_LIST`,
  `EPICS_CA_AUTO_ADDR_LIST`, `EPICS_CAS_INTF_ADDR_LIST`,
  `EPICS_CAS_BEACON_ADDR_LIST`, `TZ`, `GEECS_QS_DOC_ADDR`), so nothing is
  re-mapped and a value you see in `systemctl show-environment` terms is
  the value the process got.
- **Install-time values** — the service account and its home, the
  checkout root, the absolute poetry path, the repo URL, the Tiled URI,
  the queueserver host, the data-share mount, the configs-repo path.
  These fill the unit templates' placeholders and the rendered
  `config.ini`; they are harmless in the process environment.

Syntax is systemd `EnvironmentFile=` syntax, which is stricter than
shell: `KEY=VALUE` per line, comments **only on their own lines** (a
trailing `# comment` becomes part of the value), quote values with
spaces, no expansion.

## The systemd gotcha that shapes the unit templates

systemd expands `${VAR}` from `EnvironmentFile=` **only in command
arguments**. It does not expand in `WorkingDirectory=`, `User=`,
`Environment=` lines, or the executable path (the first token of
`ExecStart=`). So every unit template has two kinds of holes:

| Hole | Filled | Examples |
|---|---|---|
| `@PLACEHOLDER@` | at render time, by `deploy/render_units.sh` (one `sed`) | `@SERVICE_USER@`, `@SERVICE_HOME@`, `@CHECKOUT_ROOT@`, `@POETRY@`, `@SITE_ENV@` |
| `${VARIABLE}` | at start, by systemd from `site.env` | `--experiment ${GEECS_EXPERIMENT}`, `--doc-addr ${GEECS_QS_DOC_ADDR}` |

Values that services read from the environment directly (the EPICS
addressing, `TZ`, `QS_EXPERIMENT`) need no hole at all — `EnvironmentFile=`
delivers them.

## Checkout layout

`<checkout root>/<service>-checkout/` — one clone per service family:

| Clone | Services | Why |
|---|---|---|
| `gateway-checkout` | CA gateway | control-room-critical, moves rarely |
| `portal-checkout` | Data Portal | iterates in days |
| `qs-checkout` | queueserver worker **and** capture daemon; also the MCP server's install source | co-location and co-versioning are a requirement of the capture design; the MCP bakes a non-editable venv (`<root>/geecs-mcp-venv`) from it so a pull never mutates code under the running server |

The root is the site's choice (`GEECS_CHECKOUT_ROOT`): the service
account's home costs no sudo; `/opt/geecs` is the same layout with one
`chown`. The rule and its reasons are in the fleet map's
[one clone per service](fleet_map.md#one-clone-per-service) section. The
configs repo (`GEECS-Plugins-Configs`) is **not** cloned per service: it
is data, read at runtime, and the copy on the data share is the one
LabVIEW reads too — `GEECS_CONFIGS_ROOT` points at it.

## Standing up a host

```bash
# as the service account, on the new host
cp deploy/site.env.example ~/site.env && $EDITOR ~/site.env
deploy/bootstrap_host.sh ~/site.env          # clones, envs, config.ini, staged units — no sudo
# then the root lines it prints: site.env → /etc/geecs, units → /etc/systemd/system, enable
scripts/fleet_status.sh                      # from any client: every row systemd / clean / matching
```

`bootstrap_host.sh` is idempotent: rerun it after editing `site.env` or
after a clone appears; it fetches existing clones but never moves their
HEAD (a pull is a deploy — do it per service, deliberately, then restart
that unit). The fleet map's bootstrap gotchas (login-shell PATH, CRLF on
the share-mounted configs checkout, quoting share paths with spaces) are
baked into the scripts and the example file.

## Onboarding a second facility

1. Copy `deploy/site.env.example`; change every value; keep every key.
2. Give every client machine a `config.ini` per
   [Getting started](../tutorials/getting_started.md) with that
   facility's experiment, gateway address, Tiled URI, queueserver host,
   and data path.
3. Run the bootstrap on the service host; do the root steps.
4. Experiment-specific *code* (analyzers, scan-database filters) follows
   the existing per-experiment package pattern
   (`analyzers/<Experiment>/`), not this profile.
5. The first real second site will find a value this page missed. Add
   it to `site.env.example` and here in the same PR — that is the
   contract growing, not breaking.

## Keeping the repository honest

- Root `CLAUDE.md` carries this as a cross-package invariant; `/land`'s
  scope check flags a diff that adds a lab literal (`192.168.`,
  `Undulator`, a service home path, a timezone) anywhere but a docstring,
  an example block, or an `analyzers/<Experiment>/` package.
- The runbooks describe *how to render and install*, not values to
  type. When one needs a concrete value it says "from `site.env`".
- `/fleet-status` is the acceptance test: after any host change, every
  row reads systemd, a clean clone, and matching installed/pyproject
  versions.
