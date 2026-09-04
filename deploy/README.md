# deploy/ — the fleet's host-side recipe

Cross-cutting deployment tooling for the service fleet (the per-service
runbooks live next to each service; the intended fleet is
`docs/platform/fleet_map.md`; the contract this directory implements is
`docs/platform/site_profile.md`).

| File | What it is |
|---|---|
| `site.env.example` | The host half of the site profile — every facility-specific value a service host needs, HTU/Undulator as the worked example. Copy, edit, install at `/etc/geecs/site.env`. |
| `site_env_lib.sh` | Shared loader (mirrors systemd's `EnvironmentFile=` parsing). Sourced by the two scripts. |
| `render_units.sh` | Fill the `@PLACEHOLDER@` holes in every service's unit template from a site.env into a staging directory; prints the root install lines. |
| `bootstrap_host.sh` | Idempotent, unprivileged fresh-host bootstrap: per-service clones, poetry envs with each service's extras, the baked MCP venv, `config.ini` rendered from site.env if absent, units rendered to staging; prints the root steps. |

Fresh host, in order:

```bash
git clone <repo> <root>/qs-checkout && cd <root>/qs-checkout   # the worker's clone hosts the bootstrap
cp deploy/site.env.example ~/site.env && $EDITOR ~/site.env   # as the service account
deploy/bootstrap_host.sh ~/site.env                           # unprivileged, rerunnable
# ...then the printed sudo lines (site.env → /etc/geecs, units → /etc/systemd/system, enable)
scripts/fleet_status.sh                                       # from any client: every row systemd / clean
```

The unit templates themselves stay with their services
(`GeecsCAGateway/deploy/`, `GEECS-DataPortal/deploy/`,
`GeecsBluesky/qserver/deploy/`, `GeecsBluesky/capture/deploy/`,
`GEECS-MCP/deploy/`); this directory only knows their list.
