# GEECS Data Portal

A read-only web view of GEECS scan data: pick a day, pick a scan, see its
metadata and scalar plots — from any browser on the lab network, nothing
to install.  Built on the same `ScanCatalog` layer as the GEECS-Console
scan browser (`geecs_data_utils.tiled_catalog`).

```bash
poetry install
poetry run geecs-data-portal --experiment Undulator   # serves on :8200
```

Requires the `[tiled]` section of `~/.config/geecs_python_api/config.ini`
(the same file every GEECS-Plugins package reads).  Design and arc plan:
`Planning/data_portal/01_data_portal_scope.md`; developer rules:
`CLAUDE.md` here.
