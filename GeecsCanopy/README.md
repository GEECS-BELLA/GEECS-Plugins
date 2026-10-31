# GeecsCanopy

SLAC's [Canopy](https://github.com/slaclab/canopy) as the GEECS display
backend: live PVs over WebSocket (Hare), snapshots (Squirrel), an archiver
read API (Mole) and a PV directory (Raccoon), run unpatched in a derived
image. This package is the GEECS-shaped part around it.

| | |
|---|---|
| `geecs_canopy/directory.py` | Raccoon's directory adapter: the three list calls, the pattern grammar, the failure rules |
| `geecs_canopy/native.py` | The native source: `native_devices.yaml` and `panel_devices.yaml`, joined by device name, roles from the family files |
| `geecs_canopy/gateway.py` | **Transitional**: the CA gateway's served set for the LabVIEW devices, from the GEECS DB |
| `geecs_canopy/pin.py` | The Canopy image digest and commit this package runs in |
| `deploy/` | Dockerfile, compose template, systemd unit, renderer, secrets example |

`DEPLOYMENT.md` is the runbook; `CLAUDE.md` holds the rules.
The decision record is `Planning/geecs_canopy.md` §7.
