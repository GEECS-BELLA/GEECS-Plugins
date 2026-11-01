# Canopy + Control Room: first deploy on the services box (2026-11-01)

Headless Chrome over CDP against the live box (Control Room :8700, Canopy :8600), light and dark.

| File | State |
|---|---|
| `01_home_db_off_*` | the home before the configs pull, `GEECS_CANOPY_DB=off`: the run strip from the real queueserver (idle), Not live / In alarm empty (the two native cameras are both `live`), launcher waiting for `home.yaml`, no presets |
| `02_home_db_on_*` | `GEECS_CANOPY_DB=on`, configs #31 pulled: 120 devices on the roster (118 gateway, 2 native); Not live lists the 9 Disconnected LabVIEW devices with the module's cause; In alarm none; the launcher with `control_center` and the three services |
| `03_control_center_db_on_*` | the HTU Control Center drawing live values from the real gateway. "not answering": every `U_PLC` role (its readbacks are INVALID/COMM in the gateway, last update ~41 h before the capture, while its `connected` reads 1). "wrong kind": the Gaia internal shutter toggles bind a numeric setpoint, the five filter-wheel setpoints bind enum variables (screen-file fixes). ACave Current: its PV is unreachable |
