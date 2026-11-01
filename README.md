# Push 4: screen kinds — the HTU Control Center, live

The branch `feat/push4-screen-kinds` (a toggle with `values: {on, off}` over a
numeric role; the `select` kind over an enum role) drawing the regenerated
Control Center (configs branch `feat/control-center-screen-kinds`) against
the services box's Canopy and the real Undulator gateway and DB, read key
only, 2026-11-01. No write was made.

- **Zero "wrong kind" widgets** (94 of 95 live; ACave Current unreachable, a
  disabled device's bare PV, as before).
- **Gaia internal shutter A/B** (GAIA control): toggles with `values: {on: 1,
  off: 0}`; both read 99.000, so the button says `unknown` and is disabled
  (no guessed direction); the readouts below show the raw 99.000.
- **Filters** (Ghost pulsed, Probe, HASO) and **Mode imager** (Coarse, Fine):
  selects of the positions the gateway serves for `pos.Channel0`. The DB's
  choices for these wheels are the strings `1`–`6`, so those are the option
  names; the current option is the readback's.

`control_center_light.png`, `control_center_dark.png`: the page under
`prefers-color-scheme` light and dark (theme "Laser", the default).
