# GeecsBluesky as a thin layer on stock Bluesky

> **Superseded 2026-09-09.** The phase plan below was replaced by
> `03_clean_room_rebuild.md` (the document of record; its §8 has the
> phases, its §11–12 the hardware facts). Kept for the rule in the next
> section and the history; do not plan against it.

**Plan of record: GEECS-Plugins issue #807** (companion: #806, the image
writer). This directory holds the per-phase design notes; the issue holds
the decision log and the running status. Read the issue first.

## The rule this work applies

Wherever we are tempted to patch with custom code and a native Bluesky /
ophyd-async mechanism plausibly exists, stop and check the native one first.
The result must never close off adding a stock plan or a Bluesky feature
later. (Sam, 2026-09-08.)

## The extension points GEECS is allowed to occupy

| GEECS concern | Native mechanism | Where |
|---|---|---|
| shots per step, fire-between-arm-and-wait, per-step actions, failed-move pause | `per_step` / `per_shot` callable pre-bound into the registered plan | `plans/per_step.py` (phase 3) |
| validate → resolve → claim → ScanInfo → save paths → trigger profile → setup actions; save-off → disarm → closeout | one `RunEngine` preprocessor keyed on `md["geecs"]` | `preprocessors.py` (phase 2) |
| trigger box to STANDBY on pause, re-arm on resume | `Pausable` on `ShotController` (the RE calls `pause()`/`resume()` on every object it has seen) | `shot_controller.py` (phase 4) |
| background telemetry | `bluesky.preprocessors.SupplementalData` | startup (phase 2) |
| which devices exist, lazy connection | device namespace built at `environment open`; connect on first message | `namespace.py`, `devices/geecs_device.py`, `preprocessors.connect_on_demand` (phase 1) |

Everything else is a stock plan (`bluesky.plans`) registered through a
one-line table, and the `ScanRequest` document is the *client-side*
template that expands into a stock plan call (`qs_client.submit_scan`).

## Phases

1. `01_device_namespace.md` — devices as long-lived nouns, lazily connected.
2. preprocessor preamble/finalize + SupplementalData.
3. per-step + registration table (`count`, `list_grid_scan` first), hardware.
4. native pause; retire `pause_semantics.py`.
5. `submit_scan` expansion; retire the funnel and the named plans; OSPREY follows.
6. free-run as a fly scan (after #806).

Hardware acceptance parameters (Sam-approved): trigger profile
**HTU-NoGas**; scannable **`U_S1H:current` −1 → +1 A in 0.5 A steps**
(coarser for short scans); save set **amp4in**; restore the setpoint.
