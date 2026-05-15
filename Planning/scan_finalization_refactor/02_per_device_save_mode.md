# Piece 2 — Per-device save mode

**Branch model:** independent feature branch off master
**Abort risk:** low
**Depends on:** nothing

---

## Problem

The current scan engine has a single global "save mode" that decides
how per-device output files (images, TDMS chunks, etc.) get from the
device to the per-scan directory. Two modes exist today:

- **`direct`** — device writes straight into the per-scan folder over
  the network drive. Slow on big files but the file ends up in the
  right place. The scan blocks on per-shot IO.
- **`local`** — device writes to a local scratch path, and the engine
  moves the file to the per-scan folder afterward. Faster per-shot,
  but the move is a serialized post-shot step.

The mode is global to the scan. That means a scan with one slow
high-volume device and ten lightweight devices pays the worst-case
cost for all of them. Camera-light scans (e.g. magspec scoring runs
where only one or two camera images matter) get pinned to whatever
the heaviest device wants, even though most devices could safely use
the faster path.

---

## Proposed solution

Allow each device to declare a `save_mode_override` per scan, falling
back to the global default. Supported overrides:

| Mode | Behavior |
|---|---|
| `direct` | Device writes into per-scan folder over network. (Current default.) |
| `local` | Device writes to local scratch; engine moves post-shot. |
| `skip` | Device produces no saved file this scan. Data still flows over TCP and into the long-format TSV (piece 1). |

`skip` is new and is a meaningful unlock: optimization scans that only
care about scalar feedback ("what's the BPM reading") can suppress
image saving entirely for that device, dropping per-shot IO to nothing.

The override lives on the per-device save-element config, alongside
existing fields like the data variable list.

### Sketch of the wiring

```yaml
# save_elements config (existing file)
save_elements:
  - device_name: UC_CameraA
    variable_list: [exposure, gain]
    save_mode_override: skip       # NEW — don't save image files this scan
  - device_name: UC_BPM3
    variable_list: [x_um, y_um]
    # no override → use global default
```

In code: `DeviceManager` resolves the effective save mode per device at
scan setup; `DataLogger` and the scan finalization path consult that
resolution rather than the single global mode.

---

## Files touched

- `GEECS-Scanner-GUI/geecs_scanner/engine/device_manager.py` —
  resolve effective per-device save mode at scan setup; expose
  `device.effective_save_mode` for downstream use.
- `GEECS-Scanner-GUI/geecs_scanner/engine/data_logger.py` — honor
  `skip` (don't expect a file to land).
- `GEECS-Scanner-GUI/geecs_scanner/engine/scan_data_manager.py` —
  honor `skip` (don't move/check files for skipped devices).
- `GEECS-Scanner-GUI/geecs_scanner/app/save_element_editor.py` (or
  equivalent) — UI surface for the override field.
- `geecs_data_utils/save_element_config.py` (if the config schema
  lives there) — add the field with a default that preserves current
  behaviour.
- `GEECS-Scanner-GUI/CHANGELOG.md` — minor bump entry.

(File names approximate; exact paths verified during implementation.)

---

## Open questions

1. **Where does the schema live?** If `ScanConfig` is in
   `geecs_data_utils` (acknowledged organizational debt in root
   CLAUDE.md), the save-element schema may also straddle packages.
   Confirm location during implementation; do not migrate as part of
   this PR.
2. **UI surface.** Is the override a dropdown in the save-element
   editor, or only a YAML edit? Recommend dropdown — the GUI is the
   normal way users configure scans, and hiding this behind YAML
   limits adoption. But UI can land as a follow-up if scope is tight.
3. **What does `skip` mean for TDMS?** TDMS is a separate concern —
   it's a scan-level file, not per-device. `skip` only affects
   per-device side files. Document this in the docstring.
4. **Should `skip` warn at scan end?** If a device is `skip` but the
   variable list expects a file column, the s-file will have empty
   path entries. We should either (a) detect and warn at scan setup,
   or (b) accept it as the user's intent. Recommend (a).

---

## Sequencing

- Independent of pieces 1, 3, 4.
- Useful on its own immediately (faster optimization scans).
- Once piece 4 lands, `skip` mode pairs especially nicely with the
  finalization step — finalization can recognize skipped devices and
  not look for files.

---

## Out of scope

- Changing the default save mode globally.
- Adding new save modes beyond `direct`, `local`, `skip`.
- Per-shot dynamic save mode (e.g. "skip this shot only"). Per-scan
  granularity is enough; per-shot would mean a different data model.
- Migrating `ScanConfig` between packages.

---

## Abort risk

Low. The change is gated by a new optional field with a default that
preserves current behaviour. Nothing existing breaks. If `skip` mode
causes issues we don't see, the field can be ignored at runtime via a
config-side default.

---

## Branching strategy

`feature/per-device-save-mode` off master. Single PR. Squash-merge.
Can absolutely overlap with piece 1's PR.
