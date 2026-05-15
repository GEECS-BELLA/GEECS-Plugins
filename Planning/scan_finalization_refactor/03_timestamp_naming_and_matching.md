# Piece 3 — Timestamp filenames + fallback matching

**Branch model:** independent feature branch off master
**Abort risk:** low
**Depends on:** nothing

---

## Problem

Per-device files in a scan are named with a shot number in the filename,
e.g. `Scan047_UC_CameraA_010.png`. The shot number is canonical: it's
what `ScanData.build_device_file_map` uses to associate a row in the
s-file with a file on disk.

Two real-world cases break this:

1. **Devices that don't know the shot number.** Some acquisition paths
   write files with timestamps or sequence numbers that aren't the GEECS
   shot. The file is correct, the data is correct, but the matcher
   doesn't find it without manual munging.
2. **Renamed / partial scans.** When a scan is restarted, paused, or
   when files are copied from a parallel acquisition path, the shot
   number embedded in the filename can drift relative to the s-file's
   shot column.

Today the workaround is either to rename files by hand, or to use the
`device_file_stem` kwarg (added recently) to override the prefix part —
which handles the "folder name ≠ stem" case but not the "shot number
isn't in the filename" case.

---

## Proposed solution

Two additive changes:

### 3a. Timestamp naming option (in Scanner GUI)
Allow per-device save config to specify a naming pattern that includes
the device's acq timestamp:

```yaml
save_elements:
  - device_name: UC_CameraA
    filename_pattern: "{scan}_{device}_{ts:.4f}.png"   # NEW
```

Default pattern preserves current `{scan}_{device}_{shot:03d}.{ext}`
behavior. This adds a way to opt into timestamp naming where the device
flow makes shot-number naming awkward.

### 3b. Timestamp fallback in ScanData/ScanPaths (in geecs-data-utils)

`build_device_file_map` learns to match by timestamp when shot-number
matching produces gaps:

1. **Primary path (unchanged):** look for files matching the shot
   pattern. If all shots find a file, done.
2. **Fallback:** for shots with no file under the shot pattern, search
   for files in the per-device folder whose embedded timestamp falls
   inside the shot's time window (computed from the s-file's
   `elapsed_time` column ± a tolerance).
3. **Report:** any shots that still have no file are logged; behavior
   is unchanged (empty path entry in the dataframe).

This is strictly additive: scans that already work continue to work.
Only the previously-empty-path shots are affected, and only positively.

### Sketch of the fallback

```python
def _fallback_match_by_timestamp(
    device_dir: Path,
    unmatched_shots: list[ShotRow],
    *,
    tolerance_s: float = 0.5,
) -> dict[int, Path]:
    """For shots without a shot-pattern match, look for files in
    device_dir whose embedded timestamp falls within ±tolerance of
    the shot's elapsed_time."""
    candidates = _scan_dir_for_timestamped_files(device_dir)
    matches = {}
    for shot in unmatched_shots:
        for ts, path in candidates:
            if abs(ts - shot.elapsed_time) <= tolerance_s:
                matches[shot.shot] = path
                break
    return matches
```

The function is opt-in via a `match_by_timestamp_fallback=True` kwarg on
`build_device_file_map`, defaulting to `False` initially so we can
validate against real scans before turning it on by default.

---

## Files touched

- `GEECS-Data-Utils/geecs_data_utils/scan_paths.py` —
  `build_device_file_map` gains `match_by_timestamp_fallback` and the
  helper above.
- `GEECS-Data-Utils/geecs_data_utils/scan_data.py` — pass-through kwarg.
- `GEECS-Data-Utils/tests/test_scan_paths_timestamp_match.py` — new
  tests using a synthetic device dir with mixed shot/timestamp naming.
- `GEECS-Data-Utils/CHANGELOG.md` — minor bump entry.
- `GEECS-Scanner-GUI/geecs_scanner/...` (filename pattern surface) —
  scoped to the save-element config layer. Defer if 3b alone is more
  valuable.
- `GEECS-Scanner-GUI/CHANGELOG.md` — minor bump entry (if 3a included).

---

## Open questions

1. **Where does the timestamp come from in a filename?** Need a
   convention or a regex. Recommend `_t{seconds:.4f}_` or a trailing
   `_{epoch_seconds:.4f}.{ext}` and a configurable regex on
   `build_device_file_map`. Decide before 3b's PR.
2. **Tolerance.** Default 0.5 s? Could be aggressive. Defer to a tested
   default based on the spread of inter-shot intervals we see in the
   field.
3. **Should the s-file's `elapsed_time` reference frame match the
   device's filename timestamp reference frame?** Today they don't
   necessarily. This is the actual hard problem hiding inside 3b. The
   fallback function should accept an explicit offset kwarg and the
   tests should cover the offset case.
4. **Can 3a and 3b ship separately?** Yes. 3b is the more valuable of
   the two (fixes existing scans). 3a is convenience for new flows.
   Recommend splitting if 3a's scope grows.

---

## Sequencing

- Independent of pieces 1, 2, 4.
- 3b is the higher-priority half. Land it first.
- 3a can land any time, including never (3b is sufficient if devices
  already produce timestamped filenames out-of-band).

---

## Out of scope

- Rewriting how files actually arrive at the per-scan folder (piece 2
  territory).
- Changing the s-file's `elapsed_time` semantics (piece 1 territory).
- Trying to match by content hash. Timestamps are the natural key here.

---

## Abort risk

Very low for 3b: the fallback only runs for shots that had no match
under the current code path, so it can only add files to the
dataframe — never remove or relocate. If the timestamp matcher is
wrong, we get bad path entries, but no existing entries are affected.

Low for 3a: an additive config field with a default that preserves
current naming.

---

## Branching strategy

Two options:
- **Combined:** `feature/timestamp-naming-and-fallback` off master.
  One PR for 3a + 3b. Recommended if both are ready.
- **Split:** `feature/timestamp-fallback-match` first (3b), then
  `feature/timestamp-naming` follows (3a). Recommended if 3a's design
  isn't settled.
