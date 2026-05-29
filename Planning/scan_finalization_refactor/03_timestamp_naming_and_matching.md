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

`build_device_file_map` learns to match a device's per-shot file by
joining on **`acq_timestamp`**, when the shot-number filename pattern
doesn't apply.

The key insight: the device's TCP `acq_timestamp` (which lands in the
DataLogger log) and the device's filename timestamp **come from the
same source**. They should be equal modulo ms-level rounding noise from
truncating in both representations. So the fallback isn't fuzzy timing
matching for shot membership — it's exact join on `acq_timestamp` with
a small tolerance for rounding.

1. **Primary path (unchanged):** match by shot number embedded in the
   filename. If all shots resolve, done.
2. **Fallback:** for each log entry the primary path missed, look in
   the device folder for a file whose embedded timestamp is within
   ±~10 ms of that entry's `acq_timestamp`.
3. **Report:** entries that still have no file are logged; the
   resulting dataframe path entry is empty, same as today.

The per-shot association comes for free: the log already has
shot ↔ acq_timestamp per device, so once we resolve acq_timestamp →
path, we have shot → path.

This is strictly additive: scans that already work continue to work.
Only the previously-empty-path entries are affected, and only
positively.

### Sketch of the fallback

```python
def _fallback_match_by_acq_timestamp(
    device_dir: Path,
    log_entries: pd.DataFrame,        # has columns: shot, acq_timestamp
    *,
    tolerance_s: float = 0.010,       # ~10 ms — covers ms-truncation noise
) -> dict[int, Path]:
    """For each log entry, find the file in device_dir whose embedded
    timestamp matches the entry's acq_timestamp within ±tolerance.
    Returns shot → path."""
    candidates = _scan_dir_for_timestamped_files(device_dir)  # [(ts, path), ...]
    matches = {}
    for _, entry in log_entries.iterrows():
        for ts, path in candidates:
            if abs(ts - entry.acq_timestamp) <= tolerance_s:
                matches[entry.shot] = path
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
2. **Can 3a and 3b ship separately?** Yes. 3b is the more valuable of
   the two (fixes existing scans). 3a is convenience for new flows.
   Recommend splitting if 3a's scope grows.

Notes that are *not* open questions, just for the record:

- **Tolerance is ~10 ms, not a tunable.** Both the log's `acq_timestamp`
  and the filename's embedded timestamp truncate at ms level. ~10 ms
  catches rounding noise and nothing else. There's no shot-window
  ambiguity to worry about because this is an exact join, not a
  time-bucket search.
- **Reference-frame question is moot.** The log's `acq_timestamp` and
  the file's embedded timestamp come from the same source on the same
  device, so they're in the same frame by construction. (The per-device
  `t0` recorded at scan setup — see piece 1 — is what relates each
  device's `acq_timestamp` to `elapsed_time`; the matcher operates on
  `acq_timestamp` directly, before that transform, so `t0` is not
  involved.)

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
