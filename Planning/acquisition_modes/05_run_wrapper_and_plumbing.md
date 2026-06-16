# Run Wrapper, Scan Numbering, and Config Plumbing

## `geecs_run_wrapper` — the enforcement point for the contract

Extract the scan-bookkeeping currently embedded in `BlueskyScanner`
(`_claim_scan_number`, save-path configuration, metadata assembly) into a
reusable plan preprocessor:

```python
plan = geecs_run_wrapper(
    inner_plan,
    devices=...,            # for save-path configuration + t0 metadata
    experiment=...,
    acquisition_mode=...,
)
```

Responsibilities:

1. Claim the day-scoped scan number / `scans/ScanNNN/` folder (scanner-side
   code — the **one** place outside the GUI allowed to create scan folders;
   the cross-package "analysis never creates scan folders" invariant stands).
2. Set each saving device's `localsavingpath` to the
   `Y{YYYY}/{MM-Month}/{YY_MMDD}/.../Scan{NNN}/{device}/` folder; clear in a
   finalizer (today's `_scan_with_saving` behavior).
3. Inject the start-doc metadata contract (`01`): `geecs_event_schema`,
   `scan_number`, `scan_folder`, `experiment`, `acquisition_mode`,
   `rep_rate_hz`, and **`scan_id` = claimed scan number** (per-run `md`
   overrides the RunEngine's internal counter; Bluesky `scan_id` has no
   uniqueness contract, so the daily reset to 1 is fine — `uid` is the key).
4. Day rollover: a scan keeps the number/folder claimed at start even if it
   crosses midnight (matches existing scanner behavior).

`BlueskyScanner` becomes a consumer of the wrapper; custom notebook plans wrap
themselves in it and inherit numbering, save paths, and the metadata contract
in one line.

## Config plumbing

- `acquisition_mode: Literal["strict_shot_control", "free_run_time_sync"]`
  added to scan options (lives next to `rep_rate_hz`; touching
  `geecs_data_utils`' `ScanConfig` is acceptable if that placement is more
  natural — cross-package change pre-approved, bump that package too).
- Default: `strict_shot_control`.
- Free-run extras (`reference_device`, `t0_sync_window_s`, grace wait) are
  YAML-only initially; env override `GEECS_BLUESKY_ACQUISITION_MODE` for
  quick switching. No GUI exposure yet.
- `BlueskyScanner` dispatch: mode → plan factory.

## Versioning

Per repo policy: `poetry version minor` in `GeecsBluesky` (and in
`GEECS-Data-Utils` if `ScanConfig` gains the field) + CHANGELOG entries with
the code changes.
