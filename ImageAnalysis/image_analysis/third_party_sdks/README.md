# third_party_sdks — vendor SDKs live here, out of git

This directory holds vendor Python bindings that some analyzers import
but that must **not** be committed: they are large binary distributions
(hundreds of MB of DLLs), Windows-only, and licensed by the vendor. The
whole directory is `.gitignore`d except this README.

| Subdirectory | Consumer | Source |
|---|---|---|
| `wavekit_43/` | `image_analysis.analyzers.HASO_himg_has_processor` (HASO wavefront sensor) | Imagine Optic WaveKit 4.3 Python distribution — copy the `wavekit_43` folder from the vendor installer so that `wavekit_43/wavekit_py/` and `wavekit_43/dlls/x64/` sit directly under this directory |

The import path is fixed: `image_analysis.third_party_sdks.wavekit_43.wavekit_py`.
When the SDK is absent, importing the HASO analyzer module raises
`ModuleNotFoundError` (a hard import, by design) and its test module
skips via `pytest.importorskip`; every other analyzer and every
non-Windows machine is unaffected because the module is only imported on
demand by class path.

## Lab-owned files that are NOT part of the vendor installer

Two HASO4 sensor calibration files (serial 680-8244) were committed
alongside the SDK and are consumed through `wavekit_config_path` in
`config.ini`. They are not in the Imagine Optic installer, so re-copying
the SDK does not restore them. Recover them from git history:

```
git show a3e77ff7:ImageAnalysis/image_analysis/third_party_sdks/wavekit_43/WFS_HASO4_LIFT_680_8244_gain_enabled.dat > wavekit_43/WFS_HASO4_LIFT_680_8244_gain_enabled.dat
git show a3e77ff7:ImageAnalysis/image_analysis/third_party_sdks/wavekit_43/WFSL_HASO4_LIFT_680_8244_MC.lift > wavekit_43/WFSL_HASO4_LIFT_680_8244_MC.lift
```

The better home for them is the software share, with `wavekit_config_path`
pointing there; that relocation is a follow-up, not part of the untrack.

**Existing checkouts, read before pulling image-analysis ≥ 1.13.2:** git
removes files from the working tree when they go from tracked to absent
between commits, and ignore rules do not protect that transition. On any
machine that has `wavekit_43/` (the HASO analysis box in particular), copy
the directory aside before pulling and move it back afterwards, or
re-copy it from the vendor installer. Once it is back it is ignored and
stays put.

History note: `wavekit_43/` was committed in March 2025 before the ignore
rule existed, so the rule never applied; it was untracked in
image-analysis 1.13.2. The blobs remain in git history until a separate
history rewrite is decided.
