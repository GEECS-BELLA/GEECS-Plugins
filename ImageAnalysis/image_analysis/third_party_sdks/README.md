# third_party_sdks — vendor SDKs live here, out of git

This directory holds vendor Python bindings that some analyzers import
but that must **not** be committed: they are large binary distributions
(hundreds of MB of DLLs), Windows-only, and licensed by the vendor. The
whole directory is `.gitignore`d except this README.

| Subdirectory | Consumer | Source |
|---|---|---|
| `wavekit_43/` | `image_analysis.analyzers.HASO_himg_has_processor` (HASO wavefront sensor) | Imagine Optic WaveKit 4.3 Python distribution — copy the `wavekit_43` folder from the vendor installer so that `wavekit_43/wavekit_py/` and `wavekit_43/dlls/x64/` sit directly under this directory |

The import path is fixed: `image_analysis.third_party_sdks.wavekit_43.wavekit_py`.
When the SDK is absent the HASO analyzer logs an import warning and its
test module skips (`pytest.importorskip`), so every other analyzer and
every non-Windows machine is unaffected.

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
