# HASO wavefront (WaveKit)

How a HASO wavefront sensor's shots become phase maps on the analysis
core — the `haso` measure — and how to stand the install up, prove it,
and add a sensor. Everything below runs on the Linux analysis host; the
Windows box is out of the analysis path (WaveView stays for live viewing).

## The path

```
LabVIEW device  ──.himg per shot──▶  scans/ScanNNN/U_HasoLift/
                                          │
                              HasoLift_stack (kind himg_to_stack, a click)
                                          ▼
                                   U_HasoLift/U_HasoLift.h5      the capture stack: frames + headers
                                          │
                              HasoLift (the haso measure, a click)
                                          ▼
       s-file columns  U_HasoLift_phase_rms, U_HasoLift_phase_pv        per shot
       analysis/ScanNNN/U_HasoLift/Array2DScanAnalyzer/
           U_HasoLift_wavefront.h5     every shot: frame (processed phase), extras/{raw_phase,
                                       intensity, slopes_x, slopes_y, pupil}, shots
           U_HasoLift_average_processed.h5 + *_visual.png, the summaries — the usual products
```

1. **Convert first.** A run reads the device's capture stack and nothing
   else: an unconverted scan is refused with a message naming
   `HasoLift_stack`. The converter is PR 1 of this arc
   (`geecs-himg convert`, or the `HasoLift_stack` analyzer in the Portal).
2. **Backgrounds** are the core's own: a `from_scan` frame input over a
   dark scan of the same day (its stack — convert that scan too) or a
   file, subtracted by a `background_frame` step *before* the measure on
   the raw pixels. Numpy's subtraction equals the SDK's image subtraction
   bit for bit (verified 2026-09-28), so no `.has` background exists any
   more.
3. **The measure** rounds and clips the processed frame to the sensor's
   16-bit pixels, rebuilds a temporary `.himg` from them and *any* header
   of that sensor (the stack's first; the per-shot header differs only
   in its timestamp), and hands it to WaveKit in a fresh process per
   shot: image → engine (LIFT at `wavelength_nm`, `start_subpupil`) →
   slopes → intensity and raw zonal phase → (a `reference`'s slopes
   subtracted) → rectangular pupil `mask` → `filters` → processed zonal
   phase, processed slopes, pupil.
4. **Products** are the core's standard ones plus the per-scan
   *wavefront store*: one HDF5 per scan and recipe under the analysis
   tree, one row per shot, float32 (the SDK's precision), rows one chunk
   each so a follow-on recipe reads one shot at a time. The store is
   written as `.part` (created exclusively) and renamed at the end of a
   clean run; a store failure costs the store, never the run's scalars.
   A `.part` left behind by a run that died mid-way blocks the store on
   every rerun of that recipe on that scan — the run still writes its
   scalars and figures, and the portal's log carries an error naming
   the file: remove it once nothing is analyzing the scan. Nothing is
   written beside the raw data: no `.has`, no TSV sidecars.

### The recipe

```yaml
schema_version: 3
device: U_HasoLift
input: {kind: camera, file_tail: .himg, format: device_hdf5}
measure:
  kind: haso
  sensor_config: WFS_HASO4_LIFT_680_8244_gain_enabled.dat   # a file NAME under wavekit_configs_path
  mask: {top: 175, bottom: 350, left: 10, right: 670}         # numpy slice bounds on the slopes grid; centre the pupil on the feature
  filters: {tilt_x: true, tilt_y: true, curvature: true, astigmatism_0: true, astigmatism_45: true, others: false}
  wavelength_nm: 800.0
  start_subpupil: [87, 64]
  zonal_prefs: [100, 500, 1.0e-6]     # weak iterations, max iterations, residual limit
scan: {priority: 10, save: true}
summaries: [{kind: image_grid}, {kind: average}]   # the display figures
```

`mask` unset keeps the sensor's own pupil. The filter defaults are the
legacy analyzer's (tilt, curvature and both astigmatisms removed, the
rest kept). `save: false` keeps the two scalars and writes no store.

Scalars: `phase_rms` (the processed phase's standard deviation over the
finite values inside the pupil) and `phase_pv` (its peak-to-valley), in
the SDK's phase unit (µm).

### A reference (the plasma imprint)

A **reference** is a wavefront the measure subtracts: the plasma's
imprint on the probe is phase(probe + plasma) − phase(probe alone). It
is not the dark background above — that stays, on the pixels; the
reference acts on the *slopes*. Name a frame input and point the
measure at it:

```yaml
inputs:
  probe: {from_scan: {scan: 14, statistic: mean}}   # a probe-only scan of the same day, converted
measure:
  kind: haso
  # ...as above...
  reference: probe
```

The reference frame goes through the recipe's steps exactly as each
shot does (a dark background is subtracted from both), then the host
computes its raw slopes **once per process** in a fresh engine, keeps
them as a `.has` in a private temporary directory, and every shot's
worker subtracts them (the SDK's `apply_substractor`) *before* the mask
and the filters — the legacy order. The processed phase, slopes, pupil
and the two scalars then describe the difference; `raw_phase` and
`intensity` stay the shot's own, so the unsubtracted wavefront is still
in the store. The subtraction is linear: SDK subtraction and the
difference of two processed phases agree to 2e-3 µm.

What to take as the reference (measured 26_0929, Scan014 probe-only vs
Scan015 plasma, mask rows 175:350, nothing filtered):

- **The same day.** Probe-only shots reconstruct 0.004 µm RMS from their
  own mean (the method's floor); the probe drifts 0.028 µm RMS from one
  day to the next — half an imprint (0.067 µm RMS, 0.34 µm PV).
- **What to leave out of it decides what the map shows.** A reference
  with the jet off leaves the neutral gas (+) *and* the plasma channel
  (−) in the map; a reference with the **jet firing and the drive laser
  blocked** leaves the plasma alone.
- **Filters remove signal.** Curvature and astigmatism removal eats a
  plasma lens and makes the map depend on the mask window; the HTU
  recipe removes nothing, and anything removed can be removed afterwards
  (the reconstruction is linear).

## The install

One home for the SDK: `software/WaveKit/` on the data share (its
`README.md` has the provenance and the sensor registry). The repository
holds no vendor file; `third_party_sdks/` is gone.

```
WaveKit/
  wavekit_43/                   the SDK: wavekit_py/, dlls/x64/, Examples/, Documentation/
  configs/                      per-sensor .dat (the licence + calibration) and .lift
  python-3.8.10-embed-amd64/    64-bit Windows Python 3.8 with the SDK's numpy 1.19.1 unzipped in
  reference/<set>/              golden shots: .himg + Windows sidecars + reference.json
```

Client `config.ini` of the analysis host's service account:

```ini
[Paths]
wavekit_sdk_path = /mnt/<share>/software/WaveKit/wavekit_43
wavekit_python_path = /mnt/<share>/software/WaveKit/python-3.8.10-embed-amd64/python.exe
wavekit_configs_path = /mnt/<share>/software/WaveKit/configs
wavekit_launcher = env WINEDEBUG=-all wine            # add WINEPREFIX=<dir> for a prefix of its own
```

### The Wine recipe (Linux)

- **Wine 6.0.3** (Ubuntu 22.04's `wine64` + `wine32`), the release the
  port was verified on. Its C runtime lacks `fetestexcept`, which is why
  the SDK's own Python 3.8 + numpy 1.19.1 pair is used — numpy 1.2x
  crashes at import. A newer Wine may lift that, but nothing has been
  run on one; the doctor warns and the parity check decides.
- A **64-bit prefix** (`WINEARCH=win64`; a prefix created as `win32`
  cannot run a 64-bit `python.exe`). The launcher names it with
  `WINEPREFIX=`, or leaves Wine's default `~/.wine` — which on Ubuntu is
  win64 and runs the FROG worker's 32-bit Python too, so the two vendor
  programs share it on the HTU host.
- The engine extracts each sensor's slopes DLL from its `.dat` into the
  prefix's common application-data folder, `Imagine Optic/Core engine/`
  under `drive_c/ProgramData/` (a Windows 7+ prefix) or
  `drive_c/users/Public/Application Data/` (an XP-style one, the HTU
  host's `~/.wine`); the directory must exist or the engine fails with
  "cannot open file for writing". The doctor creates both.
- No dongle: the sensor's `.dat` is the licence for image → slopes →
  phase.

`geecs-wavekit-doctor` (ImageAnalysis, in the Portal's env) does all of
this and reports:

```
$ geecs-wavekit-doctor
[ OK ] config.ini names the SDK, the Windows Python and the sensor configurations
[ OK ] wine 6.0.3 (the verified release)
[ OK ] Wine prefix /home/<account>/.wine
[ OK ] created the engine directory /home/<account>/.wine/drive_c/ProgramData/Imagine Optic/Core engine
[ OK ] created the engine directory /home/<account>/.wine/drive_c/users/Public/Application Data/Imagine Optic/Core engine
[ OK ] vendor sample: HASO serial 4229, slopes grid (…), 3.1 s
[ OK ] 26_0310_Scan012: 2 shots, raw phase / processed phase / intensity float32-exact against the Windows results
OK: 0 failure(s), 0 warning(s)
```

`--no-create` only reports; `--skip-sample` / `--skip-reference` skip the
self-tests; `--reference DIR` points at another set of golden folders.

### Self-tests

1. **Licence-free**: the SDK's HASO3 sample (`Examples/DATAS`) through
   our worker — every install can run it, lab files or not (~3 s).
2. **Parity**: each `reference/<set>/` with a `reference.json` (the
   settings that produced its sidecars: `sensor_config`, `mask`,
   `filters`, `wavelength_nm`, `start_subpupil`, `zonal_prefs`) is
   recomputed shot by shot and its raw phase, processed phase and
   intensity compared with the Windows TSVs **in float32**: 100 % exactly
   equal, NaN pattern included. (A float64 text comparison shows fake
   ~4e-6 differences — the text round-trip, not the SDK.) The `.has`
   files are not compared: a few printf half-way roundings differ
   between Windows and Wine while the binary values are identical.

### The engine-state rule

`compute_slopes` carries the spot tracker's state from call to call
within one engine: a shot that follows a *differently processed* image
(a raw frame after a background-subtracted one) reconstructs
differently. The worker therefore makes a **fresh engine per shot**;
the ~6 s of engine start and LIFT calibration are the price of
reproducibility, and a pooled run (`scan.workers`) pays it in parallel.
MKL takes every core by default, so the host gives each concurrent shot
its share (`MKL_NUM_THREADS`) — a 4-core box runs one shot in ~20 s.

## Adding a sensor

1. Copy its `.dat` (and `.lift` for a LIFT model) into
   `WaveKit/configs/` and add a row to the share README's registry.
2. Write a recipe naming the `.dat` in `sensor_config`; the engine
   refuses an image whose embedded serial is not the file's, so a wrong
   pairing fails loudly (the worker reports both serials).
3. Convert one scan, run the recipe, and — if a Windows result exists —
   add a `reference/<set>/` with `reference.json` so the doctor keeps the
   pairing honest.

## Where things are

| What | Where |
|---|---|
| The measure (`HasoSpec`, packaging, scalars) | `GEECS-Analysis/geecs_analysis/measures/haso.py` |
| The engine service and the worker | `ImageAnalysis/image_analysis/algorithms/haso_wavekit.py`, `_wavekit_worker.py` (Python 3.8 syntax) |
| The doctor | `ImageAnalysis/image_analysis/algorithms/wavekit_doctor.py` (`geecs-wavekit-doctor`) |
| The host side: service factory, stack-only source, the wavefront store | `ScanAnalysis/scan_analysis/core_services.py`, `core_source.py`, `core_sink.ShotStore` |
| The `.himg` codec and the stack converter | `GEECS-Data-Utils/geecs_data_utils/io/himg.py`, `io/himg_stack.py` |
| Config keys | `[Paths] wavekit_*` — [Getting started](../tutorials/getting_started.md) |
