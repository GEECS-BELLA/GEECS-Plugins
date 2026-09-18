# Every non-scalar device type over PVA — rollout brief

*Drafted 2026-09-16 from a code audit plus live DB and wire probes against the
reference deployment (HTU / `Undulator`, no beam). Status: scoping document —
nothing here is built. The optimization arc spawned this and deliberately
excluded it: serving every non-scalar device type over PVA is a gateway + DB
job of its own, and the magspec-spectrum optimizer was dropped from that arc
because of it.*

**The headline.** "90% of the work is already done" is right, and the missing
10% is smaller than the brief's first draft assumed. Every non-Point-Grey
**image** variable on a deployed camera server is already served, decoded and
streaming at 1 Hz today, file plugin included. Every **array** variable that
matters is also already on the wire — *remotely*, in one of two clean formats —
and the reason our probes first saw nothing for the scope traces is a **DB
typing problem**: the `1darray`-typed picoscope variables are dead names, while
the variables the device really publishes are typed `string`. What is left is:
four boxes to bootstrap, one worker-side rule to replace, one small decoder
pair in the gateway, and a DB typing decision. **No LabVIEW work is required
for the picoscope.**

---

## 1. What is already built (and is type-agnostic)

Everything in the PVA gateway except the eligibility predicate and the decode
call is indifferent to what a variable contains:

- per-variable gated subscriptions (`onFirstConnect`/`onLastDisconnect`),
  supervision with backoff + DB endpoint re-resolve (#854), the `:connected`
  NTEnum, the PV-name collision guard, instance version/heartbeat/restart PVs,
  the fleet roster + generated Phoebus screen, `geecs-pva-gateway diff`;
- the **file plugin** (#806): the whole `NDFileHDF5IO` PV set, the writer
  thread, dedupe on `acq_timestamp`, the stale watermark and `Rewind`, the
  per-frame scalar attributes, the reconciliation counters;
- the **worker side**: `GeecsDetector` + stock `ADHDFDataLogic`, the stream
  resource/datum documents, Tiled registration;
- the **read side**: `geecs_data_utils.io.scan_stack` and the ImageAnalysis
  `ShotRef` / ScanAnalysis stack-mapping branches.

Two lines decide what gets any of it:
`geecs_core.db.variable_types.image_variables(rows)` (`effective_vartype ==
"image"`) for eligibility, and `decode_imaq_image_string` for the payload.
`SKIP_VARTYPES = {"image", "1darray"}` is where `1darray` sits today —
"not scalar CA data (served over PVA, if at all)".

## 2. The census (live DB, `Undulator`, enabled devices, 2026-09-16)

110 enabled devices; **55** expose at least one `image` or `1darray` variable.

| devicetype | n | image vars | `1darray` vars | host status |
|---|---|---|---|---|
| Point Grey Camera | 40 | `image`, `bakground image`, `processed image` | `HorizontalLineout`, `VerticalLineout`, `lineouts` | 39 on fleet boxes, **1 not** (`UC_Stretcher_MI`, 192.168.6.66) |
| MagSpecCamera | 3 | `Image`, `ImageInterp` | `AngleAxis`, `EnergyAxis` | all on a fleet box (192.168.8.201) |
| ThorlabsWFS | 2 | `Image`, `SpotfieldImage` | — | 1 on a fleet box, **1 not** (192.168.8.208) |
| MagSpecStitcher | 1 | `Image` (+ `interpSpec`/`interpDiv` as native TSV files, no DB variable) | — | **not** a fleet box (192.168.7.203) |
| FROG | 1 | `SpatialImage`, `frogTrace`, `retrieved FrogTrace`, `retrievedFrogTrace` | 6 (`spectrum x/y`, `temporal …`) | **not** a fleet box (192.168.6.73) |
| PicoscopeV2 | 2 | — | `ScopeTraces`, `wfm`, `wfm info` (**dead names**); the real ones — `scopeTrace.Channel0…3`, `scopeTraceGUI.Channel0…3` — are typed **`string`**, so no type-driven walk sees them (§3.1) | **not** a fleet box (192.168.7.168) |
| HamamatsuSpectrometerDAQ | 1 | — | `counts`, `wavelength`, `wavelengtharray`, `AllAcqCounts` | **OUT OF SCOPE** (Sam, 2026-09-17): no `acq_timestamp` implemented, so it is not truly triggerable yet |
| DaqPad_NI6009 | 4 | — | `AI_array.Channel 0…31` | **not** fleet boxes (4 hosts) |
| HexapodPI | 1 | — | `xyzuvw`, `xyzuvw_tolerances` | not a fleet box (192.168.8.198) |

Notes that matter for scoping:

- **No HASO wavefront device is enabled** (`U_HASO_Filters` is a filter wheel),
  so the proprietary `.himg/.has` case is not live and is not in this arc.
- **No non-scalar variable anywhere is marked `get='yes'`.** The gateway
  subscribes by explicit name, so the flag is irrelevant to it — but it means
  the DB says nothing about which of these the lab wants (Q2).
- 7 hosts carry *only* `1darray` devices: 192.168.7.142, .7.168, .7.169,
  .7.171, 192.168.8.198, .8.217, .8.218 — each a **new gateway box**
  (bootstrap + NSSM), not a code change. But see §3: the array formats ship
  remotely, so a box is not always needed.

## 3. The wire truth (live probes, remote subscriber from a dev machine)

A remote `GeecsTcpSubscriber` holding each variable as a text variable, plus a
12 s PVA monitor on the served image PVs. Payload lengths are the last update's.

| device / variable | DB type | remote push | payload |
|---|---|---|---|
| `UC_Amp4_IR_input:image` (Point Grey) | image | **empty** (local-only, Phase-0 2026-08-27) | full 600×600 via the on-box gateway |
| `UC_Amp4_IR_input` `lineouts`, `HorizontalLineout` | 1darray | **empty**, every update | out of scope (Sam, 2026-09-16: ignore Point Grey lineouts) |
| `UC_BCaveMagSpecCam1` `Image` / `ImageInterp` | image | — | **live over PVA now: 160×719 and 189×1, 13 distinct stamps in 12 s** |
| `U_GhostWFS` `Image` / `SpotfieldImage` | image | — | **live over PVA now: 35×47 and 1080×1440** |
| `U_FROG_Grenouille` `frogTrace` | image | **full, 484 KB** | IMAQ flatten, name-repeat wrapper; **decodes to (576, 768) uint8** with the existing decoder |
| `U_FROG_Grenouille` `SpatialImage`, `retrieved*`, spectra | image / 1darray | **empty** | the FROG pushes only `frogTrace` |
| `U_BCaveMagSpec` `Image` (stitcher) | image | **empty** | local-only; needs an on-box gateway |
| `U_HamaSpectro` `counts`, `wavelength` | 1darray | **full, ~36 KB each** | **plain ASCII CSV**, CRLF-terminated |
| `U_BCaveICT` `ScopeTraces`, `wfm`, `wfm info` | 1darray | **empty** | these names are never published — dead DB rows |
| `U_BCaveICT` `scopeTrace.Channel0/1`, `scopeTraceGUI.Channel0/1` | **string** | **full, 6092 B, fresh every push** | **LabVIEW flattened waveform** (spec below) |
| `U_HP_Daq` `AI_array.Channel 0` | 1darray | 102 B, fresh every push | the same waveform format, `actualSamples=2` (the DAQ is configured for 2 samples — nothing is truncated) |
| `UC_BCaveMagSpecCam*` `EnergyAxis` | 1darray | 11 B | **a single scalar** (`1.204437E-2`) — DB mistyping, or the axis *step* |
| `U_Hexapod` `xyzuvw` | 1darray | 74 B | 6 ASCII values at ~5 Hz — a live position vector, not per-shot data |

### 3.1 The scope-trace finding (supersedes "this is a LabVIEW issue")

The LabVIEW diagram writes two CVT entries per channel — `scopeTrace.Channel<N>`
and `scopeTraceGUI.Channel<N>` — from the cluster `actualSamples /
relativeInitialX / xIncrement / offset / gain`. Probed live on `U_BCaveICT`:
**both are published, remotely, byte-identical, with fresh content on every
push** (6/6 distinct payloads in 6 s). The `Enabled` case in the diagram is
*on*. Nothing is missing on the LabVIEW side.

What is wrong is the **database's typing**, and it is worth stating precisely
because it invalidates the type-driven census this brief started from:

- `devicetype_variable` for `PicoscopeV2` carries **`ScopeTraces` (id 4839),
  `wfm` and `wfm info` typed `1darray`** (`choice_id` 3) — and the device
  publishes none of those names. They are dead rows.
- It **also** carries the eight names the device really publishes —
  `scopeTrace.Channel0…3` and `scopeTraceGUI.Channel0…3` — typed
  **`string`** (`choice_id` 2). So `effective_vartype` calls a 6 KB binary
  waveform a scalar string, and no `image`/`1darray` walk can find it.
- `PXIdigitizer` has the same eight names typed **`1darray`** (ids 94/95,
  392–397, 411–418) — the correct precedent, on a devicetype with no enabled
  device. `RohdeSchwarz_RTA4000` repeats the `PicoscopeV2` mistake
  (`scopeTrace.Channel0…2` typed `string`, ids 5360–5362).

**Latent hazard from the same rows.** The CA gateway serves `string` as a
native DBR_STRING channel, which EPICS caps at 40 characters (`path` exists as
a char-array type precisely because of that cap —
`GeecsCAGateway/geecs_ca_gateway/channels.py:62`). A string-typed trace is not
served today only because it is not `get='yes'`
(`from_geecs_experiment(subscribed_only=True)`); flipping that flag on one of
these rows would publish a truncated 40-byte fragment of a waveform as if it
were a value. Worth a guard when Tier 5 touches these rows.

**Consequence for the design:** eligibility cannot be driven by the DB type
alone in the state the DB is in. Either the rows get retyped (Tier 5, with the
`PXIdigitizer` rows as the model) or the gateway carries a per-devicetype
allowlist of array variables. **Retyping is the better answer** — it keeps the
one-rule doctrine of `variable_types.py` and fixes the CA hazard at the same
time — but it is the owner's call (Q2), and the allowlist is the fallback if
DB edits are slow.

### 3.2 The GEECS waveform wire format (verified byte-exact, two device types)

```
"<actualSamples>,<relativeInitialX>,<xIncrement>,<offset>,<gain>,<name>|"
  + uint32 big-endian sample count
  + count × int16 big-endian raw samples
volts[i] = offset + gain * raw[i]       t[i] = relativeInitialX + i * xIncrement
```

Verified with `leftover_bytes == 0` and `total_len == expected_len` on
`U_BCaveICT` `scopeTrace.Channel0` (3000 samples, `dt` = 4 ns → a 12 µs record,
±1.2 mV with no beam) and on `U_HP_Daq` `AI_array.Channel 0` (2 samples). This
is one format shared by `PicoscopeV2` and `DaqPad_NI6009`, and it carries its
own axis and scaling — everything a trace needs, self-describing, no DB
metadata required.

### 3.3 The three families, after the probes

- **A — IMAQ binary image**: Point Grey, MagSpecCamera (both variables),
  ThorlabsWFS (both), MagSpecStitcher, FROG `frogTrace`. **Zero gateway code**;
  served wherever an instance runs on the device's host.
- **B1 — ASCII CSV array**: Hamamatsu `counts`/`wavelength`/`wavelengtharray` (the format still matters; the device itself is out of scope — no `acq_timestamp`).
  A one-line decoder. **Ships remotely.**
- **B2 — LabVIEW flattened waveform** (§3.2): PicoscopeV2, DaqPad_NI6009. A
  ~15-line decoder. **Ships remotely.**
- **C — genuinely absent**: Point Grey lineouts (out of scope), FROG
  `SpatialImage` / `retrieved*` / spectra, MagSpecStitcher `Image` (local-only
  image, so it is really a Tier-2 deployment item). Family C has shrunk to
  things nobody is asking for.

**A superseded belief, recorded so it is not re-derived:** the 2026-08-27
capture probes concluded that GEECS images "ship only to local subscribers",
from Point Grey evidence alone. The FROG disproves it as a general rule —
484 KB of flattened image to a remote subscriber. **Whether a payload reaches
a remote subscriber is a property of the individual LabVIEW device driver, not
of the payload kind.** Probe the wire per device; never infer it.

## 4. What "rolling it out" actually consists of

### 4.1 Tier 1 — images on hosts that already run a gateway: done, verify only

`MagSpecCamera` ×3 and `U_GhostWFS` are served today, `:hdf1:` plugins present.
The only work is **evidence**: a scan with those devices in the save set, then
`geecs-pva-gateway diff` per camera family — the same parity gate every camera
family passes before `Compression=zlib` / PNG retirement (#738).

### 4.2 Tier 2 — four boxes missing from the fleet: deployment, no code

192.168.6.66 (`UC_Stretcher_MI`), 192.168.6.73 (FROG), 192.168.7.203
(MagSpecStitcher), 192.168.8.208 (low-power WFS). Each: `deploy/bootstrap.ps1`
in a console session, add to `[pva] addr_list` **and** `file_plugin_addr_list`,
regenerate the Phoebus fleet screen, restart the worker. The `DEPLOYMENT.md`
runbook unchanged, ~1 box per short session.

### 4.3 Tier 3 — capture more than one variable per device

Two real gaps, both in `geecs_bluesky/namespace.py`:

1. **`primary_image_variable` picks exactly one variable** — `image` if the DB
   lists it, else the alphabetically first. Verified consequences:
   - FROG resolves to **`SpatialImage`**, which is empty on the wire, while
     `frogTrace` — the only variable it pushes, and the device's actual
     measurement — is not captured. A plugin armed on `SpatialImage` waits out
     `ARM_TIMEOUT_S` on every prepare. **The FROG cannot be rolled out without
     fixing this**, and the docstring's justification ("only the primary one is
     pushed on every acquisition") is exactly what is false here.
   - `MagSpecCamera` captures `Image` but not `ImageInterp`; `ThorlabsWFS`
     captures `Image` but not `SpotfieldImage`. Both second variables stream
     live and both are wanted by analysis.
   The fix is a per-devicetype (or per-device) declaration of which variables
   are capture streams, replacing the positional guess. The plugin, the
   detector and the documents are already per-variable lists — a selection
   change, not an architecture change.
2. ~~`U_BCaveMagSpec` (MagSpecStitcher) is not `looks_triggerable`~~ —
   **closed by #934** (merged into master 2026-09-17, `namespace.py`): the
   stitcher now reads `triggerable=True`, so it gets a `GeecsDetector` and can
   be plugin-backed once its box serves PVA. Verified live against the DB.
   Nothing about the stitcher is gated on #756 any more.

### 4.4 Tier 4 — arrays in the gateway (families B1 + B2), small and bounded

- `geecs_core.db.variable_types`: an `array_variables(rows)` twin of
  `image_variables`; `SKIP_VARTYPES` keeps its CA meaning while the PVA side
  stops treating the two alike.
- `config.py`: eligibility becomes "image vars **or** array vars";
  `CameraSpec` grows an array-variable list (and `CameraSpec` stops being an
  honest name — rename in the same PR).
- `server.py`: a decode branch per family — B1 `np.fromstring(sep=",")`, B2 the
  §3.2 unpack — posted through the same `NTNDArray` with **1-D dims**. Gating,
  supervision, `:connected`, timestamps and latest-wins are untouched. A B2
  payload's `offset`/`gain`/`xIncrement` should be applied on the gateway (post
  volts, not raw counts) and the axis parameters carried as NTNDArray
  attributes or companion PVs — **an open design call, §6 Q5.**
- `file_plugin.py`: relax `frame.ndim != 2` to accept 1-D and map
  `ArraySizeX_RBV = M`, `ArraySizeY_RBV = 0`. **Nothing downstream changes**:
  `get_ndarray_resource_info` filters zero-valued dimensions
  (`ophyd_async/epics/adcore/_data_logic.py:83`), so a 1-D frame is a legal
  areaDetector array and the stack becomes `(N, M)` with no read-side edit.
- Consumers: `scan_stack.read_shot` returns an `(M,)` row; ScanAnalysis's
  `Array1DScanAnalyzer` path and `Data1DConfig` need a stack branch beside the
  `tsv` / `tdms_scope` file types — the one non-trivial read-side item.

**Recommendation: serve 1-D as a 1-D `NTNDArray`, not `NTScalarArray`.** It
reuses the whole areaDetector pipeline (plugin, documents, Tiled, stack
reader), the stock data logic already supports it, and Phoebus plots it. An
`NTScalarArray` would be more EPICS-idiomatic for a waveform and would buy a
second, parallel capture path — not worth it.

### 4.4b The magspec lineouts: shape, padding and the ceiling

Settled with Sam 2026-09-17, after wiring `testarray` on `UC_BCaveMagSpecCam1`
and probing it live. This is the concrete design for `interpSpec` / `interpDiv`
(the names chosen to match the existing native-file asset fields in
`geecs_bluesky/assets/registry.py`).

**Confirmed on the real variables (2026-09-18).** `interpDiv` and `interpSpec`
are live on `UC_BCaveMagSpecCam1` in this exact format. `interpDiv`: **189 x 2**,
column 0 the angle axis -23.5..+23.5 at a uniform 0.25 step, fixed by camera
geometry. `interpSpec`: **1 x 2** with the magnet off, **285 x 2** with a
simulated field (energy axis 51.56..122.56 MeV at a 0.25 MeV step) - span/`dE`,
exactly as described below. Both changed on every push. A device restart was
needed before the new variables appeared over TCP: the first start after a
wiring change writes the names to the DB, and only the *next* start serves them
(Sam) - so "DB row present, name absent from the push frame" is an expected
intermediate state, not a fault.

**The axis is uniform within a shot and moves between shots.** An earlier draft
of this section said a non-uniform axis rides in column 0; that is wrong in a
way worth correcting. The *physical* energy axis is non-linear, which is
precisely why the device interpolates onto a **linear `dE` grid** - so what
arrives is uniformly spaced, with a shot-dependent start, stop and length. The
practical consequence is unchanged (no single axis can be shared across shots,
so it must be stored per shot) and conservative rebinning is *easier* than it
would be for a ragged grid.

**The wire shape - a fourth array format, and the best of them.** `testarray`
(the same lineout under a test name) arrived as **189 rows x 2 columns of
nested-bracket ASCII**:
`[[-2.350000E+1,0.000000E+0], [-2.325000E+1,0.000000E+0], ...]`, 5197 B,
column 0 an axis (-23.5..+23.5, uniform step 0.25 - the same 189 rows as this
camera's `ImageInterp`), column 1 the value. One variable carries **both the
axis and the values**, so a non-uniform axis needs no second variable and no
cluster: the shape a cluster expansion was reaching for, without the
expansion. It is also already 2-D, so **the file plugin stores it with no
change** - its `ndim != 2` requirement (`file_plugin.py:764`) passes and the
stack is `(N_shots, M, 2)`. The 1-D relaxation of Tier 4 is needed only for
the Hamamatsu's flat CSV and the scope's flattened waveform.

**Why the length varies (Sam).** The energy axis is computed from the
Hall-probe-measured field: the energy span across the chip is divided into
fixed `dE` steps, because the energy axis is non-linear and the interp image
needs a linear grid. So the length is `span / dE`, **not** a function of pixel
count - as the spectrometer current falls the span grows without bound, and a
small `dE` makes the axis arbitrarily long. Three regimes: constant (~99% of
the time); +-1 point of drift mid-scan as the field wanders (rare, but real);
and a deliberate **current scan**, where it changes a lot *within one run*.

**The decision: pad in the gateway to a fixed ceiling of 2048 rows, NaN fill.**
Upstream of both consumers, so the PV shape is constant (what Phoebus and the
Bluesky descriptor both want) and the writer needs no change. `float64`, not
`float32`: the wire carries 7 significant figures, which `float32` only just
covers, and the NaN tail compresses to nothing under the stacks' existing
shuffle+gzip - 189x2 is ~3 KB/shot, padded ~16 KB raw, against 7.9 MB for one
raw camera frame. The plugin will need a float `NDDataType` (areaDetector has
Float64); today it only ever sees uint8/uint16.

**A 1 x 2 payload is a valid value and must never crash anything (Sam,
2026-09-18).** It is the shape `interpSpec` defaults to when the magnet is off
- observed live - so it will occur routinely, including *mid-scan* if a magnet
trips. The padding design absorbs it for free: a 1-row frame pads to
`(2048, 2)` like any other, so the PV shape, the descriptor and the stack shape
are all unchanged and no frame is dropped. Without padding, a mid-scan
285 -> 1 -> 285 transition would fail every frame after the trip against the
session's frozen shape. Three guards, each explicit:

1. **The decoder accepts `n >= 1` rows.** A single row is not an error. (The
   first parser written in this session raised on exactly this case - a real
   warning, not a hypothetical.)
2. **Padding treats `n = 1` as ordinary.** No special case, no counter.
3. **Analysis treats `n < 2` as no-data**, not as a spectrum: with one point
   there is no bin width, so any rebinning or `dE` computation must return
   NaN / skip rather than divide by zero. Counting non-NaN rows distinguishes
   "one valid row plus padding" from a real spectrum.

**2048 is a policy ceiling, not a physical one** (an earlier draft of this
section claimed the chip width bounded it - wrong, the interp grid is
decoupled from pixel count). Therefore: **a frame longer than the ceiling is
dropped, counted and named in `WriteStatus`/`WriteMessage`** - the existing
`shape_errors` path (`file_plugin.py:775`) - and **never truncated**. A
truncated spectrum is indistinguishable downstream from a real spectrum that
ends early, which is strictly worse than a missing one. The regime that
reaches the ceiling (small `dE`, large span) already bogs down and can crash
the LabVIEW device itself, so it is not an operating point worth a ragged
storage format; a loud drop there costs nothing we would otherwise have.

**Why not a ragged format at all.** HDF5 could do it (resizable inner dims, or
true VLEN), but the binding constraint is upstream of the file: a Bluesky
**descriptor declares a fixed shape per data key at `describe()`**, before the
run, and a StreamResource carries one shape and chunk shape. Tiled,
`scan_stack.read_shot` and the analyzers all assume rectangular. VLEN also
loses the normal filter pipeline (heap-stored data) and is not expressible in
the ophyd-async stream model. Leaving the container entirely (parquet list
columns, zarr ragged) would mean giving up the stock data logic, the stream
documents and the Tiled path - the free machinery that makes this small.

**Owed downstream (analysis, not this arc).** During a current scan the
per-shot axes genuinely differ, so cross-shot products need resampling onto a
common grid. This is the argument for keeping the axis per shot in the file
rather than once per scan: only the file can tell the analyzer that the grid
moved. Three notes, settled with Sam 2026-09-17:

1. **It is already handled by degrading, not by breaking.**
   `ScanAnalysis/scan_analysis/analyzers/common/array1d_scan_analysis.py:253-281`
   guards this case explicitly (citing FROG spectral phase): with
   variable-shape lineouts `average_data` returns `None`, the averaged-line
   figure is skipped, and per-shot scalars + the waterfall are kept.
2. **The work belongs in the scan-level aggregation, not in a per-shot
   analyzer.** Per-shot scalars (total charge, peak energy, spread) are
   computed on each shot's own axis and are unaffected. Only the averaged line
   and the per-bin averaged spectra need a grid.
3. **Resampling must conserve charge.** Column 1 is integrated charge *per
   energy bin* - a per-bin quantity, not a sampled function - so linear
   interpolation onto a different `dE` silently changes the total. Convert to a
   density (charge/`dE`), or integrate to a cumulative distribution and
   re-differentiate on the new grid. Naive interpolation yields a plausible
   plot whose integral is wrong, and total charge is the headline number.
   Default common grid: the union span at the **coarsest** `dE` present -
   going finer invents resolution no shot measured.

Unchecked: how the **waterfall** plot handles variable-shape lineouts (the
guard above covers only the averaged line) - look before building.

### 4.5b The FROG: `frogTrace` is the one stream (settled 2026-09-17, Sam)

**The device exposes one image at a time — spatial *or* temporal mode — and
spatial mode is purely for alignment, not acquisition.** So `frogTrace`, the
576x768 uint8 frame found streaming, *is* the temporal image, and it is the
only FROG stream worth capturing. `SpatialImage` should never be declared.

Consequences:

- **The PR 2 declaration for `FROG` is one name: `frogTrace`.** The positional
  rule's failure here was purely alphabetical — `sorted()` puts `SpatialImage`
  first — so the declaration needs no FROG-specific machinery, just the right
  name.
- **The declaration assumes acquisition mode.** In spatial (alignment) mode
  `frogTrace` will not be produced, so a plugin armed on it would wait out
  `ARM_TIMEOUT_S`. That is acceptable: alignment is not a scanning
  configuration. Worth a line in the runbook rather than code.
- **The asset registry over-declares.** It defines both a `Spatial` and a
  `Temporal` PNG asset (`geecs_bluesky/assets/registry.py:361-368`), but the
  device can only be in one mode, so on any real scan one of those two assets
  points at files that do not exist. Not this arc's business, but it is
  probably the reason `SpatialImage` was empty on every probe, and someone
  should prune it when the FROG's native path is next touched.
- The duplicate-looking `retrieved FrogTrace` / `retrievedFrogTrace` DB rows
  are neither needed nor pushed; curate them out when convenient.

### 4.6 Three things this plan does not yet answer

1. **Live display of a 2-column array.** An `NTNDArray` of `(2048, 2)` will
   not plot as an XY trace in Phoebus — its image widget renders a two-pixel
   picture. Someone will want to watch a spectrum live within a day of it
   being served. Options: post the columns as two additional 1-D PVs, or let
   the scanner/portal render the pair. Decide **before** the PV shape becomes
   a contract, because it is one.
2. **Per-shot data volume from the new streams.** Turning on second streams is
   not free, and the point of #738 is volume and file-count pathology.
   `ImageInterp` (189×1) and the lineouts are noise, but the WFS's
   `SpotfieldImage` measured **1080×1440** — roughly 3 MB/shot at 16-bit,
   comparable to a Point Grey frame, and stored nowhere today. (Shape verified
   live; the dtype is inferred — confirm before enabling.) The capture
   declaration should make enabling a stream a deliberate per-devicetype act
   for exactly this reason.
3. **Camera-server load on the array path is unmeasured.** The 2026-08 figures
   (~9% CPU for 11 concurrent camera streams on one box) covered images.
   Padded arrays are far smaller and a non-event is the expectation, not a
   measurement. Watch a box during the first array rollout rather than
   scheduling a study.

### 4.7 Effort and deployment, honestly

Five PRs. The lab-time cost is unusually low for this arc — almost everything
can be accepted on an ordinary scan day.

| # | Content | Code | Deployment | Sessions |
|---|---|---|---|---|
| 1 | Bootstrap 4 boxes (FROG, stitcher, low-power WFS, `UC_Stretcher_MI`) | ~0 (a regenerated `.bob` + the config lists) | `bootstrap.ps1` in a **console session per box** (session 0 cannot see mapped drives; NSSM owns lifecycle), `[pva] addr_list` (+ `file_plugin_addr_list` for capture), worker restart | one lab afternoon |
| 2 | Capture-stream declaration (replaces `primary_image_variable`) | ~100–200 LOC + tests, `GeecsBluesky` | worker restart | 1–2 |
| 3 | Array support: `array_variables()`, three decoders, padding + ceiling + counters, 1-D and float in the plugin | ~300–400 LOC + tests, `GeecsPvaGateway` + a small `GEECS-Core` patch | fleet restart; **plus a fork** — see below | 1–2 |
| 4 | Record side for arrays: data keys, descriptor shape, Tiled, `scan_stack` for `(N, M, 2)` | ~150–250 LOC across data-utils / ImageAnalysis / ScanAnalysis | worker restart | 1–2 |
| 5 | Analysis: conservative rebinning, un-skip the averaged figure (§4.4b) | ~100 LOC + tests, `ScanAnalysis` | none | 1 |

**≈ 6–9 build sessions, and two or three short restart windows.** The
bootstrapping is the schedule risk, not the work: **the Windows boxes are
physically remote** (Sam, 2026-09-17 — "their hardware is far away, we'll just
have to deal with that"). `bootstrap.ps1` needs an interactive session, so
PR 1's real cost is access, not effort. The question that sets it: does RDP to
these boxes give a session that satisfies that requirement, or does someone
have to stand in front of each one? Answering that before scheduling PR 1 is
worth more than any estimate here.

**The deployment fork in PR 3, now much smaller.** With the Hamamatsu out of
scope, the Picoscope box (.7.168) is the *only* trace host — one box, not two.
Its payloads ship to remote subscribers, so a central instance for trace
devices stays possible (a systemd unit on the services box, `site.env` and
`render_units.sh` entries, a contract-page update), but against a single box
it is no longer obviously cheaper than one more `bootstrap.ps1`. Revisit only
if a second trace host appears.

**Sequencing, given free lab days.** PRs 1 and 2 are independent of the
LabVIEW work and unlock data that is *already streaming* — do them first. PR 3
can be built and accepted against the **Picoscope**, live and unambiguous
today; the magspec lineouts then need only DB rows plus one decode branch. PRs 4 and 5 follow the lineouts landing. **Nothing waits on the
LabVIEW side to start.**

**The one real risk** is PR 2: it changes behaviour for every plugin-backed
camera, all 40 Point Greys included. Default to today's single-variable
behaviour when a devicetype declares nothing, so the change is strictly
additive, and give it a full adversarial review rather than a quick merge.

### 4.5 Tier 5 — the DB naming/typing fix

Strictly cheaper than any code, and Tier 4 is half-useless without it:

- **`PicoscopeV2`: DONE 2026-09-16 (Sam's call, applied live).** The eight
  real rows `scopeTrace.Channel0…3` / `scopeTraceGUI.Channel0…3`
  (`devicetype_variable` ids 4976–4979, 5012, 5013, 5014, 5018) are retyped
  from `string` (`choice_id` 2) to `1darray` (`choice_id` 3, `variabletype`
  `'1darray'`) — the `PXIdigitizer` shape. Verified through
  `GeecsDb.get_device_variables` for both enabled ICTs; rows backed up with
  restore SQL recorded in the session. Blast radius was nil: `set='no'`, no
  `expt_device_variable` rows name them, every other experiment's PicoscopeV2
  device is disabled, and neither gateway served them before or after (CA
  skips `1darray` and only serves `get='yes'`). **This also closes the §3.1
  truncation hazard for these rows.** Caveat: `kHz_PicoScope_TurboICT`
  (disabled, kHzLPA) has per-instance `variable` rows that override the type
  row wholesale, so it keeps `string` until those four rows are fixed too.
- `RohdeSchwarz_RTA4000` repeats the mistake (`scopeTrace.Channel0…2`,
  ids 5360–5362) — no enabled device, deferred by Sam ("the only relevant live
  devicetype is PicoscopeV2").
- The dead `PicoscopeV2` rows `ScopeTraces` (4839), `wfm` (4981), `wfm info`
  (4980) are **still there** — FK-safe to delete (no `variable` row references
  them, no `expt_device_variable` row names them), awaiting a go/no-go.
- `U_PXI_Slow` (disabled) has `scopeTraceGUI.Channel0…2` typed `on,off`
  (`choice_id` 5) — a different flavour of the same mess, deferred.
- **Owed code change:** the defect note in `variable_types.py`'s module
  docstring (which already records the 2026-09-09 filter-wheel defect) should
  gain this one, so the knowledge lives with the rule rather than only in a
  session memory.
- `EnergyAxis` / `AngleAxis` on the magspec cameras: typed `1darray`, pushes
  one number — retype or rename.
- `xyzuvw` on the hexapod: a live position vector, not per-shot data — should
  not be capture-eligible whatever its type says.
- `bakground image` is misspelled in 40 devices' rows (cosmetic, but it is the
  string the eligibility walk sees).

Standing rule from `variable_types.py`: fix the DB, not the rule.

## 5. Phasing

| Phase | Content | Cost |
|---|---|---|
| P1 | Tier 2: bootstrap the four missing boxes (FROG first — it also proves remote-shipping images) | ~1 short session per box |
| P2 | Tier 5: the DB typing fix (+ the Q2 retype-vs-allowlist call), then re-probe the picoscope through the *DB-derived* roster | one session, no code |
| P3 | Tier 3: the capture-stream declaration + FROG/MagSpec/WFS second streams; decide #756 for the stitcher | 1–2 build sessions |
| P4 | Tier 4: arrays end to end — **the ICT (B2) is the acceptance device** (the Hamamatsu is out of scope) | 2 build sessions + 1 read-side |
| P5 | Tier 1 + P1–P4 parity evidence per device family (`geecs-pva-gateway diff`) | passive, rides on scan days |

**Approved by Sam 2026-09-17:** the Tier-3 capture-stream declaration (P3)
and the Tier-4 array support (P4) as written. The earlier draft's P0 ("probe
whether traces exist at all") is **closed by §3** — no lab session is owed
before building. P1 and P2 need no code and
return the most.

## 6. Open questions for the owner

- **Q1 (closed 2026-09-16).** Scope traces *are* published; the DB named the
  wrong variables. No LabVIEW work.
- **Q2 (now the central question).** Retype the DB rows, or carry a
  per-devicetype allowlist in the gateway? §3.1 shows the DB cannot be trusted
  as the eligibility source as it stands (real traces typed `string`, dead
  names typed `1darray`). Retyping is cleaner and fixes the CA hazard;
  an allowlist ships without DB edits. Related: *which* of these should be
  served at all — 32 `AI_array` channels × 4 DaqPads is not something anyone
  wants by default, and `get='yes'` says nothing here. **Sam, 2026-09-16:
  the DaqPad traces are deferred — nobody is known to use them.**
- **Q3 (partly answered).** Sam, 2026-09-16: `frogTrace` is the **raw 8-bit
  trace** (he had been thinking of `retrievedFrogTrace`, which is useless) —
  **enable it**. Remaining: does it carry real content when the FROG is
  actually acquiring? The decode is sound — (576, 768) uint8 — but every pixel
  was 0 or 1 during the probe (no laser in it). Re-check live before calling
  the FROG rollout accepted.
- **Q4 (answered 2026-09-17, Sam).** `interpSpec` / `interpDiv` are **TSV
  files and stay file-only for now — exposing them needs LabVIEW-side work.**
  Verified: they have **zero** rows in `devicetype_variable` *and* in
  `variable` (the only "interp" DB row anywhere is `ImageInterp`, dtv 4623),
  so there is nothing to subscribe to — the save path writes them and the
  wire does not carry them. Both MagSpec devicetypes write them
  (`geecs_bluesky/assets/registry.py`: `_text_array_asset` ×2 each,
  `.txt` with `data_type: "tsv"`), so the stitcher's non-scalar output is
  *one image plus two TSV traces*, not one image. Making them capturable
  needs, in order: a CVT entry on the LabVIEW side, a `devicetype_variable`
  row typed `1darray`, and then nothing else — they would land in family B
  and ride the Tier-4 decoders. Until then `LvNativeFileDataLogic` keeps
  writing them, which is the intended steady state for non-image devices.
- **Q5.** For B2 waveforms: post scaled volts (losing the raw counts) or raw
  counts plus scaling attributes? And where do `xIncrement`/`relativeInitialX`
  live — NTNDArray attributes, companion PVs, or HDF5 dataset attributes? This
  decides what an analyzer needs to reconstruct a time axis.
- **Q6.** `scopeTrace` vs `scopeTraceGUI`: identical today (the GUI decimation
  is a no-op at the current settings). Serve only the full one, or both?

## 7. Not in scope

- PNG / native-file retirement for these device types (#738) — this arc only
  makes dual-write possible for them.
- The optimization arc's live consumption of traces — it unblocks on P4, it
  does not belong here.
- Point Grey lineouts (Sam, 2026-09-16: ignore).
- A central PVA instance for families B1/B2: tempting, since they ship
  remotely and would need no per-box bootstrap, but it reintroduces the
  concentrator the distributed design exists to avoid. Decide it when a B
  device sits on a box nobody wants to bootstrap. With the Hamamatsu and the
  DaqPads both out of scope, that case no longer exists today.
- HASO / `.himg`: no such device is enabled.
