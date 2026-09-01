# GEECS Data Portal — scope & design

*Drafted 2026-08-29 from the colleague-onboarding / data-access working
session. Status: scoping document for the `feat/data-portal` arc —
nothing here is built. Decisions marked (Sam) were made in the
originating session; everything else is proposal. This document anchors
the "GEECS Data Portal" name referenced by the Platform fleet map.*

## The feature

A **read-only scan-browsing web service** on the queueserver worker host
(later the consolidated services server). Anyone on the lab network
opens a bookmarked URL — nothing to install — and gets the navigation
colleagues actually think in: day → Scan NNN → metadata, scalar-vs-scan-
variable plots, and a per-device image gallery with a shot slider.

The one-line pitch (Sam's framing of the problem it solves): the new
data stack (Tiled, HDF5 stacks) must never be *less* accessible than
"a bunch of png files in a directory tree". The portal is the bridge —
it makes the catalog-backed data path the *easiest* way to look at
scans, before any format migration asks anyone to change habits.

## What already exists (build on, don't rebuild)

- **`geecs_data_utils.tiled_catalog`** — the catalog engine, already
  below the console: the `ScanCatalog` protocol (`probe()`,
  `list_runs(experiment, day) → list[RunSummary]`,
  `load_run(uid) → RunDetail`), `TiledScanCatalog` (real),
  `StubCatalog` (hermetic tests), `read_tiled_config`. Plus
  `tiled_drift` and `tiled_schema`. The console's Scan Browser
  (`geecs_console/browser/browser_window.py`, ~1.5k lines) is almost
  entirely Qt view code over this protocol — the portal is a second
  view layer over the *same* engine, and the two front-ends cannot
  drift on query semantics because there is one implementation.
- **`geecs_data_utils.io.scan_stack`** — the HDF5 frame-stack reader
  from the capture arc (`Planning/data_capture/`): the read path for
  capture-daemon output.
- **Stranded pure helpers in the Qt module** — `resolve_scan_folder`
  (RunDetail → scan folder path; exactly the join the resource viewer
  needs) and `metadata_rows` are module-level, Qt-free functions in
  `browser_window.py`. They move down into `tiled_catalog` (phase 2)
  so both front-ends share them. Caveat: `resolve_scan_folder` is
  Qt-free but not console-free — its fallback path calls
  `geecs_console.services.ops_paths.todays_scan_folder`, itself a thin
  offline-first wrapper over `ScanPaths.get_daily_scan_folder`, so the
  move re-bases that fallback on data-utils directly and brings the
  scan-folder-invariant pin (`test_browser_scan_folder.py`'s
  tree-untouched test) along into the data-utils suite.
- **The per-shot filename join** — (run, device, shot) → file is not
  just folder + save-path column: the GEECS filename convention and its
  edge handling already live in `geecs_data_utils.scan_paths`
  (`ScanPaths.build_asset_path`, `infer_device_ext`), with production
  usage — including the directory-scan `file_map` fallback for
  nonconforming names — in `geecs_bluesky/assets/readback.py`. Phase 4
  consumes these; it does not re-derive the convention.

## Architecture

```
 colleague's browser ──HTTP──▶  Data Portal (FastAPI, systemd unit)
                                   │                │
                          ScanCatalog (Tiled,   data share mount
                          server-side API key)  (READ-ONLY)
```

Three properties carry the design:

1. **A join, not a store.** The portal owns no data. Tiled answers
   "what scans exist / what happened / where are the files"; the share
   mount serves the bytes. It works against *today's* data — event rows
   already carry save paths, so no write-side migration is a
   prerequisite. When asset serving later moves into Tiled (see
   "Future work"), the portal's fetch plumbing changes behind a stable
   URL; users notice nothing.
2. **Read-only by doctrine (Sam).** The share mounts read-only; the
   portal has no write verbs — no analysis triggering, no annotations
   in v1. That is what makes "anyone on the network" safe with no auth
   story, and keeps the service solo-maintainable.
3. **Python-native, no build chain (Sam, standing doctrine).** FastAPI +
   server-rendered templates. No npm. One `poetry install`, one systemd
   unit, a `deploy/` runbook — the same service pattern as the gateways
   and the qserver. *(Amended in phase 3, flagged for owner veto on the
   scaffold PR: scalar plots are server-rendered matplotlib PNGs, not
   the Plotly originally noted here — control-room machines may lack
   internet for CDN assets and vendoring Plotly is a build-chain smell;
   revisit only if interactivity is wanted.)*  *(Superseded 2026-08-30
   for the analysis-tabs arc: the revisit happened and interactivity IS
   wanted — vendored Plotly.js approved by the owner (one checked-in,
   version-pinned file; still no npm, no CDN — control-room machines
   stay offline-safe). See `02_labview_peruser_inventory.md` §Settled
   architecture. The v1 run page's matplotlib plot stands.)*

### The resource viewer (the genuinely new piece)

Endpoints that take (run, device, shot) → resolve the file path from
the event row (`resolve_scan_folder` + the event's save-path columns) →
serve a browser-displayable image. Server-side normalization is a
deliberate value-add: autoscale/percentile windowing and colormap the
16-bit camera PNGs into 8-bit renditions — the thing generic catalog
UIs never get right. Two loaders behind one endpoint shape:

- native per-shot files (PNG today), read from the share;
- capture-daemon HDF5 stacks via `scan_stack` (one seek per frame —
  the stacks are chunked one frame per chunk).

**Lazy loading is a hard rule**: the NAS pathology is file *count*
(`Planning/data_capture/01…` measurements), so the gallery fetches one
shot at a time on demand — never eager-thumbnail a whole scan from
native files. (Thumbnail strips become cheap for HDF5-stack scans —
one open file — and can land later as a stack-only feature.)

### Asset tiering doctrine (Sam — settled in the originating session)

Two write streams are permanent; unification happens at the **catalog**,
never the writer. The portal renders each tier by capability:

| Tier | Producer / format | Portal behavior |
|---|---|---|
| A | capture daemon → HDF5 stacks | full gallery via `scan_stack` |
| B | native files, Linux-readable (PNG, text arrays, TDMS) | gallery via share read (+ per-format loaders as wanted) |
| C | vendor-SDK-only files (HASO `.himg` — needed indefinitely) | metadata card + copyable file path; no rendering |

A device's card degrades gracefully down the tiers; every device in the
run is *visible* regardless of tier — the failure mode to avoid is two
finding stories, not two formats.

## Deliberately out of scope (v1)

- Any write path (annotations, analysis triggering, re-processing).
- Authentication (lab-network-internal + read-only; revisit only if
  exposure changes).
- Serving assets *through* Tiled — see "Future work".
- Replacing the console Scan Browser — the console remains the
  operator's deep tool; the portal is zero-install casual access.

## Future work (recorded here, not gating)

**StreamResource migration.** `nonscalar_save.py` emits legacy
event-model `Resource`/`Datum` documents, which Tiled stores as metadata
only — the server cannot serve the referenced files. Server-side asset
serving requires: (write side) emitting `StreamResource`/`StreamDatum`
with a mimetype and server-resolvable URI; (server side) mounting the
share on the Tiled host and registering `adapters_by_mimetype` (HDF5
needs zero custom code; PNG/text need thin ports of
`geecs_bluesky/assets/handlers.py`; Tier C stays client-side forever).
First step when picked up: pin what the deployed tiled 0.2.14 /
TiledWriter actually support. The portal deliberately does not wait for
this — it insulates users from it.

## Phase plan (PRs into `feat/data-portal`)

1. **`portal/design-note`** — this document.
2. **`portal/catalog-helpers`** — move `resolve_scan_folder` +
   `metadata_rows` from `browser_window.py` down to
   `geecs_data_utils.tiled_catalog`; console imports updated
   (Data-Utils minor, Console patch). Near-mechanical, behavior
   preserved — includes re-basing `resolve_scan_folder`'s fallback off
   `ops_paths` onto data-utils and moving the tree-untouched invariant
   pin down with it (see "What already exists").
3. **`portal/scaffold`** — new top-level `GEECS-DataPortal/` package
   (per repo precedent for services that join multiple packages:
   GEECS-MCP, the gateways): FastAPI app over `ScanCatalog`, day/experiment
   picker, filterable run list, run detail with metadata + scalar
   plots. Hermetic tests on `StubCatalog`.
4. **`portal/resource-viewer`** — the image endpoints and gallery:
   path resolution, PNG normalization, `scan_stack` loader, lazy
   per-shot fetching, tier-degradation cards.
5. **`portal/deploy`** — `deploy/` systemd unit + `DEPLOYMENT.md`
   runbook; fleet map edit promoting the portal from *planned
   additions* to a service row.

Then the promotion PR `feat/data-portal` → master (constituent PRs each
reviewed → no re-review, per CONTRIBUTING § Branch topology), merged by
the maintainer after a live check against real scans on the lab network.

## Open questions (answers wanted before/during phase 3)

- **Day-listing performance**: `list_runs` latency against the real
  catalog at a full day's scan count — measure early; if slow, the fix
  is a small cache inside the portal, not a schema change.
- **Tiled anonymous read** (`allow_anonymous_access: true`) — separate
  server-config nicety for the stock `/ui`; the portal doesn't need it
  (its API key is server-side). Verify anonymous-cannot-write before
  enabling.
- **Experiment scoping**: single-experiment config per deployment
  (matching the gateway pattern) vs a picker — proposal: config default
  with a picker, since the catalog is experiment-keyed anyway.
