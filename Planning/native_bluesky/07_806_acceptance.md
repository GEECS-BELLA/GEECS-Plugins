# #806 — hardware acceptance of the file plugin (2026-09-11)

Raw record behind the verdict in `06_pva_file_plugin.md` §8, same
conventions as `05_phase1_acceptance.md`: what was done to which host,
what it showed.  Setup: HTU-NoGas, camera `UC_Amp4_IR_input` (camera
server 192.168.6.100, exposure 1 ms), PR #823 at 69a9870a, the production
worker untouched (feature branch, idle), the run driven **in process**
from the staging clone on the worker host.  #823 merged the same day
(0fd767fa); the Tiled read and the fleet roll below followed the merge.

## Runbook (as executed)

1. **Camera server 192.168.6.100** (ssh as the domain account, key auth):
   `venv\Scripts\python -m pip install h5py` (3.16.0 — PyPI reachable
   from the box); the box's `launch.bat` predated GEECS-Core (four-package
   reinstall line, `geecs-core` 0.4.0 in the venv) and was replaced by the
   repo's current one — without it the restart would have crash-looped on
   `geecs_core.pv_naming.hdf_plugin_prefix`.
2. **Share clone** (`Active Version\GEECS-Plugins`, the fleet pin):
   `git checkout phase/04-pva-file-plugin` (its two local edits do not
   overlap the branch); rollback = `git checkout master` there.
3. `pvput undulator:pvagateway:192_168_6_100:restart 1` → **24 s** later
   `version` reads `0.7.0` and `…:image:hdf1:Capture_RBV` exists (false).
4. **UNC visibility as the service** (LocalSystem via the machine
   account): `FilePath = \\192.168.6.12\hdna2\data\…\Scan005\UC_Amp4_IR_input\`
   → `FilePathExists_RBV` **true**; a nonexistent sub-folder → false.
   (A first attempt through the p4p CLI doubled every backslash in the
   shell — use a script, not the CLI, for Windows paths.)
5. **Worker** (`~/deploy-staging/GEECS-Plugins` on the branch,
   `poetry install --extras "ca tiled qserver qs-client"`): the service
   account's `config.ini` gained `[Paths] geecs_pva_plugin_data_base_path =
   \\192.168.6.12\hdna2\data` and `[pva] file_plugin_addr_list =
   192.168.6.100` (backup `config.ini.bak-806`; the production worker's
   code ignores both).
6. `GEECS_HW=1 EPICS_CA_ADDR_LIST=192.168.6.14 EPICS_PVA_ADDR_LIST=192.168.6.100
   … pytest tests/test_806_hardware.py -m hardware -s` from
   `GeecsBluesky/` (configs from the share's main).

## First run — Scan006: FAILED at `prepare` (a rule found, fixed in 69a9870a)

`RemoteError: no frame from UC_Amp4_IR_input bakground image within 8 s`
(and `processed image`).  The camera's DB lists **three** image-typed
variables; the namespace attached a plugin to each, and the two secondary
ones are pushed only when an operation produces them, so their plugins
never saw a frame and the stock `prepare` failed on the put — with the
plugin's reason in the message (the `WriteMessage` note works).  Both
plugins were left with `Capture_RBV` false, `save` read off, the claimed
folder holds only `ScanInfo`, `scan.log` (`finished (fail)`) and the empty
device dir.  **Rule now:** the worker captures the primary image variable
only (`image`, else the first); the gateway still serves every image
variable's PVs.

## Second run — Scan007: PASSED (`1 passed in 63.6 s`, build 23 s, 108 devices)

| | |
|---|---|
| rows / stack | 5 rows; `UC_Amp4_IR_input.h5` `(5, 600, 600)` uint16, chunks `(1, 600, 600)`, 3.9 MB, `finalized` |
| documents | one stream resource (`application/x-hdf5`, dataset `/entry/data/data`, uri `…/Scan007/UC_Amp4_IR_input/UC_Amp4_IR_input.h5`); five stream datums, indices `(0,1) … (4,5)` |
| stamps | stack `3872002807.386 … 3872002811.386` == the rows' `acq_timestamp` == the five PNG names |
| stack check (`scan.log`) | `stack check: uc_amp4_ir_input: 5 frame(s) in UC_Amp4_IR_input.h5 match the rows' stamps` (appended after the plugin finalized) |
| pixel parity | `geecs-pva-gateway diff Scan007` → `pass (matched=5 identical=5 png_only=0 stack_only=0)` — PNG dual-write and the stack are bit-identical |
| cadence | `[1.0, 1.001, 1.0, 0.999]` s — **1 Hz holds** with the count wait (frame → NAS write → `NumCaptured_RBV` monitor) in the loop; wall 14.0 s for 5 shots (first-shot phase + build) |
| plugin counters | `frames_received 10, frames_written 5, duplicates_dropped 1, stale_skipped 4, rewound 0, shape/decode/open/append 0` — the 4 stale are the idle re-pushes before the first fire, the duplicate a re-push of the last shot |
| camera after | `save` reads `off` (the LabVIEW-native logic's `stop` at unstage) |
| Tiled | run registered; the array read first returned **500** (`Refusing to serve file://…/Scan007/…/UC_Amp4_IR_input.h5 because it is outside the readable storage area for this server` — `readable_storage` in the Tiled server's config listed only its own storage dir).  After the server fix below, the `(5, 600, 600)` array read back through Tiled **identical** to the file |

## What the run settles

- The ASSUMED bullets of `03` §7 for #806: lossless, deduped, stale-filtered
  counting within a capture window — **verified**; Tiled's stock adapter
  read of the plugin's file — **verified** once the server allowed the
  data share (below).
- The machine account **can write** the data share over UNC: the stack
  was created and finalized by the service.
- Uncompressed 5 × 720 KB = 3.9 MB against 5 × 235 KB of PNGs: `zlib`
  (gzip-1 + shuffle) is the expected production setting once the diff is
  trusted — a put on `Compression`, no release.

## Settled after the merge (2026-09-11)

1. **Tiled server.** `/mnt/hdna2/data` added under `readable_storage` in
   the server's config and `Environment=HDF5_USE_FILE_LOCKING=FALSE` on
   the `tiled` unit (a `systemctl edit` drop-in, `override.conf`), then
   restarted: Scan007's array reads back through Tiled identical to the
   file.  Both are `site.env`/unit facts and belong to the end-of-branch
   deployment touch (`03` §10.5); until then they live in the drop-in
   and the config file on the host.
2. **Production worker.** `~/qs-checkout` pulled to 0fd767fa, its env
   reinstalled (p4p), `[pva] file_plugin_addr_list` widened to the nine
   camera servers once they were rolled, and `geecs-qserver` restarted
   at 15:00 — the namespace reads the host list at startup, so the
   restart is what made every camera on those boxes plugin-backed.
3. **Fleet roll (PR #824, GeecsPvaGateway 0.7.1).** What the first box
   taught: the boxes predating GEECS-Core carried a `launch.bat` whose
   reinstall line lacked `geecs-core`, so a bare `:restart` on the new
   code crash-loops; and `h5py` is a bootstrap-time dependency.  #824
   makes both a mechanism — `deploy/requirements-fleet.txt` (the pinned
   closure, installed `--no-deps` on both sides), `deploy/stage_wheels.sh`
   (wheels staged under `<Active Version>/pva-wheels`), and a launcher
   that installs the pins offline before the reinstall.  All nine
   gateways (192.168.6.80, 6.100, 7.161–7.164, 8.197, 8.199, 8.201) went
   to 0.7.1 with their `:hdf1:` PVs between 14:55 and 14:59 by
   stop-service / copy-launcher / start-service over an elevated ssh
   session per box; `h5py` arrived through the launcher's wheel step, no
   hand install.
   **Lesson (from the review of #824): never copy `launch.bat` over a
   running service and then `:restart`** — `cmd` resumes a batch file by
   *byte offset* in whatever file is now at that path, so the running
   launcher continues at a random line of the new one.  The per-box step
   is stop the service, copy the launcher, start the service.  (`nssm` is
   not on `PATH` on the boxes; `sc` / `Stop-Service` work, and the copy
   needs a session that can read the share — an elevated console or
   elevated ssh.)

## Watch period

Every camera scan now writes a stack beside its PNGs.  Diff the first
scans of each camera family (`geecs-pva-gateway diff ScanNNN`); when the
diffs are clean across families, put `Compression=zlib` (gzip-1 +
shuffle) — a PV put, no release — and re-measure the writer thread's
per-frame cost before leaving it on (`06` §9).
