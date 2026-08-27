# Phase-0 capture probes

Empirical gates for the central PVA image-capture arc
(`Planning/data_capture/01_central_pva_capture_scope.md`). Both scripts are
read-only standalone tools, run from the **GeecsPvaGateway** poetry env
(it carries `geecs-core`, `geecs-ca-gateway`, and `p4p`):

```bash
cd GeecsPvaGateway
poetry run python ../GeecsBluesky/capture/probes/probe_g1_shot_counter.py \
    --experiment Undulator --device UC_Amp4_IR_input --duration 300
poetry run python ../GeecsBluesky/capture/probes/probe_g2_pva_deep_queue.py \
    --experiment Undulator --device UC_Amp4_IR_input --duration 300 --shallow
```

Placement note: this top-level `capture/probes/` directory is standalone
operational tooling (the `qserver/` pattern — not part of the installable
package). The capture *daemon* itself will live in the importable
`geecs_bluesky/capture/` subpackage (the `qs_client` pattern); don't
confuse the two homes.

- **G1** (`probe_g1_shot_counter.py`) — subscribes to the device's own TCP
  push stream (the same feed the PVA gateway consumes) and records every
  update's wire shot counter, `acq_timestamp`, and image-payload size.
- **G2** (`probe_g2_pva_deep_queue.py`) — monitors the camera's NTNDArray PV
  with a deep client queue (`record[queueSize=N]`), optionally alongside a
  default shallow monitor. Subscribing opens the gateway's gate (activates
  the upstream LabVIEW stream), same as a Phoebus viewer.

## Interpreting the numbers (this section is the owner — trust it over
## stale comments elsewhere)

- **Key every comparison on distinct timestamps**, never raw update counts:
  G1's `distinct` acq_timestamps (windowed to the scan) vs G2's
  `distinct_pv_timestamps` (same window, PV time = acq_timestamp minus the
  LabVIEW epoch offset 2082844800) vs the count of LV-written per-shot
  files for the same scan.
- The wire **shot counter is Master-Control-owned and not reliable for us**
  (owner statement, 2026-08-27) — it is recorded for context only.
- G2's **first update per monitor is the gateway's cached stale frame or
  placeholder** — not a real frame.
- After a scan the device resumes its idle 1 Hz push **re-sending the last
  frame with an unchanged acq_timestamp**, which the gateway re-posts —
  raw G2 `updates` therefore exceed frames over any window that outlasts
  the scan. Distinct timestamps are immune.
- **G1 from a remote host receives empty image payloads even during
  acquisition** (measured 2026-08-27: images ship only to host-local
  subscribers). `image_updates` counts non-empty payloads only;
  `empty_payload_updates` counts the present-but-empty ones. From a remote
  host, expect `image_updates == 0` — that is normal, not a fault.
- G1's `interval_*` stats span the whole session (connect ramp and idle
  pushes included) — window offline for scan-cadence questions.
- G1 summary `stream_disconnected_early_at` non-null means the TCP socket
  died before the duration elapsed — the run is invalid as a loss
  measurement; rerun.
- G2 `disconnect_events: 1` on a healthy run is the initial
  not-yet-connected notification. A dead or misnamed PV shows
  `disconnect_events: 1, updates: 0` (note `--image-var` is not validated
  against the DB).

Protocol for a decisive run: start G1 and G2 on the same camera, run one
strict scan, then compare windowed distinct-timestamp counts across G1, G2
deep, G2 shallow, and the LV-written files. Camera-server load during the
run (CPU/RAM via the session-scoped sshd, or the gateway log) is part of
the observation. Phase-0 results from 2026-08-27 (both gates PASS at 1 Hz,
11-camera load a non-event) are recorded in the scope doc.
