# Phase-0 capture probes

Empirical gates for the central PVA image-capture arc
(`Planning/data_capture/01_central_pva_capture_scope.md`). Both scripts are
read-only standalone tools, run from the **GeecsPvaGateway** poetry env
(it carries `geecs-core`, `geecs-ca-gateway`, and `p4p`):

```bash
cd GeecsPvaGateway
poetry run python ../GeecsBluesky/capture/probes/probe_g1_shot_counter.py \
    --experiment Undulator --device UC_TopView --duration 300
poetry run python ../GeecsBluesky/capture/probes/probe_g2_pva_deep_queue.py \
    --experiment Undulator --device UC_TopView --duration 300 --shallow
```

- **G1** (`probe_g1_shot_counter.py`) — subscribes to the device's own TCP
  push stream with `include_shot=True` and records the wire shot counter per
  update. Gaps during a strict scan = LabVIEW-side loss, which no downstream
  work can recover. This is the make-or-break gate.
- **G2** (`probe_g2_pva_deep_queue.py`) — monitors the camera's NTNDArray PV
  with a deep client queue (`record[queueSize=N]`), optionally alongside a
  default shallow monitor. Against G1's counts over the same window it
  separates the two drop stages: gateway latest-wins slot vs p4p client-side
  MonitorFIFO squash. Note the first update per monitor is the gateway's
  cached/placeholder frame — subtract one. Subscribing opens the gateway's
  gate (activates the upstream LabVIEW stream), same as a Phoebus viewer.

Protocol for the decisive run: start G1 and G2 on the same camera, then run
one strict 200-shot scan; compare `G1 image_updates` / distinct shots vs the
scan's shot count vs `G2 deep`/`shallow` counts, and reconcile against the
LV-written per-shot files for the same scan.

Camera-server load during the run (Task Manager on the host, or the PVA
gateway's log) is part of the observation: sustained full-rate subscription
is a new load profile for those boxes.
