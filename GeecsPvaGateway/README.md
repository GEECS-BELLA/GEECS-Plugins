# GeecsPvaGateway

Distributed pvAccess gateway serving GEECS camera images as NTNDArray PVs —
the PVA peer of GeecsCAGateway. One instance runs on each Windows camera
server and serves that host's cameras; the central CA gateway never touches a
pixel (see `GeecsCAGateway/DESIGN.md`, "images stay off CA").

```
LabVIEW GEECS camera device --loopback TCP push--> geecs-pva-gateway --PVA/NTNDArray--> Phoebus / ophyd-async / p4p
```

```bash
geecs-pva-gateway --experiment Undulator          # serve this host's cameras
geecs-pva-gateway --experiment Undulator --list   # show what would be served
```

- Served set is **DB-scoped**: enabled devices whose GEECS endpoint IP is this
  machine and that expose image-typed variables. No per-host config file.
- PV names follow the shared contract (`geecs_core.pv_naming`):
  `undulator:uc_amp2_ir_input:image`.
- Subscriptions are **gated per variable**: each image variable's GEECS TCP
  subscription starts with its first PVA client and stops with its last —
  unwatched variables (and whole unwatched cameras) cost the LabVIEW device
  nothing.
- Frames are **latest-wins**: a slow consumer drops stale frames, never
  backlogs. The archival record is the **file plugin's** (below) or the
  GEECS native file path, not this stream.
- Each image variable also gets an **areaDetector-shaped HDF5 file plugin**
  (`undulator:uc_amp2_ir_input:image:hdf1:` + the `NDFileHDF5` PV names):
  a lossless second consumer of the same frame that writes one
  `<device>.h5` stack per scan into the run folder, driven by the worker's
  stock ophyd-async `ADHDFDataLogic` (#806). Served only where `h5py` is
  installed (a re-bootstrap per box). `geecs-pva-gateway diff <scan
  folder>` compares a scan's stacks against its native PNGs.

See `CLAUDE.md` for architecture and `DEPLOYMENT.md` for the Windows camera
server runbook (install, firewall, NSSM service).
