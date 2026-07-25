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
- PV names follow the shared contract (`geecs_ca_gateway.pv_naming`):
  `undulator:uc_amp2_ir_input:image`.
- Subscriptions are **gated**: a camera's GEECS TCP subscription starts with
  its first PVA client and stops with its last — unwatched cameras cost the
  LabVIEW device nothing.
- Frames are **latest-wins**: a slow consumer drops stale frames, never
  backlogs. The archival record is the GEECS file path, not this stream.

See `CLAUDE.md` for architecture and `DEPLOYMENT.md` for the Windows camera
server runbook (install, firewall, NSSM service).
