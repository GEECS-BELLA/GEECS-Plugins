# Platform

The access-and-contract layer everything else sits on: how you talk to
GEECS devices, how GEECS is exposed to the wider EPICS ecosystem, and how a
scan and its data are described.

<div class="grid cards" markdown>

-   :material-lan-connect:{ .lg .middle } **Python API**

    ---

    The low-level Python interface to the GEECS control system — the
    GEECS wire protocol (UDP command/response, TCP live subscription),
    device and variable objects, the experiment database, and the shared
    [`config.ini`](../geecs_python_api/scripting_guide.md). Most tools use
    it indirectly; the scripting guide covers direct use.

    [:octicons-arrow-right-24: Overview](../geecs_python_api/overview.md) ·
    [Scripting guide](../geecs_python_api/scripting_guide.md)

-   :material-transit-connection-variant:{ .lg .middle } **GEECS Gateway**

    ---

    The GEECS access layer as EPICS services: a central Channel Access
    server mirroring GEECS devices as scalar/control PVs, plus distributed
    pvAccess gateways on the camera servers serving live images as
    NTNDArray PVs — so Phoebus, an Archiver Appliance, or ophyd-async /
    Bluesky can talk to GEECS like any other IOC, no bespoke bridge
    required.

    [:octicons-arrow-right-24: Client overview](../geecs_gateway/client_overview.md) ·
    [Camera images (PVA)](../geecs_gateway/image_pvs.md)

-   :material-file-tree:{ .lg .middle } **GEECS Schemas**

    ---

    The typed contract for how a scan is described: the handful of config
    kinds that drive the engine, in plain language, plus a per-field
    reference generated straight from the code so it can never drift.

    [:octicons-arrow-right-24: Scanner configs](../geecs_schemas/schemas_overview.md) ·
    [Running a scan](../geecs_schemas/running_a_scan.md) ·
    [Schema reference](../geecs_schemas/schema_reference.md)

</div>

!!! note "Python API is under refactoring"

    The Python API is being reworked. Treat `ScanDevice` and the
    experiment-database lookup as the stable public surface; other
    internals may move.
