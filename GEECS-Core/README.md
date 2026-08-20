# GEECS-Core

The GEECS access library: everything a Python consumer of GEECS devices
needs, and nothing else.

- **`geecs_core.transport`** — the GEECS UDP/TCP wire protocol, asyncio-native
  and stdlib-only: `GeecsUdpClient` (get/set with ACK + exe-reply correlation)
  and `GeecsTcpSubscriber` (framed `Wait>>` push subscriptions).
- **`geecs_core.db`** — `GeecsDb`, the experiment MySQL database client
  (device endpoints, variable metadata, experiment rosters; credentials via
  the standard `~/.config/geecs_python_api/config.ini` chain).
- **`geecs_core.pv_naming`** — the one shared GEECS→EPICS PV naming policy.
- **`geecs_core.exceptions`** — the one `GeecsError` tree.
- **`geecs_core.testing`** — `FakeGeecsServer`/`FakeGeecsDevice`, an
  in-process server speaking the real wire protocol for offline tests.
- **`geecs_core.client`** *(from 0.2)* — `GeecsDevice`, the entry-level
  synchronous get/set/subscribe client for scripts and notebooks.

Consumers: `GeecsCAGateway` (the CA soft-IOC), `GeecsPvaGateway` (the
distributed image server), `GeecsBluesky` (DB metadata, naming, exceptions),
`GEECS-Console`, and end-user scripts (via `client`). See `DESIGN.md` for the
layering rules and what does *not* belong here.

```bash
cd GEECS-Core
poetry install
poetry run pytest tests -q   # offline — no hardware, no lab network
```
