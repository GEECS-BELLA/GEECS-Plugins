# GEECS-Core — Design

`geecs-core` is the GEECS access **library**: the code every consumer of GEECS
devices needs, and nothing else. It was extracted from `GeecsCAGateway`
(2026-08-20) when a fourth consumer (the thin `GeecsDevice` client replacing
the deprecated GEECS-PythonAPI) made the "library parts inside a service
package" arrangement untenable — installing a thin TCP wrapper should not drag
caproto along, and the gateway's own DESIGN.md had carried this extraction as
a known gap since the transport first moved there.

## What lives here

```
geecs_core/
  exceptions.py     # contract: the one GeecsError tree (stdlib-only)
  pv_naming.py      # contract: GEECS→PV naming rules (stdlib-only) —
                    #   producer (gateways) and consumers (Bluesky) import
                    #   THIS module; naming must never drift by copy
  transport/        # layer 1 — the UDP/TCP wire protocol, asyncio-native,
                    #   stdlib-only (GeecsUdpClient, GeecsTcpSubscriber)
  db/               # layer 2 — the experiment MySQL database (GeecsDb,
                    #   blocking, lazy mysql-connector) + alarms.py (the
                    #   pydantic model for the ca_alarm_limits table) +
                    #   the three DB rules every consumer shares:
                    #   variable_types (a variable's effective type),
                    #   scalar_policy (a device's subscribed get='yes' list)
                    #   and device_streams (which non-scalar variables a
                    #   devicetype captures / never serves)
  client/           # layer 3 — the entry-level synchronous GeecsDevice
                    #   over layers 1+2, and the one place a background
                    #   event loop bridges sync callers to the async
                    #   transport
  testing/          # FakeGeecsServer / FakeGeecsDevice — an in-process
                    #   UDP+TCP server speaking the real wire protocol, so
                    #   every consumer can test offline
```

## The three rules

1. **Dependencies flow strictly downward.** `client` imports `transport` and
   `db`; `transport` and `db` import only the root contracts; nothing in this
   package imports `client`. External consumers (the gateways, GeecsBluesky,
   GeecsScanner) use layers 1–2 and the contracts; only end-user scripts use
   `client`. A change that wants an upward or sideways import is in the wrong
   place.

2. **One sync/async bridge point.** `transport/` is pure asyncio and owns no
   threads or loops; `db/` is plain blocking calls. `client/` is the only
   module allowed to bridge the two (its shared background loop). Services
   with their own event loop (the gateways) consume `transport` natively and
   must never touch `client`.

3. **Admission rule.** New code belongs here only if *every* consumer of GEECS
   devices needs it. Gateway config models, derived channels, CA/PVA channel
   machinery, PV serving → the gateway packages. Scan orchestration, ophyd
   devices → GeecsBluesky. Anything analysis- or scan-folder-shaped →
   ScanAnalysis/data-utils. When in doubt, leave it out — this package's value
   is what it refuses to contain. One recorded exception: `db/alarms.py`
   carries CA alarm *evaluation* logic whose only consumer is the CA gateway —
   it rides here because `AlarmLimits` is `GeecsDb.get_ca_alarm_limits`'s
   return type and splitting the model from its own methods would be worse.
   A second, admitted on the `scalar_policy` precedent: `db/device_streams.py`
   (which non-scalar variables a devicetype captures, and which arrays it
   never serves) is read by the worker and by the PVA gateway — two
   consumers that may not import each other; the CA gateway serves scalars
   only and never reads it.

Two supporting conventions:

- **Import hygiene**: `import geecs_core.transport` must stay stdlib-only. The
  package `__init__` re-exports the exception tree eagerly (stdlib) and
  everything heavier lazily — do not add eager imports of `db` or `client`
  there.
- **Configuration**: credentials resolve via the fleet-standard chain
  `~/.config/geecs_python_api/config.ini` → `[Paths] geecs_data` →
  `{geecs_data}/Configurations.INI` `[Database]`. The directory name is a
  historical fossil of the package that first defined it — it is a
  fleet-installed contract; do not rename it. `GEECS-Data-Utils` keeps its own
  independent read of the *paths* half — deliberately not unified, because
  data-utils must stay dependency-free and analysis machines legitimately have
  the paths half without lab-network access.

## Wire-protocol knowledge

The protocol quirks (exe-reply correlation, the `nval,`/`nvar` frame anchors,
`"no error,"` status, lossy numeric coercion, local-IP detection) are
documented on the transport modules themselves and pinned by this package's
tests; the operational history behind them lives in
`GeecsCAGateway/CLAUDE.md` ("Wire-protocol quirks that bit us").

## What the DB's columns actually mean

Three tables describe a variable, and their columns are easy to confuse —
this section exists because a reading of them cost a day (2026-09-22).

**`devicetype_variable` / `variable` — `set` is "user settable".** It says
whether the variable can be changed **live during operations**. Some settings
are configured once and never set live — a communications route, a channel
enable on a scope — and those are not user settable. It is not a scan
concept, and it has nothing to do with whether anything can *read* the
variable. `variable` is the per-instance override and replaces its
devicetype row **wholesale** (`_merge_variable_rows`).

**`devicetype_variable` / `variable` — `defaultvalue` is the configured
value.** For a variable that is not set live, this *is* the device's state:
`PicoscopeV2`'s `Enable.Ch<X>` is which channels are wired, which is why the
capture gate reads it (`db/device_streams.py`). Resolve it the same way as
everything else: instance row if present, else the devicetype default.

**`expt_device_variable` — only `get` matters.** It says the experiment
subscribes this variable, which is what makes the CA gateway serve a PV for
it. Its `set`, `startvalue` and `endvalue` describe scan-boundary writes that
**Master Control** performs; nothing in the Bluesky path reads them, and they
should not be taken as a statement about the variable. A row whose `set` is
`no` carries no live meaning in its value columns at all.

**Do not infer device state from a served PV that the device never pushes.**
A subscribed variable the device does not include in its push frame leaves
its PV at the initial value — for an enum, index 0, which is whatever the
first choice happens to be. On `Enable.Ch<X>` (choices `on,off`) that reads
`on` for every channel, wired or not. The DB row is the honest source for a
configuration fact; a readback is the honest source only for something the
device actually publishes.
