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
                    #   pydantic model for the ca_alarm_limits table)
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
   GEECS-Console) use layers 1–2 and the contracts; only end-user scripts use
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
