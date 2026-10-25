# GeecsPanels

Device panels: a browser GUI per native-device kind, starting with the
areaDetector camera. A **prototype** for looking at the cameras, trying
things out and asking for features. Where it is deployed in the end is an
open question (`Planning/device_panels.md`).

- `/` lists the experiment's native devices (`native_devices.yaml` in the
  configs repository), grouped by kind.
- `/camera/<device>` is one camera's panel: the live image, with a colour
  map and levels of your own, and the live statistics, each with its age.
  Open several cameras in several tabs.
- `/scope/<device>` is an oscilloscope's panel (geecs-epics' scope contract).
- `/motor/<device>` is a positioner's panel: one card per axis — position
  and setpoint, move and tweak, stop, the soft limits and switches, homing,
  velocity — every word from the EPICS motor record.
- `/supply/<device>` is a power supply's panel: current and voltage
  readbacks, the setpoint beside them, output on/off, the fault and
  interlock bits, the fault reset and the regulation loop (shown only).

`--read-only` makes every page a viewer: nothing writes to a device.

## The PV bridge (`/api`)

The same process is the PV bridge the web repository's pages read
(`Planning/epics_estate.md` §2.3): every device as its family's roles.

- `GET /api/devices` — the registry joined with the experiment's
  `panel_devices.yaml` (and the bench file): each device's family, prefix,
  axes, permitted actions, note, and its roles as it has them.
- `GET /api/families` — the family files (GEECS-Schemas' package data).
- `GET /api/devices/<device>/events` — the device's stream (Server-Sent
  Events, `event: device`): a full `DeviceFrame` on connect, then the
  changed roles each push period (`--push-period`, default 0.25 s); the
  `id:` is the frame's `seq`, and a reconnect's `Last-Event-ID` resumes.
  `?limit=N` closes after N frames.
- `GET /api/devices/<device>/arrays?columns=N` — its arrays (profiles,
  histogram, a scope's traces), read on request and min/max-decimated.
- `GET /api/frame/<device>` — a camera's latest JPEG (the camera page's
  frame route under its `/api` name: one upstream fetch for both).
- `POST /api/put` — one write (`device`, `role`, `axis` for a motor,
  `value`, `operator`, `confirm` for a hazard role) through the bridge's
  gates, answered `accepted`, `refused` or `denied` and journaled either
  way; a write needs the device's stream open (a stale page cannot write).
  `POST /api/devices/<device>/unlock` releases a stuck run lock, naming who.
- `GET /api/journal?entries=N` — the last puts, newest first.

The models are `geecs_panels.contracts` (the stream's frames, published
as `api/events.schema.json`) and `api/openapi.json`.
`fixtures/` holds a minute of each kind's stream recorded from the
geecs-epics sim host (`fixtures/record.py`), which the web repository's
fixture server replays.

## Bench devices and per-device actions

`--bench PATH` adds positioners and power supplies on bench IOCs that the registry does not list
(a bench prefix, or a device the GEECS DB still owns, which the scan worker
must not see), and sets the panel's options per device:

```yaml
devices:
  U_ESP302_01:                    # the page's name, its URL segment: no / ? # % \ or blanks
    kind: epics_motor
    prefix: "undulator:esptest1:" # the bench IOC's prefix
    axes: [m1, m2, m3]
    allow: [move, tweak, stop]    # actuators on mirror mounts: no home, no velocity
    note: >-
      The ESP302's model-2 driver polls only during a move it started:
      between moves the readback is the last reading, and a move made at
      the controller's front panel is not seen until a record processes.
```

A power supply has no shared record set (each family's IOC names its own
PVs), so its entry maps **roles** to PV suffixes under the prefix:

```yaml
  U_ChicaneInner:
    kind: power_supply
    prefix: "undulator:fpstest1:"
    roles:                        # required: current_sp current voltage output_sp output reset
      current_sp: "current_ramp:SP"   # the ramped write; optional: voltage_sp, loop_mode (shown only)
      current: current
      voltage: voltage
      output_sp: "enable_output:SP"
      output: enable_output
      reset: "reset:SP"
      loop_mode: loop_mode
    bits: {fault: fault, interlock_1: interlock_1, interlock_2: interlock_2}
    allow: [current, output, reset]
    note: >-                      # measured 2026-10-24 (PDT) at the unit's slew of 10 A/s
      Steps ring: a 0.5 A step peaked at 0.75 A (+0.34 s) and settled
      within 5 mA by +1.8 s; a step to 0 A undershot to -0.25 A. The
      10 A/s ramp (50 ms for 0.5 A) is faster than the regulation loop,
      so it does not tame small steps.
```

The bench file may instead be a **panel devices** document (GEECS-Schemas'
`PanelDevices`, the form of the configs repository's
`<experiment>/panel_devices.yaml`): each device names its family and only
what differs on it. The per-kind pages and the bridge both read it; a
supply in the older form above has a page but no bridge stream.

```yaml
schema_version: 1
devices:
  U_ChicaneInner:
    family: fastps                # a family file: motor_record, fastps, easydriver, ad_camera_chain, scope_contract
    prefix: "undulator:fpstest1:"
    allow: [current, output, reset]
    note: Steps ring (see above).
  U_ESP302_01:
    family: motor_record
    prefix: "undulator:esptest1:"
    axes: [m1, m2, m3]
    allow: [move, tweak, stop]
```

A setpoint is refused unless it sits inside its record's drive limits
(`DRVL`/`DRVH`, read and distinct), below `HIHI` and above `LOLO` when
their severity is INVALID (the IOC's own refusal; until the severities and
an INVALID one's bound are read, refused): an `ao` clamps a put past `DRVH` and
writes the limit, so the panel's check is the gate that refuses. After each
put the panel reads the record's alarm, so a supply's `#NAK` is an error on
the page, not a silent success.

`allow` names the panel actions the device permits: for a positioner,
`move` (`.VAL`), `tweak` (`.TWV`, `.TWF`/`.TWR`), `stop` (`.STOP`),
`home` (`.HOMF`/`.HOMR`) and `velocity` (`.VELO`), and a device that
permits a motion must permit `stop`; for a power supply, `current`,
`voltage`, `output` and `reset` (each needs its role). The page leaves out the controls of the
other actions and names them, and the server refuses their fields with a
403. Without `allow`, a device keeps every action. The IOC's access
security still has the last word.

## Run it on a laptop

```
cd GeecsPanels
poetry install
poetry run geecs-panels          # http://127.0.0.1:8500
```

It reads `~/.config/geecs_python_api/config.ini`:

- `[Experiment] expt` — the experiment (or pass `--experiment`).
- `[Paths] scanner_config_root_path` — the configs repository, for the
  registry.
- `[epics] ca_addr_list` — must name the camera IOC hosts as well as the
  gateway. An IOC on its own CA port is written `host:port`, e.g.
  `ca_addr_list = 192.168.6.14 192.168.7.166 192.168.7.166:5066`.

Options: `--port`, `--host`, `--max-fps` (default 10), `--stale-after`
(seconds without a new frame before a value is marked stale, default 3).
