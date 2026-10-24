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

`--read-only` makes every page a viewer: nothing writes to a device.

## Bench devices and per-device actions

`--bench PATH` adds positioners on bench IOCs that the registry does not list
(a bench prefix, or a device the GEECS DB still owns, which the scan worker
must not see), and sets the panel's options per device:

```yaml
devices:
  U_ESP302_01:                    # the page's name
    kind: epics_motor
    prefix: "undulator:esptest1:" # the bench IOC's prefix
    axes: [m1, m2, m3]
    allow: [move, tweak, stop]    # actuators on mirror mounts: no home, no velocity
    note: >-
      The ESP302's model-2 driver polls only during a move it started:
      between moves the readback is the last reading, and a move made at
      the controller's front panel is not seen until a record processes.
```

`allow` names the panel actions the device permits: for a positioner,
`move` (`.VAL`), `tweak` (`.TWV`, `.TWF`/`.TWR`), `stop` (`.STOP`),
`home` (`.HOMF`/`.HOMR`) and `velocity` (`.VELO`). A device that permits a
motion must permit `stop`. The page leaves out the controls of the
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
