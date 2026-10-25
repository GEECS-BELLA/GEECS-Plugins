# Device-stream fixtures

A minute of each kind's `GET /api/devices/{name}/events`, recorded from
the geecs-epics sim host (`hosts/sim/compose.yaml`) with `record.py`: one
`DeviceFrame` per line, the first the full frame a connect gets, every
later line a delta. The web repository's fixture server replays them, so a
page can be judged with nothing else running.

| File | Device (sim) | Frames | Deltas/s | Delta bytes (mean / max) | Full frame |
|---|---|---|---|---|---|
| `UC_VisaEBeam6.ndjson` | `undulator:uc_visaebeam6:` (ADAravis over aravis' fake camera, ~25 fps) | 239 | 3.97 | 1718 / 1885 | 10.4 kB |
| `U_SimMotor1.ndjson` | `undulator:simmotor1:` m1, m2 (motorSim) | 97 | 1.60 | 283 / 801 | 5.7 kB |
| `U_EzTest1.ndjson` | `undulator:eztest1:` (the Easy-Driver fake) | 118 | 1.95 | 339 / 547 | 1.4 kB |

Recorded 2026-10-25, push period 0.25 s, while a script moved the sims:
m1 to 3 and back, m2 to −2 and back (each move shows as `running` then
`ok`), the Easy-Driver's current setpoint to 0.5 A and its output on and
off. The camera re-sends its statistics every frame (each advances with
STAT's counter, so its `advanced_at` changes); everything else is sent only
when it changes. The supply's `degraded`, "setpoint refused (CALC)", is the
sim record's own INVALID alarm after the put, reported as Appendix A rule
11 says. The FAST-PS sim (`fpstest1-sim`) did not finish `iocInit` under
emulation in this session, so the supply kind is the Easy-Driver's alone.

Re-record (from this package's environment, the sim host up):

```
python fixtures/record.py --seconds 60 --ca "127.0.0.1:5066 127.0.0.1:5100 127.0.0.1:5068" --device UC_VisaEBeam6=ad_camera_chain:undulator:uc_visaebeam6: --device U_SimMotor1=motor_record:undulator:simmotor1::m1,m2 --device U_EzTest1=easydriver:undulator:eztest1:
```
