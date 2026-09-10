# Phase 0 — hardware measurements

Raw records behind the verdicts in `03_clean_room_rebuild.md` §7 and the
facts in §11. Each measurement names what it was for, what was written to
hardware, and what it showed. Read-only CA monitors unless stated.

## M1 — OFF latency, timeout-event posting, single-shot stamp offsets (2026-09-09)

**Purpose.** Retire three §7 ASSUMED items: does OFF stop edges and how
fast; does the CA gateway post the devices' 1.5 s timeout events (unchanged
stamp); what are the per-device drain offsets for one known shot (the §4.F
calibration, in prototype form).

**Method.** From the Mac over VPN (rtt ≈ 22 ms). `camonitor` on
`acq_timestamp` of all 55 roster-triggerable Undulator devices
(`looks_triggerable`); 3.5 s warm-up in STANDBY (edges free-running at
1 Hz); `caput` `Trigger.Source:SP` → *Single shot external rising edges*
(the OFF/ARMED source); 6 s observation; `caput`
`Trigger.ExecuteSingleShot:SP` → *on*; 4 s observation; restore *External
rising edges*. HTU-NoGas semantics (no amplitude variable).

**Findings.**

1. **OFF stops edges within one period.** The put completed in 155 ms;
   every live device then received exactly one more stamp (+1.000 s — the
   edge already in flight, arriving 0.02–0.48 s after the put), then
   **silence for the remaining ~5.5 s**.
2. **The CA gateway posts no timeout events.** Zero unchanged-value updates
   reached any monitor during OFF. Whatever the device sends at its 1.5 s
   timeout does not appear on the `acq_timestamp` PV. Consequences: monitor
   silence cannot be read as liveness (use the `CONNECTED` PV, which
   exists); the "positive miss signal" idea in §11.2 is dead unless the
   gateway changes.
3. **A single shot fires on the *next external edge*, not on the put.** The
   `ExecuteSingleShot` put completed in 203 ms; stamps arrived 0.96–1.48 s
   after the put started (one slow device at 2.48 s — its *stamp* agreed
   with the others to 8 ms; only its message was late, exactly the §11.4
   stamp-vs-arrival distinction). So the strict per-shot budget is: fire put
   (~200 ms) + wait for the next 1 Hz edge + drain (≤ 220 ms) + transport.
   **1 Hz strict is reachable only if the software between stamp arrival and
   the next fire put completing stays under ~550 ms.**
4. **Drain offsets span 0–220 ms**, wider than the ~100 ms estimate: the
   spectrometers and MagSpec cameras 0–9 ms, the VISA line 37–62 ms, the
   Amp cameras 47–104 ms, several transport/diagnostic cameras 150–220 ms
   (`UC_OAPin2` 220, `UC_TopView` 182, `UC_FinalSteeringLeak` 178). All
   well inside period/2, so offset-corrected rounding has ≥ 280 ms margin
   at 1 Hz. Measured once; repeat across a few shots before storing as
   calibration (§4.F).
5. **Incidental:** `UC_BCaveMagSpecCam2` stepped +2.000 on the in-flight
   edge — it had missed the previous shot (the ~1 % drop, seen live).
   `UC_Stretcher_MI` stayed silent on the in-flight edge yet caught the
   single shot. 13 of 55 triggerable devices were connected but not
   stamping in STANDBY (idle acquirers: ICTs, DAQ, WFS, gauges, plungers).

**Raw output.**

```
roster: 107 devices, 55 triggerable
alive+stamping in STANDBY: 42/55: ['UC_ALineEBeam2', 'UC_ALineEBeam3', 'UC_ALineEbeam1', 'UC_Amp2Depletion_South', 'UC_Amp2_IR_input', 'UC_Amp3_IR_input', 'UC_Amp4Depletion_South', 'UC_Amp4_IR_input', 'UC_Amp4_IR_output', 'UC_BCaveIn', 'UC_BCaveMagSpecCam1', 'UC_BCaveMagSpecCam2', 'UC_ChicaneSlit', 'UC_DMSurface', 'UC_DiagnosticsPhosphor', 'UC_ExpanderIn1', 'UC_ExpanderIn1_Pulsed', 'UC_FinalSteeringLeak', 'UC_GaiaMode', 'UC_GratingMode', 'UC_ModeImager', 'UC_OAPin1', 'UC_OAPin2', 'UC_Phosphor1', 'UC_Stretcher', 'UC_Stretcher_MI', 'UC_TC_Output', 'UC_TargetIn', 'UC_TopView', 'UC_TubeIn', 'UC_UndulatorImagingSpec', 'UC_UndulatorRad2', 'UC_VisaEBeam1', 'UC_VisaEBeam2', 'UC_VisaEBeam3', 'UC_VisaEBeam4', 'UC_VisaEBeam5', 'UC_VisaEBeam6', 'UC_VisaEBeam7', 'UC_VisaEBeam8', 'U_148Spectrometer', 'U_HamaSpectro']
connected but not stamping: ['UC_BCaveMagSpecCam3', 'UC_GhostFocus', 'UC_GhostUpstream', 'U_Aline3Filter', 'U_BCaveICT', 'U_FROG_Grenouille', 'U_GhostWFS', 'U_HP_Daq', 'U_UndulatorExitICT', 'U_VacuumGauge', 'U_VisaPlungers']
Trigger.Source before: 'External rising edges'
OFF put completed in 155 ms
--- 6 s in OFF: updates after the put, per device
  UC_ALineEBeam2               1 upd: +0.05s new(+1.000)
  UC_ALineEBeam3               1 upd: +0.05s new(+1.000)
  UC_ALineEbeam1               1 upd: +0.02s new(+1.000)
  UC_Amp2Depletion_South       1 upd: +0.02s new(+0.999)
  UC_Amp2_IR_input             1 upd: +0.02s new(+0.999)
  UC_Amp3_IR_input             1 upd: +0.19s new(+1.000)
  UC_Amp4Depletion_South       1 upd: +0.05s new(+1.000)
  UC_Amp4_IR_input             1 upd: +0.05s new(+1.000)
  UC_Amp4_IR_output            1 upd: +0.09s new(+1.000)
  UC_BCaveIn                   1 upd: +0.48s new(+1.000)
  UC_BCaveMagSpecCam1          1 upd: +0.05s new(+1.000)
  UC_BCaveMagSpecCam2          1 upd: +0.31s new(+2.000)
  UC_ChicaneSlit               1 upd: +0.05s new(+1.000)
  UC_DMSurface                 1 upd: +0.16s new(+1.000)
  UC_DiagnosticsPhosphor       1 upd: +0.19s new(+1.000)
  UC_ExpanderIn1               1 upd: +0.02s new(+0.999)
  UC_ExpanderIn1_Pulsed        1 upd: +0.38s new(+1.000)
  UC_FinalSteeringLeak         1 upd: +0.13s new(+0.999)
  UC_GaiaMode                  1 upd: +0.05s new(+1.000)
  UC_GratingMode               1 upd: +0.06s new(+1.000)
  UC_ModeImager                1 upd: +0.19s new(+0.999)
  UC_OAPin1                    1 upd: +0.11s new(+0.999)
  UC_OAPin2                    1 upd: +0.26s new(+0.999)
  UC_Phosphor1                 1 upd: +0.19s new(+1.000)
  UC_Stretcher                 1 upd: +0.02s new(+1.000)
  UC_Stretcher_MI              0 upd: (silent)
  UC_TC_Output                 1 upd: +0.23s new(+1.000)
  UC_TargetIn                  1 upd: +0.23s new(+1.000)
  UC_TopView                   1 upd: +0.21s new(+1.000)
  UC_TubeIn                    1 upd: +0.05s new(+0.999)
  UC_UndulatorImagingSpec      1 upd: +0.05s new(+0.989)
  UC_UndulatorRad2             1 upd: +0.05s new(+1.000)
  UC_VisaEBeam1                1 upd: +0.02s new(+1.000)
  UC_VisaEBeam2                1 upd: +0.02s new(+1.000)
  UC_VisaEBeam3                1 upd: +0.02s new(+1.000)
  UC_VisaEBeam4                1 upd: +0.02s new(+0.999)
  UC_VisaEBeam5                1 upd: +0.02s new(+1.000)
  UC_VisaEBeam6                1 upd: +0.02s new(+0.999)
  UC_VisaEBeam7                1 upd: +0.02s new(+1.000)
  UC_VisaEBeam8                1 upd: +0.02s new(+1.000)
  U_148Spectrometer            1 upd: +0.02s new(+1.001)
  U_HamaSpectro                1 upd: +0.05s new(+1.000)
--- SINGLESHOT put completed in 203 ms; waiting 4 s for stamps
  device                       arrival ms   stamp-min ms
  UC_Stretcher_MI                     959           28.0
  UC_VisaEBeam8                       997           37.0
  UC_Stretcher                        998           44.0
  U_148Spectrometer                   998            0.0
  UC_VisaEBeam3                       998           57.0
  UC_VisaEBeam6                       998           43.0
  UC_VisaEBeam2                       998           57.0
  UC_VisaEBeam5                       998           43.0
  UC_VisaEBeam7                       998           43.0
  UC_VisaEBeam4                       998           57.0
  UC_VisaEBeam1                       998           62.0
  UC_ALineEbeam1                      998           47.0
  UC_ExpanderIn1                      998           60.0
  UC_Amp2Depletion_South              998           47.0
  UC_Amp2_IR_input                    998           53.0
  UC_Amp4Depletion_South             1041           65.0
  UC_UndulatorImagingSpec            1041           63.0
  U_HamaSpectro                      1041            5.0
  UC_UndulatorRad2                   1041           73.0
  UC_ALineEBeam3                     1041           46.0
  UC_BCaveMagSpecCam1                1041            9.0
  UC_TubeIn                          1041           76.0
  UC_Amp4_IR_input                   1041           66.0
  UC_ChicaneSlit                     1041           61.0
  UC_GaiaMode                        1041           72.0
  UC_ALineEBeam2                     1041           44.0
  UC_GratingMode                     1068           81.0
  UC_Amp4_IR_output                  1091          104.0
  UC_OAPin1                          1117          151.0
  UC_FinalSteeringLeak               1177          178.0
  UC_DMSurface                       1178          162.0
  UC_ModeImager                      1178          160.0
  UC_Phosphor1                       1178          151.0
  UC_DiagnosticsPhosphor             1178          151.0
  UC_Amp3_IR_input                   1178          104.0
  UC_TopView                         1204          182.0
  UC_TargetIn                        1226          177.0
  UC_TC_Output                       1226          115.0
  UC_OAPin2                          1270          220.0
  UC_ExpanderIn1_Pulsed              1385           56.0
  UC_BCaveIn                         1476           27.0
  UC_BCaveMagSpecCam2                2478            8.0
Trigger.Source restored: 'External rising edges'
```

## M2 — phase-0 hardware acceptance: one camera as a StandardDetector (2026-09-09, Scan 065)

**Purpose.** Retire §7's "acquire/data split can be satisfied by a GEECS
camera" and "a step scan composes" assumptions; measure the strict per-shot
cadence; prove the detector's own lifecycle brackets native saving.

**Method.** `tests/test_phase0_hardware.py` on the worker host
(`~/deploy-staging/GEECS-Plugins`, branch `phase/00-one-camera-detector`,
commit 0f931c3f). `GeecsDetector("UC_Amp4_IR_input")` with 8 DB-subscribed
scalars + `LvNativeFileDataLogic` on a `StaticPathProvider` into a claimed
scan folder; `ShotControl.from_profile(HTU-NoGas)`; `bps.mv(shot_control,
"ARMED")` → stock `bp.count([cam], 3, per_shot=geecs_per_shot)` → stock
`bp.list_scan([cam], U_S1H.Current, [-1, -0.5, 0, 0.5, 1],
per_step=geecs_per_step)` → finalize: STANDBY + restore the setpoint.

**Findings.**

1. **Both stock plans ran strict GEECS scans unchanged.** 3 + 5 events,
   `exit_status: success`, every row carrying the camera's stamp, its 8
   scalars, `save_path`, and (scan) the magnet readback within tolerance
   (−0.9998 … 0.9999 A). Setpoint restored to 8e-05 A; trigger back in
   STANDBY; `save` read `off` (enum `['on', 'off']`, index 1).
2. **Native saving bracketed by the detector's lifecycle.** `Scan065/
   UC_Amp4_IR_input/` holds exactly 8 PNGs, one per event, named with the
   row's stamp (`UC_Amp4_IR_input_3871866045.102.png` ↔
   `acq_timestamp 3871866045.102`); still 8 two minutes later — no orphans
   in STANDBY. `localsavingpath` was written as the Windows share path
   (`Z:\data\...\Scan065\UC_Amp4_IR_input`) by the config.ini mapping.
3. **Cadence.** Count: 0.95 s, 0.99 s between shots — **1 Hz strict
   achieved** with no motor. Scan: exactly 2.000 s per step — every other
   edge. The step's move (`CaMotor` set + tolerance confirm) plus the
   ~200 ms fire put pushes past the ~550 ms budget measured in M1, so the
   single shot lands on the next-next edge. Expected from M1; the plan
   layer can recover it (issue the fire earlier, overlap move and prepare)
   — not a phase-0 concern.
4. The drain offset rides in the descriptor's configuration
   (`uc_amp4_ir_input-drain_offset: 0.0`, uncalibrated).
5. 28 s wall for 8 shots including the DB roster, connects and the 7 s
   first move (0 → −1 A).

**Verdict.** Phase 0 accepted. The three-logic split fits a GEECS camera
without an areaDetector IOC; the strict `take_reading` is the only GEECS
code in the scan path.
