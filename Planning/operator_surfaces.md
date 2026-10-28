# Operator surfaces: what operators have, and what the web estate gives them

Status: design note, 2026-10-26, **provisional until built**. Written from
the maintainer's screenshots and notes of the LabVIEW GEECS GUIs (HTU
Control Center, Master Control, the diagnostics camera GUI) and the
GeecsCanopy exercise (`Planning/geecs_canopy.md`). It names the operator
surfaces, maps each to its owner in the new estate, and proposes the first
slice. The front end is designed from Canopy outward and from these
surfaces; nothing from the parked geecs-web arc carries weight here.
Delete when the surfaces it describes exist (`Planning/README.md`).

## 1. What operators have today

Three kinds of surface, all LabVIEW, each a compiled program:

1. **Device-type front panels** opened for a chosen device: the camera GUI
   (tabs Main, Adv. Controls, Crosshair, Analysis; an image with ROI tools),
   a scope GUI that only displays traces.
2. **Control GUIs**: dense composed screens. The main one per experiment
   (HTU Control Center: ~85 controls in 13 coloured groups by function:
   laser shutters and amplifiers, alignment stages, jet, compressor,
   hexapod, valves, vacuum, spectrometer magnets) plus a few per subsystem
   (chicane, EMQ triplet, steering magnets). Five to ten per experiment.
3. **Master Control**: the entry point. Choose the experiment; then tabs
   Variable/Scan (any device variable: get, set, step, set default; scan
   setup and run with a live parameter plot and Scan#/Shot#), Device
   (device, GUI, host; start and stop), Experiment (GUI launcher, save
   live configuration, presets), Scan Data (per-device timing and replies
   during a scan), Expert (engine knobs); a rail with the interlock state,
   **Devices not on**, **Devices with Errors**, DB and experiment editors;
   a status bar with the last device reply, the state and the error.

Facts from the maintainer that bind the design:

- A struck-through label means the device is **off**: in LabVIEW a device
  is an executable that must be launched (after a reboot) and can crash.
  Off is a process fact. In EPICS an IOC is a service the host runs and
  restarts, with a heartbeat; health is read from that and from the
  channel, not from a process list.
- LabVIEW GEECS uses **one number for set and readback**. The maintainer
  calls this a trap. EPICS separates them as a matter of course (`.VAL`
  and `.RBV`, `X` and `X_RBV`, `:SP` and the readback); the surfaces simply
  keep that baseline.
- "Save Live Configuration" dumps every variable's current value; a
  **preset** is closer to a macro: perform a predefined sequence of sets.
- "Set Default" overwrites the DB default with the current value, and what
  a default means differs by device type: a power-on setting for trigger
  mode or ROI, meaningless for a position. Only an expert knows which.
- Operators split their time about evenly between the control GUIs and
  Master Control's variable get/set, which also serves variables no GUI
  has yet. Master Control is also where systems management starts: why is
  this device not working, which host is it on.
- The two device lists should be empty and usually are not, because
  devices crash and the tools to resolve it are missing. The hope is that
  EPICS plus agents make this robust.

**Habits kept, artifacts dropped.** The screenshots show how operators
work, not what to build. Kept: health at a glance; two lists that should be
empty; one place to start from; a launcher; macros; dense screens; any
variable reachable without a GUI. Dropped, as artifacts of the LabVIEW
architecture: executables as the unit of health; one number for set and
readback; labels unrelated to names; "Set Default" as a per-type
mish-mash; engine and network knobs on an operator screen. When a later
decision cites a screenshot, it should be for a habit, never an artifact.

## 2. The surfaces of the new estate and their owners

| Surface | Successor of | Owner |
|---|---|---|
| Experiment **home** ("Master Control") | MC rail, Experiment tab, scan status | new web package in GEECS-Plugins (§7) over Canopy and the scanner |
| **Screens** (dense composed control GUIs) | the control GUIs | screen files in the configs repo + one generic renderer in that package |
| **Device pages** (camera, scope, motor, supply) | the front panels | the same package; data from the family files |
| **Variable inspector** | MC Variable/Scan get/set | the same package over Raccoon and Hare |
| **Device drill-down** (status, host, IOC, last error, restart) | MC Device tab, the two lists | the home, over IOC heartbeats, the asyn connection flags and the gateway's status PVs |
| Scan console | MC Variable/Scan scan half, Scan Data | GeecsScanner (exists) |
| Scan browser, logbook, config editor | Scan Data, DB and Expt editors | portal, logbook, the portal's editor (exist); no DB editor yet |
| Presets and configurations | Experiment tab | Squirrel (snapshots) and GeecsBluesky action plans (sequences), §5 |

Everything that moves a PV goes through the GEECS device module on Canopy
(operator identity, allow lists, journal, bounds, hazard confirm); the
scanner keeps its own write path to the queueserver.

## 3. The screen file and its renderer

A control GUI is a **file, not a program**: a titled group holds widgets,
each with a display label and a separate binding (device plus family role,
or a bare PV). One renderer draws every file. Labels are free text, so the
LabVIEW quirk of labels unrelated to names becomes explicit and harmless;
the migration cost is one label-to-device mapping per screen, done once.

The widget vocabulary the Control Center needs is seven kinds, each mapped
to a family role type: numeric setpoint (setpoint role; shows the readback
beside it when the family names one, with a visible "differs" state while
settling), two-state toggle (bit or enum), the flipper composite (toggle
plus in/out indicators), latching button (command or bit with confirm),
read-only numeric (readback), annotation text, static image. Groups carry
position and a colour token, never a literal colour.

Per-widget state vocabulary, shared with the home's lists, claiming no
more than the evidence supports: **live**; **stale** (channel up, the
producing counter stopped); **not answering** (the IOC answers, the
hardware behind it does not: the asyn connection flag, a device status
PV); **unreachable** (no connection and no evidence why: the gateway's
`connected` PV is a TCP-subscription fact per `GeecsCAGateway/PV_CONTRACT.md`,
and an absent IOC heartbeat may be a network outage; no restart is
implied); **not running** only when an independent service-status source
confirms the process is stopped (a host or container state the estate does
not expose yet; until it does, such devices read "unreachable");
**error** (alarm severity); **unknown**. The drill-down, not the widget,
is where "unreachable" is resolved into a cause.

Annotations survive migration as text in the file. A note of the form
"shutter laser before switching" becomes a hazard confirm on the widgets it
guards. Density is a requirement: the main screen shows ~85 controls at
once in a compact mode, not a dashboard layout.

## 4. The experiment home

One page, kept familiar: title "Master Control" if the operators prefer.

- **Top strip:** interlock and alarm state, run state and Scan#/Shot# from
  the scanner, the live scan parameter plot while a scan runs.
- **Devices not on** and **Devices with Errors**, as two lists with the
  same names, each row opening the device drill-down.
- **Launcher:** the screens from the configs repo grouped by area, the
  device pages by type with a device picker, the variable inspector, the
  scan console, the portal, the logbook, the config editor.
- **Presets:** the Squirrel groups and action plans for this experiment,
  the active-snapshot verdict, the journal's last commands with operator.

The drill-down is where agents earn their place: a device's row shows its
status, host and IOC, the last error text (the "last slow response"), and
the allow-listed actions (restart its container, re-dial); an "explain"
action hands the same facts to the facility assistant through GEECS-MCP.

## 5. Presets, configurations, defaults

- **Save Live Configuration** = a Squirrel snapshot of the experiment's
  registered PVs. Restore of a group = "send preset" for value sets.
- **Preset as a macro** (an ordered sequence of sets with waits) = an
  **action plan**, which GeecsBluesky already runs (`run_action`) and
  GEECS-Schemas already describes. The home lists both kinds under one
  "Presets" heading; neither needs a custom implementation.
- **Set Default** splits into two mechanisms that LabVIEW conflated, and
  the family files say which applies to a role:
  1. **Autosave** is continuous last-value persistence by the IOC: the
     deployed camera's exposure, trigger mode and analysis ROI are in a
     monitored autosave set, so whatever value was last set, including a
     temporary one, is what boot restores. A family role carries
     `autosaved: true` to say so. Positions are not autosaved. This is
     descriptive; nothing to "save".
  2. **A saved default** is an explicit choice that survives later
     adjustments: a Squirrel snapshot of a device's or an area's settings
     tagged golden, applied deliberately by an operator (or by a module
     action after an IOC restart), through the module's guarded put path.
     Boot precedence is: autosave restores last values, then the golden
     snapshot is applied explicitly if wanted; nothing applies it silently.
  The inspector's "save as default" therefore means "save a golden
  snapshot of these roles", never a write into the IOC's autosave. For DB
  devices the DB default stays what LabVIEW makes of it. This moves the
  per-device-type mish-mash out of experts' heads into the family files
  and the snapshot store, where the next person can read it.

## 6. Device-type front panels

**Camera.** The LabVIEW GUI's features map almost one to one onto the
areaDetector chain the native camera already runs and the page GeecsPanels
already renders: ROI (ROI plugin), exposure, gain, trigger mode and delay,
rate (`ArrayRate_RBV`), last update (freshness by the STAT counter), max
counts against a saturation level (the raw STAT0 maximum), two named
crosshairs with target, size, colour and "set to centroid" (the overlay
plugin, built), centroid, sigma, max, mean, total (the statistics plugin),
mean and histogram and lineout plots, colour table and autoscale (client
side), the save toggle (the scanner's save set), device errors (alarms).
Three real gaps, each a decision rather than a page: a **spatial low-pass
filter** and **perspective correction** are not stock plugins (an analysis
step offline, or a custom plugin in the IOC chain); **FWHM and 45° second
moments** are not statistics-plugin outputs (derive from sigma x, y, xy and
the profiles, in the module or a derived channel). Background subtraction
and rotate/flip exist as plugins and are an IOC-chain decision. The
Analysis tab's per-shot trends are what Vole, Canopy's live buffer, is for.

**Scope.** The LabVIEW panel only shows traces; the GeecsPanels scope page
over the picoscope contract is already the better front panel. Keep it.

**Motor, supply.** Exist as GeecsPanels pages from the family files; they
become the drill-down's device page and the inspector's rich view.

## 7. The renderer stack: decided by a falsifiable first slice

Two candidates, chosen on merit against Hare's frames and the surfaces
above, not on what exists:

- The **Python stack** already used by the scanner, portal and logbook
  (FastAPI, Jinja, the kit, plain JS with a small reactive library),
  screens interpreted at runtime. One toolchain; the risk is whether a
  dense screen with hundreds of live widgets stays pleasant to build.
- A **component framework** (React or similar) with a build step and
  typed clients. Stronger for rich widgets; a second toolchain and build.

The first slice decides: build the Control Center as a screen file plus the
renderer, and the home's top strip and two lists, in the Python stack. If
the dense screen is painful there, the screen files are unchanged and only
the renderer moves to a component framework. Either way the kit's tokens,
status words and overlay ladder apply, and every page is judged by the
maintainer clicking it.

## 8. Where things live

| Content | Home |
|---|---|
| Device-type knowledge (roles, hazards, freshness counters, power-on flags) | family files, package data in GEECS-Schemas |
| An experiment's screens, home layout, presets' groups, label-to-device mappings | the configs repo, `<Experiment>/screens/`, `home.yaml` |
| The renderer, the home, the device pages, the inspector | one new package in GEECS-Plugins, served behind the front door |
| Live PVs, writes, snapshots, directory, archive reads | Canopy with the GEECS module (`Planning/geecs_canopy.md`) |
| Screenshots and notes of the LabVIEW GUIs (lab device names) | the configs repo or the maintainer's notes, never this public repo |

## 9. Migration of the suite

Per screen: the screenshot and the label-to-device mapping go to an agent,
which writes the screen file keeping the arrangement; the maintainer clicks
it against the sim or the live devices; the LabVIEW program retires only
when its screen has run. Five to ten screens per experiment at a day or
less each. Master Control retires last, after the inspector and the
drill-down exist, since it is still the entry point for variables no screen
has and for finding a broken device.

## 10. The first slice

1. The screen format and renderer, proved on the HTU Control Center from
   its screenshot (the mapping supplied by the maintainer), against the
   sim where devices exist there and "unreachable" for the rest.
2. The home's top strip, the two device lists and the launcher.
3. Setpoint-with-readback and the state vocabulary on every widget.

Judged by: an operator recognises the screen and completes the gate's
steps 05 to 09 on it; the two lists match reality when a sim IOC is
stopped; no literal colour and no code per screen.

## 11. Decisions asked of the maintainer

1. Keep the Master Control name and the two list names on the home?
2. Who supplies the label-to-device mapping for the Control Center: you,
   or an agent reading the LabVIEW GUI source, with you checking?
3. The three camera gaps (§6): IOC-chain plugins, module-side derivation,
   or offline analysis only, per gap.
4. The two default mechanisms (§5: descriptive `autosaved` flags in the
   family files; golden snapshots in Squirrel for saved defaults): agree
   in principle?
5. Whether the first slice runs before or after the first `GeecsCanopy`
   package PR (it needs Hare and the module's put; the sim instance on the
   Mac serves it meanwhile).
