# Block 1 Audit: GEECS-Scanner Codebase Analysis

**Branch:** `audit/block1-codebase-analysis`
**Focus:** GEECS-Scanner-GUI and its relationships to surrounding packages

---

## 1. How a Scan Actually Runs (the real picture)

Before observations, it is worth stating plainly what the execution model is, because it is not immediately obvious from the file names.

```
GEECSScanner (Qt window)
  └─ RunControl              ← thin bridge, no Qt
       └─ ScanManager        ← setup / teardown / threading
            └─ ScanStepExecutor  ← the actual scan loop
                 ├─ DeviceManager
                 ├─ DataLogger
                 ├─ ScanDataManager
                 └─ BaseOptimizer (optional)
```

`ScanManager` does not execute the loop itself. It builds the scan steps, calls
`pre_logging_setup()`, then hands a `ScanStepExecutor` to a background thread.
The executor runs `execute_scan_loop()` which iterates scan steps. After the loop
exits, `ScanManager.stop_scan()` runs teardown. This split is good in principle,
but neither class has a documented interface contract and they share mutable state
in ways that make the boundary fuzzy.

---

## 2. Package Responsibilities (actual vs. stated)

| Package | Stated purpose | What it actually does |
|---------|---------------|----------------------|
| `GEECS-PythonAPI` | Device communication | Core device control, DB lookup, UDP/TCP, threading |
| `GEECS-Scanner-GUI` | Scanner GUI | Scanner *engine* + GUI. The engine is the majority of the code. |
| `GEECS-Data-Utils` | Data utilities | Scan path resolution, data loading, **scan types/modes**, **`ScanConfig`** |
| `ScanAnalysis` | Post-scan analysis | Analysis framework with live watch; imports scanner config conventions |
| `ImageAnalysis` | Image analysis algorithms | Standalone; consumed by ScanAnalysis |
| `LogMaker4GoogleDocs` | Google Docs API | Experiment log management; consumed by ScanAnalysis |

**Observation:** The name "GEECS-Scanner-GUI" is misleading — the package contains the
scan engine, not just a GUI. This naming has downstream consequences: the engine
cannot be used without importing the GUI package, and it discourages headless use.

---

## 3. The Config System: Multiple Layers, No Coherent Owner

There are at least four distinct config concepts in play, each with a different
format and location:

### 3a. `ScanConfig` (geecs_data_utils — dataclass)
```python
@dataclass
class ScanConfig:
    scan_mode: ScanMode = ScanMode.NOSCAN
    device_var: Optional[str] = None
    start: Union[int, float] = 0
    end: Union[int, float] = 1
    step: Union[int, float] = 1
    wait_time: float = 1.0
    ...
```
A plain Python `@dataclass`. Lives in `geecs_data_utils` despite being a scanner
concept. No validation — you can construct a `ScanConfig(scan_mode=ScanMode.STANDARD,
device_var=None)` with no error, and the scan will fail at runtime.

`geecs_scanner/data_acquisition/types.py` is a deprecation shim that re-exports
this with a `DeprecationWarning`. The migration happened but the object didn't
become a Pydantic model in the process — a missed opportunity.

### 3b. `SaveDeviceConfig` / `DeviceConfig` (geecs_scanner schemas — Pydantic)
```python
class DeviceConfig(BaseModel):
    synchronous: Optional[bool] = False
    save_nonscalar_data: Optional[bool] = False
    variable_list: Optional[List[str]] = None
    ...
```
Already Pydantic. Lives in `geecs_scanner/data_acquisition/schemas/`. Good pattern
that isn't applied consistently elsewhere.

### 3c. `ActionSequence` (geecs_scanner schemas — Pydantic)
Also already Pydantic. Validated at action execution time. Good.

### 3d. The `config_dictionary` passed to `ScanManager.reinitialize()`
A raw `dict` assembled by the GUI from YAML file reads. Passed directly into the
engine with no schema. The engine unpacks it with dict key access. This is the
biggest gap: the most important config — the one describing what devices to use,
what to save, what actions to run — has no model.

**Summary:** Pydantic exists and is used in the right places for sub-components,
but the top-level scan configuration that drives everything is an unvalidated dict.

---

## 4. The GUI / Engine Boundary

### Where the boundary is clean
`run_control.py` is a reasonable bridge: thin, no Qt imports, translates GUI calls
into engine calls, polls `scan_manager` for progress.

### Where the boundary breaks down

**`app/lib/action_control.py`** — This is the clearest violation. `ActionControl`
lives in `app/lib/` and wraps `ActionManager` for use from the GUI. But it contains:
```python
from PyQt5.QtWidgets import QMessageBox, QApplication

class ActionControl:
    @staticmethod
    def _display_error_message(message: str):
        msg_box = QMessageBox()
        msg_box.exec_()
```
A Qt dialog is being shown from inside what should be an engine-layer adapter.
This is the exact pattern the operator-intervention feature needs to do *correctly*:
the engine signals an error, the GUI decides how to display it. Here they're fused.

**`GEECSScanner` (geecs_scanner.py)** — The main Qt window directly reads engine
state by polling `run_control.get_progress()`, `run_control.is_active()`, and
`run_control.is_stopping()` on a timer. There are no Qt signals from the engine.
The GUI reconstructs what is happening rather than being told.

**`ScanStepExecutor.trigger_on/off`** — The trigger functions are set via attribute
injection (`self.trigger_on_fn = ...` somewhere, discoverable only at runtime via
`hasattr`). This means the trigger interface is implicit and invisible at the class
definition. Any caller that forgets to inject the function gets silent no-ops.

**DB queries in the GUI** — `RunControl.get_database_dict()` calls
`get_database_dict()` from `scan_manager`, which eventually queries the GEECS
MySQL database. This is engine work happening in a GUI-facing method, which is
fine — but it means the GUI initiates database connections on user interaction,
not at startup.

---

## 5. The Scan Lifecycle: State is Implicit

`ScanManager` tracks state through a collection of flags and events:
- `self.initialization_success: bool`
- `self.stop_scanning_thread_event: threading.Event`
- `self.pause_scan_event: threading.Event`
- `self.is_in_setup: bool` (in `RunControl`, not even in `ScanManager`)
- `self.is_in_stopping: bool` (also in `RunControl`)

There is no `ScanState` enum. The valid state combinations are not documented.
It is possible to call `stop_scan()` before `pre_logging_setup()` has finished,
or to call `start_scan_thread()` while a scan is still tearing down, with no
guard. The `is_busy()` method in `RunControl` only checks `is_in_setup`, not the
full range of states in which a new scan should be rejected.

### Pause is partially implemented
`pause_scan_event` is a `threading.Event` used by `ScanStepExecutor`. The executor
correctly blocks on `pause_scan_event.wait()` and resumes when it is set. But
there is no corresponding pause state in `RunControl` or `ScanManager`, and no
event emitted to the GUI when a pause is requested vs. when it takes effect. A
user clicking "pause" and then "start" again before the pause has propagated to
the scan thread could get unexpected behavior.

---

## 6. Error Handling: Three Inconsistent Approaches

Across the codebase there are three distinct patterns for handling `GeecsDeviceCommandRejected`:

**Pattern A — `scan_executor.py` (good pattern, wrong level)**
```python
except GeecsDeviceCommandRejected as e:
    if attempt < max_retries - 1:
        time.sleep(retry_delay)
    else:
        self.stop_scanning_thread_event.set()  # graceful stop
        return
```
This is the closest to correct: retry then trigger a graceful stop. But it lives
in `ScanStepExecutor` — it only applies to mid-scan device moves. Setup, teardown,
and action execution use different patterns.

**Pattern B — `action_manager.py` (catch and warn)**
```python
except GeecsDeviceCommandRejected as e:
    logger.warning("Set %s:%s — command rejected, continuing: %s", ...)
```
Silently continues. Appropriate for closeout, but used uniformly without regard
for whether the command was critical.

**Pattern C — bubbles to caller (crashes)**
Everything else. `configure_device_save_paths()`, `save_hiatus()`, `_set_trigger()`,
and most of the setup path have no handling. A rejection crashes the calling thread.

The deeper problem: there is no way to express "this command is critical" vs. "this
command is best-effort" at the call site. The policy is implicit in the location
of the call rather than explicit in the code.

---

## 7. `ScanMode` and `ScanConfig` in `geecs_data_utils`: A Misfit

`ScanMode` (STANDARD, NOSCAN, OPTIMIZATION, BACKGROUND) and `ScanConfig` are
scanner concepts. They describe how to run a scan, not how to interpret scan data.
They live in `geecs_data_utils` because that package was intended as a "ground truth
for all data structure organization" — but the result is that `ScanAnalysis` and
`LogMaker` import scanner configuration types, creating a dependency that feels wrong
in the other direction.

The `geecs_data_utils` package does have genuinely shareable things:
- `ScanTag` — identifies a scan by date + number. Universally useful.
- `ScanPaths` — resolves folder structure from a tag. Universally useful.
- `ScanData` — loads scan files for analysis. Universally useful.
- `ECSDump` / `DeviceDump` — parses ECS live dump files. Universally useful.

`ScanConfig` and `ScanMode` are not universally useful — they are scanner-execution
concepts. They should live in the scanner engine. `ScanAnalysis` doesn't need to
know how a scan was configured; it only needs to know where its data is.

---

## 8. `save_hiatus`: Two Methods, One Name, Different Meanings

`ScanManager.save_hiatus(hiatus_period)` turns off save for all non-scalar devices,
sleeps, then turns save back on. This is a device-control operation.

`ScanStepExecutor.save_hiatus(duration)` just calls `time.sleep(duration)`. That
is all it does. The name implies device control but it is only a sleep.

In `ScanStepExecutor.wait_for_acquisition()`:
```python
if hiatus and self.data_logger.shot_save_event.is_set():
    self.save_hiatus(hiatus)   # ← calls executor's version: just sleeps
    self.data_logger.shot_save_event.clear()
```
The device-control part of the hiatus (`ScanManager.save_hiatus`) is apparently
not wired into the executor's hiatus path. This is either a bug or the two methods
are intended to be different things with the same name, which is confusing either way.

---

## 9. `ScanStepExecutor`: Good Structure, Implementation Gaps

The existence of `ScanStepExecutor` is a good architectural decision. Separating
"execute one step" from "manage the overall scan" is the right split. But:

- It receives `device_manager`, `data_logger`, `scan_data_manager` as untyped
  constructor arguments — there are no type annotations on the `__init__` parameters.
- `trigger_on_fn` / `trigger_off_fn` are injected via `hasattr` checks rather than
  being declared in the interface. If the trigger is not injected, the scan proceeds
  silently without triggering — this could collect bad data.
- `move_devices()` (the serial version) is still present but commented out in
  `execute_step()`. Dead code in the hot path.
- `results` is stored as `self.results = None` and then written to via the data
  logger — the relationship between `results` in the executor and `results` in
  `scan_manager` is not clear.

---

## 10. The File Pipeline Complexity

Data flows from device → disk → scan folder in several stages:
1. Device saves files locally (e.g. `C:\SharedData\device_name\`) during scan
2. `FileMover` workers watch for new files, match them to scalar shot timestamps,
   rename them, and move them to the scan folder
3. At scan end, orphan detection sweeps for missed files
4. If `save_local=True`, an additional copy to the network drive happens

The `FileMover` uses regex timestamp matching against device filenames. This means
the file pipeline implicitly depends on every camera device following a naming
convention. Any device that doesn't match the pattern silently drops data.

There is also no transactional guarantee: if the process exits between step 2 and
step 3, files can be left in intermediate states. The orphan detection is a best-
effort recovery mechanism, not a guarantee.

---

## 11. Structural Observations on the Broader Ecosystem

**`GEECS-PythonAPI` device subclasses** contain experiment-specific code:
```
geecs_python_api/controls/devices/HTU/   ← Undulator experiment
geecs_python_api/controls/devices/HTT/   ← Thomson experiment
```
Lab-specific code is coupled into a library that should be generic. New experiments
require modifying the library.

**`geecs_data_utils.GeecsPathsConfig`** reads INI files to discover the root data
path for each experiment. These INI files are machine-specific and gitignored.
This is reasonable, but any tool that wants to find scan data must have this
configuration set up correctly — there's no graceful fallback or informative error.

**`ScanAnalysis` and `LogMaker`** are genuinely independent and well-separated.
They consume scan data via `ScanTag` / `ScanPaths` / `ScanData` from `geecs_data_utils`
and do not need to know about scanner internals. This is the right pattern.

**No test infrastructure** for the scanner engine. There are test configs under
`tests/ScannerTools/test_configs/` but no test harness that exercises `ScanManager`
or `ScanStepExecutor` without real hardware.

---

## 12. What a Cohesive System Would Look Like

Given all of the above, here is a concrete restatement of the roadmap with more
grounding in what actually exists:

### Rename / restructure the package
`GEECS-Scanner-GUI` → `GEECS-Scanner` at the repo level, with an internal split:
```
GEECS-Scanner/
  geecs_scanner/
    engine/         ← pure Python, no Qt, independently importable
      scan_manager.py
      scan_executor.py
      device_manager.py
      action_manager.py
      data_logger.py
      scan_data_manager.py
      file_mover.py
    app/            ← Qt only, thin shell
      windows/
      widgets/
      run_control.py
```

### Move `ScanConfig` and `ScanMode` back into the scanner engine
`geecs_data_utils` keeps `ScanTag`, `ScanPaths`, `ScanData`, `ECSDump`. Scanner-
specific types (`ScanConfig`, `ScanMode`) move to `geecs_scanner/engine/`.
`ScanAnalysis` only imports from `geecs_data_utils` (which it already does for data
loading). Nothing breaks in the analysis path.

### Promote `ScanConfig` to a Pydantic model
The current `@dataclass` with no validation means invalid configs are only caught
at runtime. A Pydantic `ScanConfig` would validate that `device_var` is non-None
when `scan_mode=STANDARD`, that `start < end`, etc. The GUI becomes a form that
produces a `ScanConfig` or raises a `ValidationError` before any device connection
is attempted.

### Define a `ScanEvent` protocol
The GUI should not poll. The engine should emit. A minimal set of events covers
all current GUI needs:
```python
@dataclass
class ScanEvent: pass

@dataclass
class ScanStateChanged(ScanEvent):
    old_state: ScanState
    new_state: ScanState

@dataclass
class ScanProgressUpdate(ScanEvent):
    step: int
    total_steps: int
    elapsed_s: float

@dataclass
class ScanCommandError(ScanEvent):
    device: str
    variable: str
    value: Any
    error: Exception
    recoverable: bool
```
`ScanManager` accepts a callback `on_event: Callable[[ScanEvent], None]`.
`RunControl` registers a callback that emits Qt signals. `ActionControl`'s
`_display_error_message` becomes a Qt slot connected to `ScanCommandError`.

### Make the trigger interface explicit
`trigger_on_fn` / `trigger_off_fn` should be constructor parameters or an abstract
base class, not dynamically-injected attributes:
```python
class TriggerInterface(Protocol):
    def trigger_on(self) -> None: ...
    def trigger_off(self) -> None: ...
```
A `NullTrigger` (no-op) can be the default. Missing trigger injection becomes a
type error rather than silent misbehavior.

### Define an explicit `ScanState` enum
```python
class ScanState(Enum):
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    PAUSED_ON_ERROR = "paused_on_error"
    STOPPING = "stopping"
    DONE = "done"
    FAILED = "failed"
```
All state transitions go through a single method. The GUI reflects state without
polling — it receives `ScanStateChanged` events.

### Unify error handling via a command policy
The three current patterns (crash / warn+continue / retry+stop) should be
explicit policies on the `ScanStepExecutor`:
- **CRITICAL**: retry N times; if still failing → `PAUSED_ON_ERROR`
- **BEST_EFFORT**: log warning, continue
The policy is declared per-phase (setup vs. mid-scan move vs. teardown), not
implicit in which exception handler happens to be nearby.

---

## 13. Suggested Sequential Work Blocks (revised)

Given the actual code state, the blocks in priority order:

1. **Pydantic `ScanConfig` + move to engine** — highest leverage; fixes the most
   common source of mysterious runtime failures; unblocks type-checking the engine.
2. **Explicit `ScanState` enum + transitions** — makes the lifecycle auditable;
   prerequisite for the event model.
3. **`ScanEvent` protocol + engine/app split** — removes Qt from engine; enables
   headless use; removes polling from GUI; makes operator-intervention possible.
4. **`TriggerInterface` protocol** — small but high-value; removes implicit
   attribute injection; enables testing with a mock trigger.
5. **Command policy layer** — unifies the three error-handling patterns; enables
   `PAUSED_ON_ERROR` to be reached from any phase.
6. **Operator intervention UI** — the payoff; requires blocks 2–5 to be sound.

---

*End of Block 1 audit. Questions, corrections, and "you missed X" welcome before proceeding to Block 2.*
