# GEECS Python API Software Interlock Library

The Software Interlock library provides a flexible pipeline for monitoring and flagging device states via the GEECS Python API. This library integrates with Master Control to

## Overview

The library consists of three main components:

1. **Monitor Condition Builders** — Composable factories for common check types
   - `ThresholdCheck`: value vs. threshold comparison
   - `AlignmentCheck`: tolerance-based alignment
   - `MultiCheck`: combines conditions with OR logic
   - `CustomCheck`: arbitrary predicates

2. **DeviceMonitorGroup** — Manages device subscriptions and state access with **built-in staleness detection**

3. **InterlockBuilder** — High-level facade for server setup and lifecycle

**KEY SAFETY FEATURE:** All interlocks automatically fail-safe (return unsafe) if device data is stale (hasn't updated for `staleness_timeout_ms`).

## File Structure

## Operator GUI

Use `InterlockGUI` to display live monitor state and diagnostics while keeping
the existing `InterlockConstructor` and condition builders as the safety
authority. The GUI provides Start, Stop, and an explicit Arm settings action
for checks such as `ConfigurationChangeCheck`.

```python
from geecs_python_api.controls.interlocks import InterlockGUI

gui = InterlockGUI(
   interlock,
   armable_checks=[tapePointing_config, AegaeonCam_config],
   title="Windmill PM Interlock",
)
gui.run()
```

The windmill example is available at
`InterlockScripts/PW_interlocks/windmill_interlock_gui.py`.
