# Interlock Abstraction: Quick Start Guide

This guide explains the new interlock utilities that reduce boilerplate while maintaining flexibility for custom monitor logic.

## Overview

The new abstraction consists of three main components:

1. **Monitor Condition Builders** — Composable factories for common check types
   - `ThresholdCheck`: value vs. threshold comparison
   - `AlignmentCheck`: tolerance-based alignment
   - `MultiCheck`: combines conditions with OR logic
   - `CustomCheck`: arbitrary predicates

2. **DeviceMonitorGroup** — Manages device subscriptions and state access with **built-in staleness detection**

3. **InterlockBuilder** — High-level facade for server setup and lifecycle

**KEY SAFETY FEATURE:** All interlocks automatically fail-safe (return unsafe) if device data is stale (hasn't updated for `staleness_timeout_ms`). No separate wrapper needed—it's built in!

---

## Quick Start

### Basic Example

```python
import logging
import time
from geecs_python_api.controls.interlocks import (
    InterlockBuilder,
    ThresholdCheck,
    MultiCheck,
)

# Configure logging (optional)
logging.basicConfig(level=logging.INFO)

# Create builder
builder = InterlockBuilder('BELLA', host='127.0.0.1', port=5001)

# Register device
cam1 = builder.add_device('cam1', 'CAM-PL1-TapeDrivePointing',
    ['MaxCounts', 'MeanCounts', 'acq_timestamp'])

# Create condition builders
max_check = ThresholdCheck(builder.device_group, 'cam1', 'MaxCounts', 4000, '>')
mean_check = ThresholdCheck(builder.device_group, 'cam1', 'MeanCounts', 0, '<')

# Combine conditions
multi_check = MultiCheck([max_check, mean_check])

# Register monitor
builder.add_monitor('Camera Check', multi_check, interval=0.1)

# Start server (logging instead of print)
builder.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    builder.stop()  # Logging instead of print
```

---

## Comparison: Old vs. New

### Old Pattern (~60 lines)

```python
import time
from geecs_python_api.controls.interface.geecs_database import GeecsDatabase
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from geecs_python_api.controls.interlocks.geecs_interlock_server import InterlockServer

# Setup experiment
GeecsDevice.exp_info = GeecsDatabase.collect_exp_info("BELLA")

# Create device
cam1 = GeecsDevice("CAM-PL1-TapeDrivePointing")
cam1.subscribe_var_values(['MaxCounts', 'MeanCounts', 'acq_timestamp'])

# Define factory function with closure state
def camera_thresh_check(camera, variable_name, thresh, timeout=5000):
    last_device_timestamp = None
    last_check_time = time.time()

    def check():
        nonlocal last_device_timestamp, last_check_time
        current_time = time.time()
        device_state = camera.state
        value = device_state[variable_name]
        device_timestamp = device_state["acq_timestamp"]

        if device_timestamp is not None:
            if device_timestamp == last_device_timestamp:
                time_frozen = current_time - last_check_time
                if time_frozen > timeout:
                    print(f"WARNING: No new data for {time_frozen:.1f} seconds")
                    return True
            else:
                last_device_timestamp = device_timestamp
                last_check_time = current_time

        if value is None:
            print(f"WARNING: No {variable_name} data received")
            return True

        return value < thresh

    return check

# Create server and register
server = InterlockServer(host=SERVER_IP, port=SERVER_PORT)
server.register_monitor("Camera Check", camera_thresh_check(cam1, "MaxCounts", 4000), interval=0.1)
server.start()

print("Camera interlock server running...")
try:
    while True:
        time.sleep(0.02)
except KeyboardInterrupt:
    server.stop()
```

### New Pattern (~25 lines)

```python
import logging
import time
from geecs_python_api.controls.interlocks import (
    InterlockBuilder,
    ThresholdCheck,
    MultiCheck,
)

logging.basicConfig(level=logging.INFO)

builder = InterlockBuilder('BELLA', host=SERVER_IP, port=SERVER_PORT)

cam1 = builder.add_device('cam1', 'CAM-PL1-TapeDrivePointing',
    ['MaxCounts', 'MeanCounts', 'acq_timestamp'])

builder.add_monitor(
    'Camera Check',
    MultiCheck([
        ThresholdCheck(builder.device_group, 'cam1', 'MaxCounts', 4000, '>'),
        ThresholdCheck(builder.device_group, 'cam1', 'MeanCounts', 0, '<'),
    ]),
    interval=0.1
)

builder.start()  # Logs startup, no print statements

try:
    while True:
        time.sleep(0.02)
except KeyboardInterrupt:
    builder.stop()  # Logs shutdown, no print statements
```

**Benefits:**
- ✅ 60% less boilerplate
- ✅ No factory functions or closure management
- ✅ Declarative condition definitions
- ✅ Logging instead of print (integrates with experiment monitoring)
- ✅ Easy to compose complex checks

---

## Condition Builders

### ThresholdCheck

Compares a value against a threshold using any operator.

```python
ThresholdCheck(
    device_monitor_group,
    device_name,
    variable_name,
    threshold,
    operator='<'  # '<', '>', '<=', '>=', '==', '!='
)
```

**Example:**
```python
# MaxCounts > 4000 is unsafe
max_check = ThresholdCheck(dg, 'cam1', 'MaxCounts', 4000, '>')

# MeanCounts < 100 is unsafe
mean_check = ThresholdCheck(dg, 'cam1', 'MeanCounts', 100, '<')
```

### AlignmentCheck

Checks if a value is within tolerance of a target.

```python
AlignmentCheck(
    device_monitor_group,
    device_name,
    value_variable,      # e.g., 'centroidx'
    target_variable,     # e.g., 'Target.X'
    tolerance
)
```

**Example:**
```python
# Centroid within 5 pixels of target
x_align = AlignmentCheck(dg, 'cam1', 'centroidx', 'Target.X', 5)
y_align = AlignmentCheck(dg, 'cam1', 'centroidy', 'Target.Y', 5)
```

### MultiCheck

Combines multiple conditions with OR logic (any unsafe = entire check unsafe).

```python
MultiCheck([
    condition1,
    condition2,
    condition3,
])
```

**Example:**
```python
multi = MultiCheck([
    ThresholdCheck(dg, 'cam1', 'MaxCounts', 4000, '>'),
    AlignmentCheck(dg, 'cam1', 'centroidx', 'Target.X', 5),
    AlignmentCheck(dg, 'cam1', 'centroidy', 'Target.Y', 5),
])
```

### Staleness Detection (Built-in)

**You don't need to do anything!** All interlocks automatically fail-safe if device data is stale.

The `DeviceMonitorGroup` tracks the timestamp for each device and automatically returns `None` (which triggers fail-safe in condition builders) if data hasn't updated within the timeout.

**Configure staleness timeout when creating InterlockBuilder:**

```python
# Default 5000ms timeout
builder = InterlockBuilder('BELLA', host, port)

# Or customize:
builder = InterlockBuilder('BELLA', host, port, staleness_timeout_ms=3000)
```

This is the safety mechanism—if any device stops sending data, all its checks immediately return unsafe (True).

### CustomCheck

For arbitrary logic not covered by built-in checks.

```python
def custom_logic(device_state):
    # Return True (unsafe) or False (safe)
    return device_state['MaxCounts'] > 4000

custom = CustomCheck(dg, 'cam1', custom_logic)
```

---

## Logging Configuration

All output goes through Python's `logging` module (no print statements).

### Basic Setup

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### With File Output

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('interlock.log'),
        logging.StreamHandler()  # Also print to console
    ]
)
```

### Log Levels

- **INFO**: Server start/stop, monitor registration
- **WARNING**: Device offline, data missing, staleness detected
- **DEBUG**: Detailed state transitions, subscription steps
- **ERROR**: Exceptions, subscription failures

---

## Context Manager Support

Alternatively, use context manager for automatic cleanup:

```python
with InterlockBuilder('BELLA', host, port) as builder:
    cam1 = builder.add_device('cam1', 'CAM-PL1-TapeDrivePointing', [...])
    builder.add_monitor('Check', multi_check)
    # Server auto-starts here

    time.sleep(10)

# Server auto-stops here on exit
```

---

## Debugging

Inspect monitor state at runtime:

```python
# Get state of one monitor
state = builder.get_monitor_state('Camera Check')
print(f"State: {state}")  # True=unsafe, False=safe

# Get all monitors
all_states = builder.get_all_monitors()
for name, state in all_states.items():
    print(f"{name}: {state}")
```

---

## Advanced: Custom Predicates

For complex logic:

```python
def complex_logic(device_state):
    max_counts = device_state.get('MaxCounts', 0)
    mean_counts = device_state.get('MeanCounts', 0)

    # Custom: both must be above threshold
    return not (max_counts > 3000 and mean_counts > 100)

custom = CustomCheck(dg, 'cam1', complex_logic)
builder.add_monitor('Complex Check', custom)
```

---

## Migration from Old Pattern

1. Replace device setup with `InterlockBuilder`
2. Replace factory functions with condition builders
3. Replace `server.register_monitor()` with `builder.add_monitor()`
4. Replace `print()` with logging configuration
5. Replace `server.start()/stop()` with `builder.start()/stop()`

Old monitor registration:
```python
server.register_monitor("Check", camera_thresh_check(cam1, "MaxCounts", 4000), interval=0.1)
```

New monitor registration:
```python
builder.add_monitor(
    "Check",
    ThresholdCheck(builder.device_group, 'cam1', 'MaxCounts', 4000, '>'),
    interval=0.1
)
```

---

## Backwards Compatibility

The new utilities are purely additive. Existing code using `InterlockServer` directly continues to work unchanged.

```python
# Old pattern still works
from geecs_python_api.controls.interlocks import InterlockServer

server = InterlockServer(host, port)
# ... setup ...
server.start()
```

---

## FAQ

**Q: Why use logging instead of print?**
A: Logging integrates with experiment monitoring systems, allows output redirection, and enables production-grade logging to files/syslog.

**Q: Why is staleness detection built-in instead of optional?**
A: Safety-first design. Any interlock should inherently fail-safe if device data stops updating. This is automatic and always active—no wrapper needed, no option to forget it.

**Q: Can I customize the staleness timeout?**
A: Yes:
```python
# Default 5000ms
builder = InterlockBuilder('BELLA', host, port)

# Custom timeout
builder = InterlockBuilder('BELLA', host, port, staleness_timeout_ms=3000)
```

**Q: Can I mix old and new patterns?**
A: Yes. `InterlockBuilder` wraps `InterlockServer`, so they're compatible.

**Q: How do I monitor multiple devices?**
A: Use multiple `add_device()` calls:
```python
cam1 = builder.add_device('cam1', 'CAM-1', [...])
cam2 = builder.add_device('cam2', 'CAM-2', [...])

multi = MultiCheck([
    ThresholdCheck(builder.device_group, 'cam1', 'MaxCounts', 4000, '>'),
    ThresholdCheck(builder.device_group, 'cam2', 'MaxCounts', 3000, '>'),
])
```

**Q: What happens if a device goes offline?**
A: Two things:
1. If device stops sending data (staleness timeout), all its checks return unsafe—automatically fail-safe.
2. If a variable is missing, condition builders return `None` which is treated as unsafe.

Both are conservative (fail-safe) behavior.

---

## Examples

See `tape_interlock_refactored_example.py` for a complete working example.
