# GEECS Python API Software Interlock Library

The Software Interlock library provides a flexible pipeline for creating device monitors. This library integrates with Master Control to ...

## Overview

The library consists of three main components:

1. **Monitor Condition Builders** — Composable factories for common check types
   - `ThresholdCheck`: value vs. threshold comparison
   - `AlignmentCheck`: tolerance-based alignment
   - `MultiCheck`: combines conditions with OR logic
   - `CustomCheck`: arbitrary predicates

2. **DeviceMonitorGroup** — Manages device subscriptions and state access with **built-in staleness detection**

3. **InterlockBuilder** — High-level facade for server setup and lifecycle

**KEY SAFETY FEATURE:** All interlocks automatically fail-safe (return unsafe) if device data is stale (hasn't updated for `staleness_timeout_ms`). No separate wrapper needed—it's built in!

## File Structure