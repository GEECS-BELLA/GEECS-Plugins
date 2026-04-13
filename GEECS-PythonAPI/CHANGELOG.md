# Changelog — geecs-python-api

All notable changes to this package will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.1] — 2026-04-13

### Fixed
- `dequeue_command()` now catches `GeecsDeviceCommandRejected` (logged as
  warning) and bare `Exception` (logged as error) so rejected commands never
  produce unhandled "Exception in thread" output in daemon threads
- `_process_command()` guards against `dev_udp is None` (device already closed)
  with an early return instead of a bare `assert`, eliminating `AssertionError`
  tracebacks when dequeue threads outlive `device.close()`
