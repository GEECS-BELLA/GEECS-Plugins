# Changelog

All notable changes to `geecs-panels` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to semantic versioning.


## [0.1.0] - 2026-10-22

### Added

- The prototype (`Planning/device_panels.md`, build step 1): a landing page
  from the experiment's `native_devices.yaml`, and the read-only camera
  panel — the live image pulled one frame at a time through a shared proxy,
  a client-side colour map and levels, and the STAT/STAT0 readings over
  SSE, each judged fresh by its plugin's frame counter.
