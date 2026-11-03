# Changelog

All notable changes to `geecs-panels` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to semantic versioning.


## [Unreleased]

### Changed

- Saturation is judged against the camera IOC's `FullScale` record (shown
  as a reading) when it publishes one, and against the pixel format's full
  scale otherwise. The IOCs now put every camera of 12 bits or fewer on
  0-4095 with a driver shift, so the format alone misjudges a shifted
  Blackfly (Mono16, top 4095) and a shifted Mono10 camera (top 4092).

## [0.1.0] - 2026-10-22

### Added

- The prototype (`Planning/device_panels.md`, build step 1): a landing page
  from the experiment's `native_devices.yaml`, and the read-only camera
  panel — the live image pulled one frame at a time through a shared proxy,
  a client-side colour map and levels, and the STAT/STAT0 readings over
  SSE, each judged fresh by its plugin's frame counter.
