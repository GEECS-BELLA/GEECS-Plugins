# Changelog

All notable changes to `geecs-canopy` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-28

### Added

- The package, on the GeecsArchiver pattern: the deploy recipe (Canopy
  `0a5ee23` pinned by digest in `geecs_canopy/pin.py`, a derived image with
  GEECS-Core, GEECS-Schemas and `mysql-connector-python`, Postgres 16 and
  Redis 7 in the same compose project, the `geecs-canopy` unit, the
  `/canopy` prefix, the upgrade/rollback runbook) and the Raccoon directory
  adapter (`geecs_canopy.directory:GeecsDirectoryAdapter`) over the native
  registry, the panel file and the family files, with the transitional
  GEECS DB source for the gateway's devices.
