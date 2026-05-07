# Changelog — geecs-data-utils

All notable changes to this package will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.0] — 2026-05-06

### Added
- `GeecsPathsConfig` now reads an optional `wavekit_config_path` key from the
  `[Paths]` section of `config.ini` and exposes it as an attribute (consistent
  with the existing `frog_dll_path` / `frog_python32_path` pattern). Returns
  `None` if the key is absent or the path does not exist.

## [0.2.1] — current
<!-- Add entries here when changes are made -->
