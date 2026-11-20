# Changelog

All notable changes to `geecs-analysis-worker` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to semantic versioning.


## [0.1.0] - 2026-11-20

### Added

- The package: the analysis worker's HTTP API (`geecs_analysis_worker.app`)
  and the `scan-analysis-worker` command, moved out of ScanAnalysis, whose
  `scan_analysis.worker` keeps the job runner. The command reads its bind
  address and port from `GEECS_ANALYSIS_WORKER_BIND` and
  `GEECS_ANALYSIS_WORKER_PORT` when no flag names them. The unit template
  `deploy/geecs-analysis-worker.service`.
