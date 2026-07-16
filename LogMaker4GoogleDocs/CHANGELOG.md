# Changelog — logmaker-4-googledocs

All notable changes to this package will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.1] — 2026-07-16

### Changed

- Mechanical pre-commit normalization (repo-wide `pre-commit run --all-files`
  pass): trailing-whitespace / end-of-file / CRLF fixes and ruff-format
  reformatting across the legacy scripts (`docgen.py`, `appendScanPW.py`,
  `appendScanOldLog.py`, `appendEpilog.py`, `createGdocPW.py`,
  `docgen_old v2020.py`), plus one unused-import removal (`pickle`).
  No behavior changes.

## [0.1.0] — current
<!-- Add entries here when changes are made -->
