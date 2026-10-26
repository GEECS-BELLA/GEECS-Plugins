# Changelog

All notable changes to `geecs-plotter` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to semantic versioning.


## [0.1.0] - 2026-10-26

### Added

- The skeleton and the first plot (`Planning/geecs_plotter.md`, M1): a page
  that takes a day and a scan number, lists the scan's s-file columns, and
  plots one against another with the vendored Plotly bundle; X defaults to
  the scan variable. `/api/columns` and `/api/figure` serve the same as
  JSON, the figure with its notebook snippet.
