# Changelog — scan-analysis

All notable changes to this package will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.3.0] — 2026-05-06

### Added
- `ScatterPlotterAnalysis` now supports a configurable `x_column` parameter:
  any s-file column can be used as the x-axis. For 1D scans the per-bin
  statistic of the column is used; for noscans the per-shot values are used.
  Falls back to the scan parameter (1D scans) or shot number (noscans) when
  `x_column` is not set.
- `PlotParameter` gains an optional `y_range: tuple[float, float]` field to
  fix the y-axis limits for a series. When omitted matplotlib auto-ranges.
- Output PNG is now saved to `<scan_folder>/analysis/scatter_plots/<filename>.png`
  (previously saved outside the scan folder tree).
- New Pydantic config models `PlotParameterConfig` and `ScatterAnalyzerConfig`
  (`type: "scatter"`) allow scatter plots to be declared in experiment YAML
  configs alongside `array2d` / `array1d` analyzers.
- `create_analyzer()` factory handles `ScatterAnalyzerConfig` and converts
  config objects to `PlotParameter` / `ScatterPlotterAnalysis` instances.
- Unit tests for `PlotParameterConfig` and `ScatterAnalyzerConfig` (no
  external data required).
- Integration tests for `ScatterPlotterAnalysis` and the factory round-trip
  against Undulator 2026-05-05 Scan018 (`@pytest.mark.integration`).

## [1.3.1] — 2026-05-12

### Fixed
- **ConfigFileGUI**: `Name` field in the General Settings panel is now editable.
  Previously it was frozen to the YAML filename stem, making it impossible to set
  a `name` that differs from the config file name. The field is now populated from
  the loaded config's `name` value and saved back normally.
- **ConfigFileGUI**: Float fields now support scientific notation (e.g. `1e-13`).
  `QDoubleSpinBox` with `decimals=6` silently rounded sub-micro values to zero.
  Replaced with `ScientificDoubleSpinBox`, a thin subclass that overrides
  `validate` / `valueFromText` / `textFromValue` to accept and display scientific
  notation, with precision set to 15 significant digits.

### Added
- Unit tests for `ScientificDoubleSpinBox` (validate, text↔value conversion,
  parametrized round-trip).
- Round-trip tests for `config_io` save/load cycle covering `name` field
  independence from filename and survival of scientific-notation floats.

## [1.1.1] — current
<!-- Add entries here when changes are made -->
