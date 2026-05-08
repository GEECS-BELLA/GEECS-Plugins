# GEECS-Data-Utils — Developer Context for Claude

Foundational data layer. Provides scan path navigation, scalar data loading,
data binning/aggregation, and a queryable Parquet-based scan metadata database.
Used by ImageAnalysis, ScanAnalysis, and GEECS-Scanner-GUI.

## Package Layout

```
geecs_data_utils/
  __init__.py                  # Public API: ScanTag, ScanPaths, ScanData, GeecsPathsConfig
  scan_paths.py                # ScanPaths: folder navigation + scan_info.ini parsing
  scan_data.py                 # ScanData: ScanPaths + scalar DataFrame loading + binning
  type_defs.py                 # ScanTag, ScanMode, ScanConfig, ECSDump Pydantic models
  geecs_paths_config.py        # GeecsPathsConfig: base path + experiment resolution
  config_base.py               # ConfigDirManager: generic config directory management
  config_roots.py              # Singleton instances for image/scan analysis config dirs
  utils.py                     # month_to_int, SysPath, ConfigurationError
  plotting_utils.py            # Simple matplotlib helpers for binned data
  scans_database/
    database.py                # ScanDatabase: filter + load Parquet dataset
    builder.py                 # ScanDatabaseBuilder: create/update Parquet dataset
    entries.py                 # ScanEntry, ScanMetadata: Pydantic models for Parquet rows
    filter_models.py           # FilterSpec, FilterArgs
    filters/                   # YAML filter preset files
```

## Core Abstractions

### `ScanTag`

Pydantic model identifying a scan. Immutable and hashable.

```python
ScanTag(year=2024, month=1, day=15, number=42, experiment="Undulator")
```

### `ScanPaths`

Wraps and validates a scan folder path; provides access to metadata and
sub-paths.

```python
paths = ScanPaths(tag=ScanTag(...), base_directory="/data")
paths.get_folder()              # Path to Scan042/
paths.get_analysis_folder()     # Path to Scan042/analysis/
paths.load_scan_info()          # Dict parsed from scan_info.ini
paths.get_folders_and_files()   # Lists device folders and files
ScanPaths.get_latest_scan_tag(experiment, year, month, day, base_directory)
```

Class-level `paths_config: GeecsPathsConfig` — shared across all instances.
Call `ScanPaths.reload_paths_config()` if experiment changes at runtime.

### `ScanData`

Composes `ScanPaths` with scalar data loading and binning.

```python
# Factory methods (preferred)
sd = ScanData.from_date(year=2024, month=1, day=15, number=42,
                        experiment="Undulator",
                        load_scalars=True, source="sfile")

sd = ScanData.latest(experiment="Undulator", load_scalars=True)

# Access
sd.paths.get_folder()           # Path to scan folder
sd.data_frame                   # Optional scalar DataFrame
sd.list_columns()               # Flat list of column names
sd.find_cols("centroid", mode="contains")   # Search column names
```

`source="sfile"` reads the text scalar summary; `source="tdms"` reads the binary
TDMS file. Prefer `sfile` for speed unless you need waveform data.

### `GeecsPathsConfig`

Resolves the GEECS data root and experiment name. Reading order:
1. Explicit `set_base_path` argument
2. Experiment-specific default server path
3. `~/.config/geecs_python_api/config.ini`
4. Raises `ConfigurationError`

Also provides optional paths for config repos and FROG DLL.

## GEECS Folder Convention

```
{base_path}/{experiment}/Y{YYYY}/{MM-Month}/{YY_MMDD}/scans/Scan{NNN}/
  └── Scan{NNN}.tdms
  └── ScanData_scan.txt      (s-file: scalar summary)
  └── scan_info.ini          (scan parameters)
  └── analysis/              (created by ScanAnalysis)
{base_path}/{experiment}/Y{YYYY}/{MM-Month}/{YY_MMDD}/
  └── ECS Live dumps/        (device state snapshots, one per scan)
```

`ScanPaths` validates this convention and raises if the path doesn't conform.

## Binning System

```python
from geecs_data_utils.scan_data import BinningConfig

config = BinningConfig(
    bin_col="Bin #",
    value_cols=["signal_x", "signal_y"],
    agg="median",           # "mean" or "median"
    err="iqr",              # "std", "stderr", "mad", "iqr", "percentile"
    percentiles=(0.25, 0.75),
)
binned = sd.bin(config)
# Returns MultiIndex DataFrame:
# columns: [("signal_x", "center"), ("signal_x", "err_low"), ("signal_x", "err_high"), ...]
```

Used by ScanAnalysis renderers to produce per-bin summary plots.

## Parquet Scan Database

A Hive-partitioned Parquet dataset indexing all historical scans. Not used in
live analysis — primarily for offline search and meta-analysis.

### Schema

Partitioned by `year` and `month`:
```
parquet_root/year=2024/month=1/0.parquet
```

Each row is a `ScanEntry`: date, number, experiment, file paths, non-scalar
device list, scan metadata (parsed from scan_info.ini), ECS dump (JSON),
analysis presence flag, notes.

### Querying

```python
from geecs_data_utils.scans_database import ScanDatabase
from datetime import date

db = ScanDatabase("/data/Undulator/scan_database_parquet")
df = (db
      .with_date_range(date(2024, 1, 1), date(2024, 12, 31))
      .with_experiment("Undulator")
      .with_named_filter("my_filter", date(2024, 6, 15))  # YAML preset
      .load())
```

### Building / Updating

```python
from geecs_data_utils.scans_database import ScanDatabaseBuilder

ScanDatabaseBuilder.stream_to_parquet(
    data_root="/data",
    experiment="Undulator",
    output_path="/data/Undulator/scan_db",
    date_range=(date(2024, 1, 1), date.today()),
    mode="append",      # or "overwrite"
)
```

## Config Directory Management

`ConfigDirManager` (config_base.py) manages a directory that can hold multiple
YAML config files. Used by ImageAnalysis and ScanAnalysis to locate their
config YAML repos.

```python
from geecs_data_utils.config_roots import image_analysis_config, scan_analysis_config

# These are pre-built singletons. Path resolved from:
# 1. Environment variable
# 2. GeecsPathsConfig.image_analysis_configs_path
# 3. Raises ConfigurationError

cfg_path = image_analysis_config.get_path("UC_GaiaMode.yaml")
```

## Key Type Definitions

- **`ScanMode`** (Enum) — `STANDARD`, `NOSCAN`, `OPTIMIZATION`, `BACKGROUND`
- **`ScanConfig`** — dataclass: scan_mode, device_var, start, end, step,
  wait_time, shots_per_step, additional_description
- **`ECSDump`** / **`DeviceDump`** — Pydantic models for ECS live dump files

## Useful Utilities

```python
from geecs_data_utils.utils import month_to_int
month_to_int("January")  # → 1
month_to_int(3)           # → 3

from geecs_data_utils.utils import read_geecs_tdms
data = read_geecs_tdms(path)  # → dict[device][variable] → np.ndarray

from geecs_data_utils.plotting_utils import plot_binned, plot_binned_multi
```

## How Other Packages Use This

- **ImageAnalysis** — `ScanPaths` to locate device data folders per scan
- **ScanAnalysis** — `ScanData` for binning scalar data in summary plots;
  `ScanPaths` as the base for scan folder resolution
- **GEECS-Scanner-GUI** — `ScanConfig` / `ScanMode` enums for scan parameter
  definitions; `ScanPaths` for post-scan file organization

## Key Dependency

- `nptdms` — TDMS binary file reading
- `pyarrow` — Parquet I/O
- `duckdb` — available for ad-hoc Parquet queries but minimal use
- `pydantic >= 2.0` — all data models
