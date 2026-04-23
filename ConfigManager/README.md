# GEECS Config Manager

A PyQt5 desktop GUI for creating and editing YAML configs for GEECS ImageAnalysis and ScanAnalysis.

## What it does

- **Image 2D tab** — create/edit `CameraConfig` YAMLs (background, ROI, filtering, transforms, etc.)
- **Image 1D tab** — create/edit `Line1DConfig` YAMLs (scope/CSV data, ROI, filtering)
- **Scan tab** — create/edit `ExperimentAnalysisConfig` YAMLs (list of Array2D/Array1D analyzers)

All form fields are auto-generated from the Pydantic models in `ImageAnalysis` and `ScanAnalysis`.
Pydantic validation runs on Save, so the YAML on disk is always valid.

## Installation

ConfigManager has its own Poetry environment because PyQt5 version constraints
conflict with other packages in the root environment (particularly on Apple Silicon).

```bash
cd ConfigManager
poetry install
```

## Running

```bash
cd ConfigManager
poetry run python -m config_manager
```

Or after activating the env:

```bash
python -m config_manager
```

## Windows (lab machines)

Same steps — PyQt5 5.15.x has wheels for Windows x64. If you already have
`scananalysis` and `imageanalysis` installed in a shared environment you can
also just `pip install pyqt5 pyyaml` and run `python -m config_manager` from
the `ConfigManager/` directory.

## Pointing at your configs repo

Use **Open…** to navigate to any YAML file in your `GEECS-Plugins-configs`
checkout. The GUI auto-detects whether the file is a CameraConfig, Line1DConfig,
or ExperimentAnalysisConfig from the YAML content and switches to the right tab.
