"""
Utility Module for GEECS Scanner Data Acquisition.

This module provides essential utility functions and classes for configuration
management, logging, and device setup in the GEECS experimental control system.
It offers tools for dynamic configuration file handling, VISA configuration
generation, and flexible logging mechanisms.

**Key Features:**

- Configuration file path resolution
- Dynamic VISA configuration generation
- Flexible console logging with file and stream support
- Cross-platform file management utilities

**Components:**

1. get_full_config_path: Resolves full paths to configuration files
2. visa_config_generator: Generates device configurations for specific diagnostics
3. ConsoleLogger: Provides advanced logging capabilities

**Design Principles:**

- Centralized configuration management
- Flexible and extensible utility functions
- Robust error handling
- Support for complex experimental setups

**Dependencies:**

- pathlib: Path manipulation
- yaml: Configuration file parsing
- logging: Logging infrastructure
- shutil: File management across devices

**Typical Workflow:**
1. Resolve configuration file paths
2. Generate device-specific configurations
3. Set up logging for experimental runs
4. Manage log file movement and cleanup

Example Usage:
```python
# Resolve a configuration file path
config_path = get_full_config_path('Undulator', 'aux_configs', 'visa_config.yaml')

# Generate a VISA configuration
output_config = visa_config_generator('Visa1', 'energy')

# Set up logging
logger = ConsoleLogger(log_file='experiment.log', console=True)
logger.setup_logging()
```

Notes
-----
- Supports dynamic configuration for different experimental setups
- Provides cross-platform file and logging utilities
- Designed for flexibility and ease of use in scientific computing environments

"""

from pathlib import Path
import yaml

from geecs_scanner.utils.application_paths import ApplicationPaths

def get_full_config_path(experiment: str, config_type: str, config_file: str) -> Path:
    """
    Resolve the full filesystem path to a specific configuration file within the GEECS experimental setup.

    This method provides a robust and flexible mechanism for locating configuration files
    across different experimental contexts. It ensures configuration files are accessible
    and provides informative error messages if resolution fails.

    Parameters
    ----------
    experiment : str
        Name of the experiment subdirectory containing the configuration.
        Represents the specific experimental context or setup.
    config_type : str
        Category or type of configuration (e.g., 'aux_configs', 'save_devices').
        Allows for structured organization of configuration files.
    config_file : str
        Name of the specific configuration file to locate.
        The exact filename to be resolved within the specified configuration type.

    Returns
    -------
    Path
        Absolute filesystem path to the requested configuration file.

    Raises
    ------
    FileNotFoundError
        Raised in three potential scenarios:
        1. Base configuration directory does not exist
        2. Specified experiment directory is missing
        3. Requested configuration file cannot be found

    Notes
    -----
    Configuration Resolution Strategy:
    - Uses a fixed directory structure: 'scanner_configs/experiments/{experiment}/{config_type}'
    - Provides granular error messages for different failure points
    - Supports flexible configuration file management
    - Ensures configuration files are located consistently across the project

    Raises
    ------
    FileNotFoundError
        If any part of the configuration path is invalid or missing.

    """
    # Set the base directory to be the 'configs' directory relative to the current directory
    # base_dir = Path(__file__).parents[2] / "scanner_configs" / "experiments"
    base_dir = ApplicationPaths.base_path()

    # Ensure base_dir exists
    if not base_dir.exists():
        raise FileNotFoundError(f"The base config directory {base_dir} does not exist.")

    experiment_dir = base_dir / experiment

    if not experiment_dir.exists():
        raise FileNotFoundError(
            f"The experiment directory {experiment_dir} does not exist."
        )

    config_path = experiment_dir / config_type / config_file
    if not config_path.exists():
        raise FileNotFoundError(f"The config file {config_path} does not exist.")

    return config_path



def visa_config_generator(visa_key: str, diagnostic_type: str) -> Path:
    """
    Generate a dynamic VISA configuration file for specific experimental diagnostics.

    This function creates a specialized YAML configuration for different diagnostic
    scenarios, supporting energy and spectrometer measurements. It dynamically
    configures device settings, setup steps, and output paths based on the provided
    VISA key and diagnostic type.

    Parameters
    ----------
    visa_key : str
        Identifier for the specific VISA configuration (e.g., 'Visa1', 'Visa2').
    diagnostic_type : str
        Type of diagnostic measurement, either 'energy' or 'spectrometer'.

    Returns
    -------
    Path
        Filesystem path to the generated configuration YAML file.

    Raises
    ------
    KeyError
        If the visa_key or diagnostic_type is not found in the configuration lookup.
    FileNotFoundError
        If the input configuration file cannot be located.

    Notes
    -----
    Configuration Generation Process:
    - Loads base configuration from 'visa_plunger_lookup.yaml'
    - Merges common and diagnostic-specific device configurations
    - Generates setup steps for device positioning and initialization
    - Creates a standardized output YAML file

    Supported Diagnostic Types:
    - 'energy': Configures devices for energy measurement
    - 'spectrometer': Sets up devices for spectrometer data collection

    """
    # Load configuration file
    input_filename = get_full_config_path(
        "Undulator", "aux_configs", "visa_plunger_lookup.yaml"
    )
    with open(input_filename, "r") as file:
        visa_lookup = yaml.safe_load(file)

    # Base device info
    device_info = visa_lookup[visa_key]
    visa_ebeam_camera = f"UC_VisaEBeam{visa_key[-1]}"

    # Common device configuration
    common_devices = {
        visa_ebeam_camera: {
            "variable_list": ["timestamp"],
            "synchronous": True,
            "save_nonscalar_data": True,
        },
        "U_BCaveICT": {
            "variable_list": ["Python Results.ChA", "Python Results.ChB", "timestamp"],
            "synchronous": True,
            "save_nonscalar_data": True,
        },
    }

    # Diagnostic-specific settings
    diagnostic_settings = {
        "energy": {
            "description": f"collecting data on {visa_key}EBeam and U_FELEnergyMeter",
            "velmex_position": device_info["energy_meter_position"],
            "zaber_long_position": device_info["zaber_1"],
            "zaber_tran_position": device_info["zaber_2"],
            "extra_device": {
                "U_FELEnergyMeter": {
                    "variable_list": ["Python Results.ChA", "timestamp"],
                    "synchronous": True,
                    "save_nonscalar_data": True,
                    "scan_setup": {"Analysis": ["on", "off"]},
                }
            },
        },
        "spectrometer": {
            "description": f"collecting data on {visa_key}EBeam and UC_UndulatorRad2",
            "velmex_position": device_info["spectrometer_position"],
            "zaber_long_position": device_info["zaber_1"],
            "zaber_tran_position": device_info["zaber_2"],
            "extra_device": {
                "UC_UndulatorRad2": {
                    "variable_list": ["MeanCounts", "timestamp"],
                    "synchronous": True,
                    "save_nonscalar_data": True,
                    "scan_setup": {"Analysis": ["on", "off"]},
                }
            },
            "post_analysis_class": "CameraImageAnalysis",
        },
    }

    # Merge common and diagnostic-specific devices
    config = diagnostic_settings[diagnostic_type]
    devices = {**common_devices, **config["extra_device"]}
    if diagnostic_type == "spectrometer":
        devices[visa_ebeam_camera]["post_analysis_class"] = config.get(
            "post_analysis_class"
        )

    # Setup steps
    setup_steps = [
        {"action": "execute", "action_name": "remove_visa_plungers"},
        {
            "device": device_info["device"],
            "variable": device_info["variable"],
            "action": "set",
            "value": "on",
        },
        {
            "device": "U_Velmex",
            "variable": "Position",
            "action": "set",
            "value": config["velmex_position"],
        },
        {
            "device": "U_UndulatorSpecStage",
            "variable": "Position.Ch1",
            "action": "set",
            "value": config["zaber_long_position"],
        },
        {
            "device": "U_UndulatorSpecStage",
            "variable": "Position.Ch2",
            "action": "set",
            "value": config["zaber_tran_position"],
        },
    ]

    # Constructing the output YAML structure
    output_data = {
        "Devices": devices,
        "scan_info": {"description": config["description"]},
        "setup_action": {"steps": setup_steps},
    }

    # Save to output YAML file
    output_filename = (
        Path(input_filename).parent.parent
        / "save_devices"
        / f"{visa_key}_{diagnostic_type}_setup.yaml"
    )
    with open(output_filename, "w") as outfile:
        yaml.dump(output_data, outfile, default_flow_style=False)

    return output_filename
