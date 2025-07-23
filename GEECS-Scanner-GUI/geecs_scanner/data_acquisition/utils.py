"""
Utility Module for GEECS Scanner Data Acquisition.

This module provides essential utility functions and classes for configuration
management, logging, and device setup in the GEECS experimental control system.
It offers tools for dynamic configuration file handling, VISA configuration
generation, and flexible logging mechanisms.

Key Features:
- Configuration file path resolution
- Dynamic VISA configuration generation
- Flexible console logging with file and stream support
- Cross-platform file management utilities

Components:
1. get_full_config_path: Resolves full paths to configuration files
2. visa_config_generator: Generates device configurations for specific diagnostics
3. ConsoleLogger: Provides advanced logging capabilities

Design Principles:
- Centralized configuration management
- Flexible and extensible utility functions
- Robust error handling
- Support for complex experimental setups

Dependencies:
- pathlib: Path manipulation
- yaml: Configuration file parsing
- logging: Logging infrastructure
- shutil: File management across devices

Typical Workflow:
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

See Also
--------
- geecs_scanner.data_acquisition.device_manager.DeviceManager
- geecs_scanner.data_acquisition.scan_manager.ScanManager
"""

import logging
from pathlib import Path
import shutil
import yaml


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

    Examples
    --------
    >>> config_path = get_full_config_path('Undulator', 'aux_configs', 'visa_config.yaml')
    >>> print(config_path)
    /path/to/GEECS-Plugins/GEECS-Scanner-GUI/scanner_configs/experiments/Undulator/aux_configs/visa_config.yaml

    >>> config_path = get_full_config_path('Laser', 'save_devices', 'device_settings.yaml')
    >>> print(config_path)
    /path/to/GEECS-Plugins/GEECS-Scanner-GUI/scanner_configs/experiments/Laser/save_devices/device_settings.yaml

    Raises
    ------
    FileNotFoundError
        If any part of the configuration path is invalid or missing.

    See Also
    --------
    pathlib.Path : Used for robust path manipulation and resolution
    """
    # Set the base directory to be the 'configs' directory relative to the current directory
    base_dir = Path(__file__).parents[2] / "scanner_configs" / "experiments"

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

    Examples
    --------
    >>> config_path = visa_config_generator('Visa1', 'energy')
    >>> print(config_path)
    /path/to/Visa1_energy_setup.yaml

    >>> config_path = visa_config_generator('Visa2', 'spectrometer')
    >>> print(config_path)
    /path/to/Visa2_spectrometer_setup.yaml

    See Also
    --------
    get_full_config_path : Utility for resolving configuration file paths
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


class ConsoleLogger:
    """
    A comprehensive logging utility for managing system-wide logging operations.

    Provides advanced logging capabilities with support for file and console
    output, dynamic log level configuration, and robust handler management.
    Designed to support complex logging requirements in scientific computing
    and experimental control systems.

    Attributes
    ----------
    log_file : str
        Path to the log file where logs will be written. Defaults to 'system_log.log'.
    level : int
        Logging level (e.g., logging.INFO, logging.DEBUG). Determines the verbosity of logging.
    console : bool
        Flag to enable console output in addition to file logging. Defaults to False.
    logging_active : bool
        Indicates whether a logging session is currently active.

    Methods
    -------
    setup_logging()
        Initialize and configure logging handlers for file and optional console output.
    stop_logging()
        Terminate the current logging session and clean up handlers.
    move_log_file(dest_dir)
        Relocate the log file to a specified directory, handling cross-device moves.
    is_logging_active()
        Check the current status of logging.

    Examples
    --------
    >>> logger = ConsoleLogger(log_file='experiment.log', console=True, level=logging.DEBUG)
    >>> logger.setup_logging()
    >>> logging.info("Experiment initialization started")
    >>> logging.debug("Detailed diagnostic information")
    >>> logger.stop_logging()

    Notes
    -----
    - Supports both file and console logging
    - Prevents handler duplication
    - Provides safe logging session management
    - Cross-platform file movement support
    - Configurable log levels and output destinations

    See Also
    --------
    logging : Python's built-in logging module
    shutil : High-level file operations
    """

    def __init__(
        self,
        log_file: str = "system_log.log",
        level: int = logging.INFO,
        console: bool = False,
    ):
        """
        Initialize the ConsoleLogger with specified configuration.

        Parameters
        ----------
        log_file : str, optional
            Path to the log file. Defaults to 'system_log.log'.
        level : int, optional
            Logging level. Defaults to logging.INFO.
            Use logging constants like logging.DEBUG, logging.INFO, logging.WARNING, etc.
        console : bool, optional
            Enable console logging. Defaults to False.
        """
        self.log_file = log_file
        self.level = level
        self.console = console
        self.logging_active = False

    def setup_logging(self):
        """
        Configure logging handlers for file and optional console output.

        Sets up logging with a file handler and optionally a stream handler.
        Removes any pre-existing handlers to prevent duplication.

        Raises
        ------
        RuntimeWarning
            If an attempt is made to start logging when a session is already active.

        Notes
        -----
        - Clears existing logging handlers
        - Configures file logging
        - Optionally enables console logging
        - Sets logging format and level
        - Logs the start of the logging session
        """
        if self.logging_active:
            logging.warning("Logging is already active, cannot start a new session.")
            return

        # Remove any previously configured handlers to prevent duplication
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Configure logging with both file and optional console handlers
        handlers = [logging.FileHandler(self.log_file, encoding="utf-8")]
        if self.console:
            handlers.append(logging.StreamHandler())

        logging.basicConfig(
            level=self.level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=handlers,
        )
        self.logging_active = True
        logging.info(
            "Logging session started with configuration: "
            f"file={self.log_file}, level={logging.getLevelName(self.level)}, "
            f"console_output={'enabled' if self.console else 'disabled'}"
        )

    def stop_logging(self):
        """
        Terminate the current logging session and clean up logging handlers.

        Safely stops logging by closing and removing all active handlers.
        Prevents resource leaks and ensures clean logging shutdown.

        Notes
        -----
        - Checks if a logging session is active before stopping
        - Closes and removes all logging handlers
        - Logs the termination of the logging session
        - Resets logging state
        - Provides console output about logging termination
        """
        if not self.logging_active:
            logging.warning("No active logging session to stop.")
            return

        logging.info("Stopping logging session and cleaning up handlers.")

        for handler in logging.root.handlers[:]:
            handler.flush()  # Ensure all log messages are written
            handler.close()
            logging.root.removeHandler(handler)

        self.logging_active = False
        print(f"Logging session terminated. Log file: {self.log_file}")

    def move_log_file(self, dest_dir: Path):
        """
        Move the log file to a specified destination directory.

        Handles cross-device file moves using shutil, ensuring robust file relocation.

        Parameters
        ----------
        dest_dir : Path
            Destination directory where the log file should be moved.

        Notes
        -----
        - Uses shutil for cross-device file movement
        - Provides detailed console and logging output about move operation
        - Handles potential exceptions during file move
        - Logs the result of the file move operation

        Raises
        ------
        Exception
            If file move operation fails due to permission, disk space, or other issues.
        """
        src_path = Path(self.log_file)
        dest_path = Path(dest_dir) / src_path.name

        logging.info(f"Attempting to move log file from {src_path} to {dest_path}")

        try:
            shutil.move(str(src_path), str(dest_path))
            logging.info(f"Successfully moved log file to {dest_path}")
            print(f"Moved log file to {dest_path}")

            # Update log_file path to reflect the new location
            self.log_file = str(dest_path)
        except Exception as e:
            logging.error(f"Failed to move {src_path} to {dest_path}: {e}")
            print(f"Failed to move {src_path} to {dest_path}: {e}")
            raise

    def is_logging_active(self) -> bool:
        """
        Check the current status of the logging session.

        Returns
        -------
        bool
            True if a logging session is active, False otherwise.

        Notes
        -----
        - Provides a simple way to check logging state
        - Useful for determining whether logging can be started or stopped
        - Can be used in conditional logic for logging management
        """
        return self.logging_active
