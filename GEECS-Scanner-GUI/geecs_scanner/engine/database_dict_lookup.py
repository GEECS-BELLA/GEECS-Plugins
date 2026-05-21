"""
Database Dictionary Lookup Module for GEECS Experimental Configuration Management.

This module provides the DatabaseDictLookup class, a critical component for managing
and retrieving device configuration information across different experimental setups.
It serves as a centralized mechanism for loading and caching device-specific metadata
from configuration files.

**Key Features:**

- Dynamic experiment configuration loading
- Caching of device information
- Lazy loading and configuration reuse
- Seamless integration with GEECS device infrastructure

**Design Principles:**

- Minimize redundant configuration file reads
- Provide a flexible mechanism for experiment-specific device metadata
- Support default experiment configuration retrieval
- Robust error handling for configuration loading

**Dependencies:**

- geecs_python_api.controls.interface.load_config
- geecs_python_api.controls.interface.GeecsDatabase
- geecs_python_api.controls.devices.geecs_device.GeecsDevice

**Typical Workflow:**

1. Initialize DatabaseDictLookup
2. Reload configuration for a specific experiment
3. Retrieve device configuration dictionary
4. Use device information in experimental control logic

Example Usage:
```python
db_lookup = DatabaseDictLookup()
db_lookup.reload(experiment_name='MyExperiment')
device_config = db_lookup.get_database()
```

Notes
-----
- Configuration is cached to minimize file I/O
- Supports both explicit and default experiment name resolution
- Provides logging for configuration loading processes
- An improved method for accessing database information should be implemented.
Rather than periodically load entire database tables, allow basic sql style
quieries to be run against the database.

See Also
--------
- geecs_python_api.controls.interface.GeecsDatabase : Experimental database information collection
- geecs_python_api.controls.devices.geecs_device.GeecsDevice : Base device configuration management
"""

import logging
from typing import Optional
from geecs_python_api.controls.interface import load_config, GeecsDatabase
from geecs_python_api.controls.devices.geecs_device import GeecsDevice

logger = logging.getLogger(__name__)


class DatabaseDictLookup:
    """
    A centralized configuration management system for experimental device metadata.

    This class provides a robust mechanism for loading, caching, and retrieving
    device configuration information across different experimental contexts.
    It minimizes redundant configuration file reads and supports flexible
    experiment-specific device metadata retrieval.

    Attributes
    ----------
    experiment_name : str or None
        The name of the current experiment for which configuration is loaded.
    load_config_flag : bool
        Flag to ensure configuration is loaded at least once, even without an explicit experiment name.
    database_dict : dict
        A dictionary containing device configurations and metadata for the current experiment.

    Key Methods
    ----------
    get_database()
        Retrieve the current device configuration dictionary.
    reload(experiment_name=None)
        Load or refresh device configuration for a specific experiment.

    Notes
    -----
    - Supports lazy loading of configuration
    - Caches configuration to minimize file system access
    - Provides fallback mechanisms for default experiment configuration
    - Integrates with GEECS device and database infrastructure

    Design Principles
    ----------------
    - Minimize redundant configuration file reads
    - Provide flexible experiment-specific device metadata retrieval
    - Support default experiment configuration resolution
    - Implement robust error handling for configuration loading

    Examples
    --------
    >>> db_lookup = DatabaseDictLookup()
    >>> db_lookup.reload(experiment_name='MyExperiment')
    >>> device_configs = db_lookup.get_database()
    >>> print(device_configs)
    {'DeviceA': {...}, 'DeviceB': {...}}

    Future Improvements
    ------------------
    - Implement more advanced querying capabilities
    - Add support for partial configuration updates
    - Enhance error handling and logging

    See Also
    --------
    geecs_python_api.controls.interface.GeecsDatabase : Experimental database information collection
    geecs_python_api.controls.devices.geecs_device.GeecsDevice : Base device configuration management
    """

    def __init__(self):
        """
        Initialize a new DatabaseDictLookup instance.

        Sets up the initial state for configuration management, preparing
        for lazy loading of experiment-specific device configurations.
        """
        self.experiment_name: Optional[str] = None
        self.load_config_flag: bool = False
        self.database_dict: dict = {}

    def get_database(self) -> dict:
        """
        Retrieve the current device configuration dictionary.

        Returns
        -------
        dict
            A dictionary of device configurations for the current experiment.
            Returns an empty dictionary if no configuration has been loaded.

        Examples
        --------
        >>> db_lookup = DatabaseDictLookup()
        >>> db_lookup.reload(experiment_name='MyExperiment')
        >>> configs = db_lookup.get_database()
        >>> print(configs)
        {'DeviceA': {...}, 'DeviceB': {...}}
        """
        return self.database_dict

    def reload(self, experiment_name: Optional[str] = None):
        """
        Load or refresh device configuration for a specific experiment.

        This method manages the loading of device configurations, supporting both
        explicit experiment names and automatic default experiment detection.
        It ensures configuration is loaded efficiently by caching and minimizing
        redundant file reads.

        Parameters
        ----------
        experiment_name : str, optional
            The name of the experiment for which to load device configurations.
            If not provided, attempts to retrieve the default experiment name
            from the configuration file.

        Notes
        -----
        Configuration Loading Process:
        - Check if configuration is already loaded for the given experiment
        - Retrieve default experiment name if no name is provided
        - Collect experiment information using GeecsDatabase
        - Cache device configurations in `database_dict`
        - Provide robust error handling for configuration retrieval

        Logging:
        - Logs default experiment name
        - Warns if configuration file is not found or experiment is undefined
        - Logs warnings for device dictionary retrieval failures

        Examples
        --------
        >>> db_lookup = DatabaseDictLookup()
        >>> db_lookup.reload()  # Load default experiment configuration
        >>> db_lookup.reload('CustomExperiment')  # Load specific experiment configuration

        Raises
        ------
        No explicit exceptions, but logs warnings for configuration loading issues
        """
        if self.experiment_name == experiment_name and self.load_config_flag:
            return
        self.load_config_flag = True  # Flag ensures config file is read at least once if no experiment name given

        if experiment_name is None:
            config = load_config()

            if config and "Experiment" in config and "expt" in config["Experiment"]:
                experiment_name = config["Experiment"]["expt"]
                logger.info("default experiment is: %s", experiment_name)
            else:
                logger.warning(
                    "Configuration file not found or default experiment not defined."
                )

        self.experiment_name = experiment_name
        try:
            GeecsDevice.exp_info = GeecsDatabase.collect_exp_info(experiment_name)
            self.database_dict = GeecsDevice.exp_info["devices"]
        except AttributeError:
            logger.warning("Could not retrieve dictionary of GEECS Devices")
            self.database_dict = {}
