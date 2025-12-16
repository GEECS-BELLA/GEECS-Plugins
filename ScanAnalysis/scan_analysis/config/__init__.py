"""
Configuration system for scan analysis.

This package provides a modern, YAML-based configuration system for defining
scan analyzers, replacing the older hardcoded map_*.py files.

Main Components
---------------
- Pydantic models for analyzer configuration (analyzer_config_models)
- Config file loader with recursive search (config_loader)
- Factory for instantiating analyzers from config (analyzer_factory)

Quick Start
-----------
1. Set up config directory:

    >>> from geecs_data_utils.config_roots import scan_analysis_config
    >>> scan_analysis_config.set_base_dir("/path/to/scan_analysis_configs")

2. Load experiment configuration:

    >>> from scan_analysis.config import load_experiment_config
    >>> config = load_experiment_config("undulator")

3. Create analyzers:

    >>> from scan_analysis.config import create_analyzer
    >>> analyzers = [create_analyzer(cfg) for cfg in config.active_analyzers]

4. Sort by priority and run:

    >>> analyzers.sort(key=lambda a: a.priority)
    >>> for analyzer in analyzers:
    ...     analyzer.run_analysis(scan_tag)

Environment Variables
--------------------
SCAN_ANALYSIS_CONFIG_DIR : str
    Directory containing scan analysis YAML configs. If set, automatically
    initializes the config base directory on import.

Public API
----------
Configuration Models:
    - ImageAnalyzerConfig
    - Array2DAnalyzerConfig
    - Array1DAnalyzerConfig
    - ScanAnalyzerConfig (Union type)
    - ExperimentAnalysisConfig

Config Loading:
    - load_experiment_config
    - find_config_file
    - list_available_configs

Analyzer Creation:
    - create_analyzer
    - create_image_analyzer

Examples
--------
Example YAML configuration file (undulator_analysis.yaml):

    experiment: Undulator
    description: Standard analysis for Undulator experiment
    version: "1.0"
    upload_to_scanlog: true

    analyzers:
      # High priority (analyze first)
      - type: array2d
        device_name: UC_GaiaMode
        priority: 0
        image_analyzer:
          analyzer_class: image_analysis.offline_analyzers.beam_analyzer.BeamAnalyzer
          camera_config_name: UC_GaiaMode

      - type: array2d
        device_name: UC_ExpanderIn1_Pulsed
        priority: 10
        image_analyzer:
          analyzer_class: image_analysis.offline_analyzers.beam_analyzer.BeamAnalyzer
          camera_config_name: UC_ExpanderIn1_Pulsed

      # Medium priority
      - type: array2d
        device_name: UC_ALineEbeam1
        priority: 50
        image_analyzer:
          analyzer_class: image_analysis.offline_analyzers.beam_analyzer.BeamAnalyzer
          camera_config_name: UC_ALineEbeam1

      # Background priority (default)
      - type: array1d
        device_name: U_BCaveICT
        priority: 100
        image_analyzer:
          analyzer_class: image_analysis.offline_analyzers.standard_1d_analyzer.Standard1DAnalyzer
          kwargs:
            data_type: tdms

Complete workflow example:

    >>> from scan_analysis.config import (
    ...     load_experiment_config,
    ...     create_analyzer
    ... )
    >>> from geecs_data_utils.config_roots import scan_analysis_config
    >>>
    >>> # Set up config directory
    >>> scan_analysis_config.set_base_dir("/data/configs/scan_analysis")
    >>>
    >>> # Load experiment configuration
    >>> config = load_experiment_config("undulator")
    >>> print(f"Loaded {len(config.active_analyzers)} analyzers")
    >>>
    >>> # Get high-priority analyzers only
    >>> priority_configs = config.get_analyzers_by_priority(max_priority=10)
    >>> priority_analyzers = [create_analyzer(cfg) for cfg in priority_configs]
    >>>
    >>> # Run high-priority analysis
    >>> for analyzer in priority_analyzers:
    ...     analyzer.run_analysis(scan_tag)
    >>>
    >>> # Later, run remaining analyzers
    >>> remaining_configs = [
    ...     cfg for cfg in config.active_analyzers
    ...     if cfg.priority > 10
    ... ]
    >>> remaining_analyzers = [create_analyzer(cfg) for cfg in remaining_configs]
"""

# Configuration models
from .analyzer_config_models import (
    Array1DAnalyzerConfig,
    Array2DAnalyzerConfig,
    ExperimentAnalysisConfig,
    ImageAnalyzerConfig,
    ScanAnalyzerConfig,
)

# Config loading
from .config_loader import (
    find_config_file,
    list_available_configs,
    load_experiment_config,
)

# Analyzer factory
from .analyzer_factory import create_analyzer, create_image_analyzer

__all__ = [
    # Models
    "ImageAnalyzerConfig",
    "Array2DAnalyzerConfig",
    "Array1DAnalyzerConfig",
    "ScanAnalyzerConfig",
    "ExperimentAnalysisConfig",
    # Config loading
    "load_experiment_config",
    "find_config_file",
    "list_available_configs",
    # Factory
    "create_analyzer",
    "create_image_analyzer",
]
