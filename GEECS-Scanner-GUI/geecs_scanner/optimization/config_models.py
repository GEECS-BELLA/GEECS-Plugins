"""Model definitions for setting up optimizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from xopt import VOCS

from geecs_scanner.data_acquisition.schemas.save_devices import SaveDeviceConfig


class ImageAnalyzerConfig(BaseModel):
    """
    Configuration for dynamically creating an ImageAnalyzer instance.

    This model specifies the module and class to import for an ImageAnalyzer,
    along with any keyword arguments to be passed to the class initializer.
    Most ImageAnalyzers auto-configure from device name and config files,
    so kwargs are typically empty.

    Attributes
    ----------
    module : str
        Import path to the ImageAnalyzer module (e.g.,
        ``image_analysis.offline_analyzers.beam_analyzer``).
    class_ : str
        Name of the ImageAnalyzer class within the module.
    kwargs : dict of str, Any
        Dictionary of keyword arguments to pass to the ImageAnalyzer constructor.
        Often empty as analyzers auto-configure from device name.

    Notes
    -----
    The field `class_` is aliased as `class` in YAML/JSON configuration files.
    """

    module: str
    class_: str = Field(..., alias="class")
    kwargs: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True)


class SingleDeviceScanAnalyzerConfig(BaseModel):
    """
    Configuration for creating a SingleDeviceScanAnalyzer instance.

    This model specifies all parameters needed to instantiate either an
    Array1DScanAnalyzer or Array2DScanAnalyzer, including the device name,
    analyzer type, file pattern, and the ImageAnalyzer to use for processing.

    Attributes
    ----------
    device_name : str
        Name of the device whose data will be analyzed (e.g., 'UC_ALineEBeam3').
        This is used to locate data files and identify the device in results.
    analyzer_type : {"Array1DScanAnalyzer", "Array2DScanAnalyzer"}
        Type of scan analyzer to instantiate. Choose based on data dimensionality.
    file_tail : str, default=".png"
        File extension/suffix used to match data files for this device.
    image_analyzer : ImageAnalyzerConfig
        Configuration for the ImageAnalyzer instance to use for processing.
    analysis_mode : {"per_shot", "per_bin"}, default="per_bin"
        Analysis mode for the scan analyzer. "per_bin" is recommended for
        optimization as it leverages built-in averaging.

    Methods
    -------
    to_device_requirement()
        Generate a device_requirements entry for this analyzer.
    """

    device_name: str
    analyzer_type: Literal["Array1DScanAnalyzer", "Array2DScanAnalyzer"]
    file_tail: str = ".png"
    image_analyzer: ImageAnalyzerConfig
    analysis_mode: Literal["per_shot", "per_bin"] = "per_bin"

    def to_device_requirement(self) -> dict:
        """
        Generate device_requirements entry for this analyzer.

        Creates a properly formatted device requirement dictionary that can be
        used by the scan data manager to instantiate and configure the device.

        Returns
        -------
        dict
            Device requirements dictionary with device name as key and
            configuration as value.
        """
        return {
            self.device_name: {
                "add_all_variables": False,
                "save_nonscalar_data": True,
                "synchronous": True,
                "variable_list": ["acq_timestamp"],
            }
        }


class MultiDeviceScanEvaluatorConfig(BaseModel):
    """
    Configuration for evaluators using multiple SingleDeviceScanAnalyzers.

    This model contains a list of SingleDeviceScanAnalyzerConfig instances
    and provides utilities for auto-generating device requirements from
    the analyzer configurations.

    Attributes
    ----------
    analyzers : list of SingleDeviceScanAnalyzerConfig
        List of analyzer configurations. Each analyzer will be instantiated
        and used to collect data during optimization.

    Methods
    -------
    generate_device_requirements()
        Auto-generate device_requirements dict from analyzer configs.
    """

    analyzers: List[SingleDeviceScanAnalyzerConfig]

    def generate_device_requirements(self) -> dict:
        """
        Auto-generate device_requirements from analyzer configs.

        Iterates through all analyzer configurations and combines their
        device requirements into a single dictionary suitable for use
        by the scan data manager.

        Returns
        -------
        dict
            Device requirements dictionary with "Devices" key containing
            all device configurations.
        """
        devices = {}
        for analyzer_config in self.analyzers:
            devices.update(analyzer_config.to_device_requirement())
        return {"Devices": devices}


class EvaluatorConfig(BaseModel):
    """
    Configuration for an optimization evaluator.

    This model specifies the module and class to import for the evaluator,
    along with any keyword arguments to be passed to the class initializer.

    Attributes
    ----------
    module : str
        Import path to the evaluator module (e.g.,
        ``geecs_scanner.optimization.evaluators.HiResMagCam``).
    class_ : str
        Name of the evaluator class within the module.
    kwargs : dict of str, Any
        Dictionary of keyword arguments to pass to the evaluator constructor.

    Notes
    -----
    The field `class_` is aliased as `class` in YAML/JSON configuration files.
    """

    module: str
    class_: str = Field(..., alias="class")
    kwargs: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True)


class GeneratorConfig(BaseModel):
    """
    Configuration for the optimization generator.

    Attributes
    ----------
    name : str
        Name of the generator algorithm (e.g., ``random``, ``cnsga``,
        ``upper_confidence_bound``).
    """

    name: str


class BaseOptimizerConfig(BaseModel):
    """
    Canonical optimizer configuration schema.

    This model defines the full configuration for a GEECS optimization run.
    It combines the optimization problem specification (VOCS), evaluator
    definition, generator algorithm, and device saving strategy into a
    unified, validated schema.

    For evaluators using MultiDeviceScanEvaluator, device_requirements are
    automatically generated from the analyzer configurations in evaluator.kwargs.

    Attributes
    ----------
    vocs : VOCS
        Xopt VOCS specification defining variables, objectives, and constraints.
    evaluator : EvaluatorConfig
        Evaluator module/class specification and initialization arguments.
    generator : GeneratorConfig
        Optimization generator configuration.
    evaluation_mode : {"per_shot", "per_bin"}, default="per_shot"
        Mode of evaluation, either per individual shot or aggregated across shots.
    xopt_config_overrides : dict of str, Any
        Dictionary of optional overrides for Xopt configuration.
    device_requirements : dict, optional
        Device requirements dictionary. If None and evaluator.kwargs contains
        'analyzers', this will be auto-generated.
    save_devices : SaveDeviceConfig, optional
        Schema defining device saving strategies and workflow actions.
    save_devices_file : str or Path, optional
        Path to an external YAML file containing a SaveDeviceConfig definition.
    name : str, optional
        Optional name for this optimization configuration.
    description : str, optional
        Optional descriptive text for this optimization configuration.

    Notes
    -----
    This model allows arbitrary types such as `VOCS` by setting
    `arbitrary_types_allowed=True` in the Pydantic config.

    The `vocs` field is not validated or serialized by Pydantic and must be
    constructed manually.

    Raises
    ------
    ValueError
        If `vocs.variables` or `vocs.objectives` is empty.
    """

    vocs: VOCS
    evaluator: EvaluatorConfig
    generator: GeneratorConfig

    evaluation_mode: Literal["per_shot", "per_bin"] = "per_shot"
    xopt_config_overrides: Dict[str, Any] = Field(default_factory=dict)

    device_requirements: Optional[Dict] = None
    save_devices: Optional[SaveDeviceConfig] = None
    save_devices_file: Optional[Union[str, Path]] = None

    name: Optional[str] = None
    description: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("save_devices_file")
    @classmethod
    def _expand_path(cls, v):
        """
        Expand and normalize the save_devices_file path.

        Parameters
        ----------
        v : str or Path or None
            Path value from the configuration.

        Returns
        -------
        Path or None
            Expanded absolute path if provided, otherwise None.
        """
        return Path(v).expanduser().resolve() if v is not None else v

    @model_validator(mode="after")
    def _load_and_check(self) -> "BaseOptimizerConfig":
        """
        Post-validation hook for additional checks and defaults.

        Loads a SaveDeviceConfig from an external file if specified,
        auto-generates device_requirements from evaluator kwargs if needed,
        and performs basic sanity checks on the VOCS.

        Returns
        -------
        BaseOptimizerConfig
            The validated and updated configuration object.

        Raises
        ------
        ValueError
            If `vocs.variables` or `vocs.objectives` is empty.
        """
        # Load save_devices from file if specified
        if self.save_devices is None and self.save_devices_file is not None:
            with open(self.save_devices_file, "r") as f:
                loaded = yaml.safe_load(f)
            self.save_devices = SaveDeviceConfig.model_validate(loaded)

        # Auto-generate device_requirements from evaluator kwargs if needed
        if self.device_requirements is None:
            evaluator_kwargs = self.evaluator.kwargs
            if "analyzers" in evaluator_kwargs:
                # Create MultiDeviceScanEvaluatorConfig to generate requirements
                evaluator_config = MultiDeviceScanEvaluatorConfig(
                    analyzers=evaluator_kwargs["analyzers"]
                )
                self.device_requirements = (
                    evaluator_config.generate_device_requirements()
                )

        # Validate VOCS
        if not self.vocs.variables:
            raise ValueError("vocs.variables must not be empty.")

        # BAX generators don't require objectives (they model observables only)
        # Only validate objectives for non-BAX generators
        if (
            not self.vocs.objectives
            and self.generator.name != "multipoint_bax_alignment"
        ):
            raise ValueError(
                "vocs.objectives must not be empty for non-BAX generators. "
                "BAX generators (e.g., multipoint_bax_alignment) model observables only."
            )

        return self

    def evaluator_import_path(self) -> tuple[str, str]:
        """
        Return the evaluator import path.

        Returns
        -------
        tuple of (str, str)
            Module name and class name for the evaluator.
        """
        return self.evaluator.module, self.evaluator.class_
