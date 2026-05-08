"""ScanExecutionConfig — the validated container the GUI hands to the scan engine."""

from __future__ import annotations

from typing import Optional

from geecs_data_utils import ScanConfig
from pydantic import BaseModel, ConfigDict, Field

from geecs_scanner.engine.models.save_devices import SaveDeviceConfig
from geecs_scanner.engine.models.scan_options import ScanOptions


class ScanExecutionConfig(BaseModel):
    """Complete, validated specification for one scan run.

    This is what the GUI produces and what the engine consumes. Replaces
    the raw ``config_dictionary`` dict that previously carried device config,
    options, and action sequences in an untyped bag.

    Attributes
    ----------
    scan_config : ScanConfig
        Scan parameters — variable, start/stop/step, scan mode.
    options : ScanOptions
        Engine-level execution options (rep rate, time sync, TDMS, etc.).
    save_config : SaveDeviceConfig
        Device saving configuration including setup/closeout action sequences.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    scan_config: ScanConfig
    options: ScanOptions = Field(default_factory=ScanOptions)
    save_config: SaveDeviceConfig = Field(default_factory=SaveDeviceConfig)

    @classmethod
    def from_gui_dict(
        cls,
        config_dict: dict,
        scan_config: ScanConfig,
        options: Optional[ScanOptions] = None,
    ) -> "ScanExecutionConfig":
        """Construct from the dict the GUI currently builds.

        Parameters
        ----------
        config_dict : dict
            Raw ``run_config`` dict with keys ``Devices``, ``scan_info``, and
            optionally ``setup_action``, ``closeout_action``.
        scan_config : ScanConfig
            Scan parameters.
        options : ScanOptions, optional
            Execution options.  Overrides any ``"options"`` key in
            *config_dict* if provided.
        """
        raw = {k: v for k, v in config_dict.items() if k != "options"}
        effective_options = options if options is not None else ScanOptions()
        return cls(
            scan_config=scan_config,
            options=effective_options,
            save_config=SaveDeviceConfig(**raw),
        )

    def to_device_manager_dict(self) -> dict:
        """Return the dict format DeviceManager.load_from_dictionary() expects."""
        return self.save_config.model_dump(exclude_none=True)
