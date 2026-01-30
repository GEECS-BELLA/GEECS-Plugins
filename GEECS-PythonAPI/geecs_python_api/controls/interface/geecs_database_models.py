"""Pydantic models for type-safe GEECS database entities.

These models provide structured, validated representations of experiment data
with proper typing, enabling better IDE support and cleaner APIs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


def _empty_string_to_none(v: Any) -> Any:
    """Convert empty strings to None (database stores '' instead of NULL)."""
    if v == "" or v == "":
        return None
    return v


class Variable(BaseModel):
    """Represents a device variable with its configuration and constraints.

    Attributes
    ----------
        name: Variable name (e.g., 'current', 'position')
        device_name: Parent device name
        default_value: Default/initial value
        min: Minimum allowed value
        max: Maximum allowed value
        step_size: Recommended step size for changes
        units: Physical units (e.g., 'A', 'mm')
        choices: Enumerated choices if this is a discrete variable
        tolerance: Tolerance for value comparisons
        alias: Human-readable alias
    """

    name: str
    device_name: str
    default_value: Optional[str] = Field(default=None, alias="defaultvalue")
    min: Optional[float] = None
    max: Optional[float] = None
    step_size: Optional[float] = Field(default=None, alias="stepsize")
    units: Optional[str] = None
    choices: Optional[str] = None
    tolerance: Optional[float] = None
    alias: Optional[str] = None
    settable: Optional[bool] = Field(default=None, alias="set")

    # Raw data from database for backward compatibility
    raw_data: Dict[str, Any] = Field(default_factory=dict, exclude=True)

    model_config = {"populate_by_name": True, "extra": "allow"}

    @field_validator("min", "max", "step_size", "tolerance", mode="before")
    @classmethod
    def parse_numeric_field(cls, v: Any) -> Any:
        r"""Parse numeric fields that may contain empty strings or multi-line values.

        The GEECS database has some quirks:
        - Empty strings '' instead of NULL
        - Some fields contain newline-separated values like '0\n1.0'
        """
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return v
        if isinstance(v, str):
            v = v.strip()
            if v == "":
                return None
            # Handle multi-line values - take the first line
            if "\n" in v:
                v = v.split("\n")[0].strip()
            # Try to parse as float
            try:
                return float(v)
            except ValueError:
                return None
        return v

    @field_validator("settable", mode="before")
    @classmethod
    def parse_settable_field(cls, v: Any) -> Any:
        """Parse settable field which may be 'yes'/'no' string or boolean."""
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ("yes", "true", "1")
        if isinstance(v, int):
            return bool(v)
        return None

    @classmethod
    def from_db_row(cls, row: Dict[str, Any]) -> Variable:
        """Create a Variable from a database row dictionary."""
        return cls(
            name=row.get("variablename", ""),
            device_name=row.get("devicename", ""),
            defaultvalue=row.get("defaultvalue"),
            min=row.get("min"),
            max=row.get("max"),
            stepsize=row.get("stepsize"),
            units=row.get("units"),
            choices=row.get("choices"),
            tolerance=row.get("tolerance"),
            alias=row.get("alias"),
            set=row.get("set"),
            raw_data=row,
        )

    def __repr__(self) -> str:
        """Concise representation showing key variable properties."""
        parts = [f"Variable('{self.name}'"]
        if self.min is not None or self.max is not None:
            parts.append(f", range=[{self.min}, {self.max}]")
        if self.units:
            parts.append(f", units='{self.units}'")
        if self.step_size is not None:
            parts.append(f", step={self.step_size}")
        parts.append(")")
        return "".join(parts)


class Device(BaseModel):
    """Represents a GEECS device with its variables and connection info.

    Attributes
    ----------
        name: Device name (e.g., 'U_EMQ1MagnetPS')
        description: Human-readable description of the device
        device_type: Type of device (e.g., 'MagnetPowerSupply')
        ip_address: Network IP address
        port: Communication port
        variables: Dictionary of variable name -> Variable
    """

    name: str
    description: Optional[str] = None
    device_type: Optional[str] = None
    ip_address: Optional[str] = Field(default=None, alias="ipaddress")
    port: Optional[int] = Field(default=None, alias="commport")
    variables: Dict[str, Variable] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}

    def get_variable(self, var_name: str) -> Optional[Variable]:
        """Get a variable by name, or None if not found."""
        return self.variables.get(var_name)

    def list_variable_names(self) -> List[str]:
        """Return list of all variable names for this device."""
        return list(self.variables.keys())

    def __repr__(self) -> str:
        """Concise representation showing device summary."""
        desc = f", desc='{self.description}'" if self.description else ""
        return (
            f"Device('{self.name}', type='{self.device_type}'{desc}, "
            f"variables={len(self.variables)})"
        )


class Experiment(BaseModel):
    """Represents a complete GEECS experiment with all its devices.

    This is the main entry point for accessing experiment configuration.
    Use GeecsDatabase.load_experiment() to create an instance.

    Attributes
    ----------
        name: Experiment name (e.g., 'Undulator')
        devices: Dictionary of device name -> Device
        data_path: Root path for experiment data
        mc_port: Multicast UDP port for slow controls
    """

    name: str
    devices: Dict[str, Device] = Field(default_factory=dict)
    data_path: Optional[Path] = None
    mc_port: Optional[int] = None

    def get_device(self, device_name: str) -> Optional[Device]:
        """Get a device by name, or None if not found."""
        return self.devices.get(device_name)

    def list_device_names(self) -> List[str]:
        """Return list of all device names in this experiment."""
        return list(self.devices.keys())

    def list_devices_by_type(self, device_type: str) -> List[Device]:
        """Return list of devices matching the given type."""
        return [d for d in self.devices.values() if d.device_type == device_type]

    def get_variable(self, device_name: str, var_name: str) -> Optional[Variable]:
        """Get a specific variable from a device.

        Args:
            device_name: Name of the device
            var_name: Name of the variable

        Returns
        -------
            Variable if found, None otherwise
        """
        device = self.get_device(device_name)
        if device:
            return device.get_variable(var_name)
        return None

    def find_devices_with_variable(self, var_name: str) -> List[Device]:
        """Find all devices that have a variable with the given name."""
        return [d for d in self.devices.values() if var_name in d.variables]

    def __repr__(self) -> str:
        """Concise representation showing experiment summary."""
        total_vars = sum(len(d.variables) for d in self.devices.values())
        return (
            f"Experiment('{self.name}', devices={len(self.devices)}, "
            f"total_variables={total_vars})"
        )
