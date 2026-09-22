"""JSON persistence for interlock device and monitor configurations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .interlock_constructor import InterlockConstructor
from .monitor_conditions import (
    AlignmentCheck,
    ConfigurationChangeCheck,
    MultiCheck,
    ThresholdCheck,
)


def _condition_to_dict(condition: Any) -> dict[str, Any]:
    if isinstance(condition, ThresholdCheck):
        return {
            "type": "threshold",
            "device": condition.device_name,
            "variable": condition.variable_name,
            "operator": condition.operator,
            "threshold": condition.threshold,
        }
    if isinstance(condition, AlignmentCheck):
        return {
            "type": "alignment",
            "device": condition.device_name,
            "value_variable": condition.value_variable,
            "target_variable": condition.target_variable,
            "tolerance": condition.tolerance,
        }
    if isinstance(condition, ConfigurationChangeCheck):
        return {
            "type": "configuration_change",
            "device": condition.device_name,
            "variables": list(condition.variables),
        }
    if isinstance(condition, MultiCheck):
        return {
            "type": "multi",
            "conditions": [_condition_to_dict(child) for child in condition.conditions],
        }
    raise ValueError(f"Unsupported condition type: {type(condition).__name__}")


def _condition_from_dict(
    condition_data: dict[str, Any], interlock: InterlockConstructor
) -> Any:
    condition_type = condition_data.get("type")
    group = interlock.device_group
    if condition_type == "threshold":
        return ThresholdCheck(
            group,
            condition_data["device"],
            condition_data["variable"],
            float(condition_data["threshold"]),
            operator=condition_data["operator"],
        )
    if condition_type == "alignment":
        tolerance = condition_data["tolerance"]
        if isinstance(tolerance, list):
            tolerance = [float(value) for value in tolerance]
        else:
            tolerance = float(tolerance)
        return AlignmentCheck(
            group,
            condition_data["device"],
            condition_data["value_variable"],
            condition_data["target_variable"],
            tolerance,
        )
    if condition_type == "configuration_change":
        return ConfigurationChangeCheck(
            group, condition_data["device"], list(condition_data["variables"])
        )
    if condition_type == "multi":
        return MultiCheck(
            [
                _condition_from_dict(child, interlock)
                for child in condition_data["conditions"]
            ]
        )
    raise ValueError(f"Unsupported condition type: {condition_type}")


def save_configuration(interlock: InterlockConstructor, path: str | Path) -> None:
    """Save an interlock constructor configuration as JSON."""
    monitors = []
    for name, (
        check,
        interval,
        safe_release_checks,
    ) in interlock.monitor_registrations.items():
        monitors.append(
            {
                "name": name,
                "interval": interval,
                "safe_release_checks": safe_release_checks,
                "condition": _condition_to_dict(check),
            }
        )

    configuration = {
        "format": "geecs-interlock",
        "version": 1,
        "experiment": interlock.exp_name,
        "host": interlock.host,
        "port": interlock.port,
        "devices": [
            {
                "alias": alias,
                "name": interlock.device_group.device_names[alias],
                "variables": list(
                    interlock.device_group.device_variables.get(alias, [])
                ),
            }
            for alias in interlock.device_group.devices
        ],
        "monitors": monitors,
    }
    Path(path).write_text(json.dumps(configuration, indent=2), encoding="utf-8")


def load_configuration(interlock: InterlockConstructor, path: str | Path) -> None:
    """Replace a stopped interlock constructor with a saved JSON configuration."""
    if interlock.server is not None:
        raise RuntimeError("Stop the interlock before importing a configuration")

    configuration = json.loads(Path(path).read_text(encoding="utf-8"))
    if configuration.get("format") != "geecs-interlock":
        raise ValueError("This file is not a GEECS interlock configuration")
    if configuration.get("version") != 1:
        raise ValueError("Unsupported interlock configuration version")
    configured_experiment = str(configuration.get("experiment", "")).strip()
    selected_experiment = str(interlock.exp_name).strip()
    if configured_experiment.casefold() != selected_experiment.casefold():
        raise ValueError(
            f"Configuration is for experiment '{configured_experiment}', "
            f"but the GUI is using '{interlock.exp_name}'"
        )

    interlock.monitor_registrations.clear()
    for alias in list(interlock.device_group.devices):
        interlock.device_group.remove_device(alias)

    for device_data in configuration.get("devices", []):
        device = interlock.add_device(
            device_data["alias"], device_data["name"], list(device_data["variables"])
        )
        device.use_alias_in_TCP_subscription = False

    for monitor in configuration.get("monitors", []):
        condition = _condition_from_dict(monitor["condition"], interlock)
        interlock.add_monitor(
            monitor["name"],
            condition,
            interval=float(monitor.get("interval", 0.1)),
            safe_release_checks=int(monitor.get("safe_release_checks", 3)),
        )
