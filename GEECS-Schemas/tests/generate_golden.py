"""Regenerate the golden snapshots in ``tests/golden/`` from the fixtures.

Run after an *intentional* schema or converter change, then review the diff:

    poetry run python GEECS-Schemas/tests/generate_golden.py
"""

import json
from pathlib import Path

from geecs_schemas.convert import (
    convert_action_library,
    convert_optimizer_config,
    convert_save_element,
    convert_scan_preset,
    convert_scan_variables,
    convert_shot_control,
    merge_trigger_variant,
)

TESTS = Path(__file__).parent
FIXTURES = TESTS / "fixtures"
GOLDEN = TESTS / "golden"


def write(name: str, payload: dict) -> None:
    """Write one golden JSON snapshot."""
    GOLDEN.mkdir(exist_ok=True)
    path = GOLDEN / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {path}")


def main() -> None:
    """Regenerate every golden snapshot."""
    aline = convert_save_element(FIXTURES / "save_elements/UC_Aline1.yaml")
    write(
        "UC_Aline1.converted.json",
        {
            "save_set": aline.save_set.model_dump(mode="json"),
            "actions": {k: v.model_dump(mode="json") for k, v in aline.actions.items()},
            "notes": aline.notes,
        },
    )

    visa = convert_save_element(
        FIXTURES / "save_elements/visa1_spectrometer_setup.yaml"
    )
    write(
        "visa1_spectrometer_setup.converted.json",
        {
            "save_set": visa.save_set.model_dump(mode="json"),
            "actions": {k: v.model_dump(mode="json") for k, v in visa.actions.items()},
        },
    )

    catalog = convert_scan_variables(
        FIXTURES / "scan_devices/scan_devices.yaml",
        FIXTURES / "scan_devices/composite_variables.yaml",
    )
    write("thomson_scan_variables.json", catalog.model_dump(mode="json"))

    undulator = convert_scan_variables(
        FIXTURES / "scan_devices/scan_devices_undulator.yaml",
        FIXTURES / "scan_devices/composite_variables_undulator.yaml",
    )
    write("undulator_scan_variables.json", undulator.model_dump(mode="json"))

    base = convert_shot_control(FIXTURES / "shot_control/HTU-Normal.yaml")
    off = convert_shot_control(FIXTURES / "shot_control/HTU-LaserOFF.yaml")
    profile = merge_trigger_variant(base, off, "laser_off")
    write("htu_trigger_profile.json", profile.model_dump(mode="json"))

    library = convert_action_library(FIXTURES / "actions/actions_undulator.yaml")
    write(
        "amp4_dump_hp_plan.json",
        library.plans["Amp4_DUMP_HP"].model_dump(mode="json"),
    )

    preset = convert_scan_preset(FIXTURES / "presets/00_focuscan.yaml")
    write(
        "focuscan_scan_request.json",
        preset.scan_request.model_dump(mode="json"),
    )

    optimizer = convert_optimizer_config(
        FIXTURES / "optimizer_configs/hexapod_alignment.yaml"
    )
    write(
        "hexapod_optimization_spec.json",
        optimizer.optimization.model_dump(mode="json"),
    )


if __name__ == "__main__":
    main()
