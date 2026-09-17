"""Model tests for ScanVariables."""

import pytest
from pydantic import ValidationError

from geecs_schemas import (
    CompositeMode,
    PseudoComponent,
    PseudoScanVariable,
    ScanVariable,
    ScanVariables,
)


def make_catalog():
    return ScanVariables.model_validate(
        {
            "variables": {
                "jet_z": {
                    "target": "U_ESP_JetXYZ:Position.Axis 3",
                    "kind": "motor",
                },
                "gas_pressure": {"target": "U_HP_Daq:AnalogOutput.Channel 1"},
                "e_beam_angle_x": {
                    "kind": "pseudo",
                    "mode": "relative",
                    "targets": [
                        {"target": "U_S3H:Current", "forward": "composite_var * 1"},
                        {"target": "U_S4H:Current", "forward": "composite_var * -2"},
                    ],
                },
            }
        }
    )


class TestScanVariables:
    def test_round_trip(self):
        catalog = make_catalog()
        again = ScanVariables.model_validate(catalog.model_dump(mode="json"))
        assert again == catalog

    def test_kind_defaults_to_setpoint(self):
        catalog = make_catalog()
        simple = catalog.variables["gas_pressure"]
        assert isinstance(simple, ScanVariable)
        assert simple.kind == "setpoint"

    def test_pseudo_resolves(self):
        pseudo = make_catalog().variables["e_beam_angle_x"]
        assert isinstance(pseudo, PseudoScanVariable)
        assert pseudo.mode is CompositeMode.RELATIVE
        assert isinstance(pseudo.targets[1], PseudoComponent)
        assert pseudo.targets[1].forward == "composite_var * -2"

    def test_confirm_defaults_to_none(self):
        simple = make_catalog().variables["gas_pressure"]
        assert isinstance(simple, ScanVariable)
        assert simple.confirm is None

    def test_confirm_accepts_measured_variable(self):
        # Topology C: set a software current limit, confirm on measured current.
        catalog = ScanVariables.model_validate(
            {
                "variables": {
                    "EMQ1 Current": {
                        "target": "U_EMQTripletBipolar:Current_Limit.Ch1",
                        "confirm": "U_EMQTripletBipolar:Current.Ch1",
                    }
                }
            }
        )
        entry = catalog.variables["EMQ1 Current"]
        assert isinstance(entry, ScanVariable)
        assert entry.confirm == "U_EMQTripletBipolar:Current.Ch1"

    def test_confirm_shape_enforced(self):
        with pytest.raises(ValidationError, match="Device:Variable"):
            ScanVariables.model_validate(
                {"variables": {"bad": {"target": "A:B", "confirm": "no-colon-here"}}}
            )

    def test_target_shape_enforced(self):
        with pytest.raises(ValidationError, match="Device:Variable"):
            ScanVariables.model_validate(
                {"variables": {"bad": {"target": "no-colon-here"}}}
            )

    def test_unknown_field_fails_loudly(self):
        with pytest.raises(ValidationError, match="tolerance"):
            ScanVariables.model_validate(
                {"variables": {"jet_z": {"target": "A:B", "tolerance": 0.1}}}
            )

    def test_pseudo_requires_targets(self):
        with pytest.raises(ValidationError):
            ScanVariables.model_validate(
                {
                    "variables": {
                        "empty": {"kind": "pseudo", "mode": "absolute", "targets": []}
                    }
                }
            )

    def test_bad_mode_rejected(self):
        with pytest.raises(ValidationError):
            ScanVariables.model_validate(
                {
                    "variables": {
                        "x": {
                            "kind": "pseudo",
                            "mode": "offset",
                            "targets": [{"target": "A:B", "forward": "composite_var"}],
                        }
                    }
                }
            )


def test_split_device_variable_is_the_public_form_of_the_target_rule() -> None:
    """First ``:`` separates, both parts stripped and non-empty; anything else is a ValueError."""
    from geecs_schemas import split_device_variable

    assert split_device_variable(" U_ESP_JetXYZ : Position.Axis 3 ") == (
        "U_ESP_JetXYZ",
        "Position.Axis 3",
    )
    assert split_device_variable("U_S1H:Current:extra") == ("U_S1H", "Current:extra")
    for bad in ("U_S1H", ":Current", "U_S1H:", "U_S1H: ", "S1H current"):
        with pytest.raises(ValueError, match="Device:Variable"):
            split_device_variable(bad)


def test_description_and_inverse_round_trip() -> None:
    from geecs_schemas.scan_variables import ScanVariables

    catalog = ScanVariables.model_validate(
        {
            "schema_version": 1,
            "variables": {
                "S3H": {
                    "target": "U_S3H:Current",
                    "kind": "motor",
                    "description": "ALine steering",
                },
                "R56_at_100MeV": {
                    "kind": "pseudo",
                    "mode": "absolute",
                    "description": "chicane R56 at 100 MeV",
                    "targets": [
                        {"target": "U_ChicaneInner:Current", "forward": "sqrt(x)"}
                    ],
                    "inverse": "U_ChicaneInner**2",
                },
            },
        }
    )
    assert catalog.variables["S3H"].description == "ALine steering"
    pseudo = catalog.variables["R56_at_100MeV"]
    assert pseudo.description == "chicane R56 at 100 MeV"
    assert pseudo.inverse == "U_ChicaneInner**2"
    assert ScanVariables.model_validate(catalog.model_dump(mode="json")) == catalog
