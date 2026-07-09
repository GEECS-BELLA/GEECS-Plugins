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
