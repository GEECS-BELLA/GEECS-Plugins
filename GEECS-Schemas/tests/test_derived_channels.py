"""Model tests for derived CA gateway channels."""

import pytest
from pydantic import ValidationError

from geecs_schemas import DerivedChannels


def make_document() -> DerivedChannels:
    return DerivedChannels.model_validate(
        {
            "derived_channels": [
                {
                    "device": "U_ChamberVac",
                    "variable": "Pressure",
                    "expression": "10**(v - 5)",
                    "inputs": [
                        {
                            "symbol": "v",
                            "device": "U_DaqPad1",
                            "variable": "Analog Input 10",
                        }
                    ],
                    "egu": "Torr",
                    "precision": 3,
                }
            ]
        }
    )


class TestDerivedChannels:
    def test_round_trip(self):
        document = make_document()
        again = DerivedChannels.model_validate(document.model_dump(mode="json"))
        assert again == document

    def test_same_source_device_enforced(self):
        with pytest.raises(ValidationError, match="one source device"):
            DerivedChannels.model_validate(
                {
                    "derived_channels": [
                        {
                            "device": "U_ChamberVac",
                            "variable": "PressureRatio",
                            "expression": "a / b",
                            "inputs": [
                                {
                                    "symbol": "a",
                                    "device": "U_DaqPad1",
                                    "variable": "Analog Input 10",
                                },
                                {
                                    "symbol": "b",
                                    "device": "U_DaqPad2",
                                    "variable": "Analog Input 1",
                                },
                            ],
                        }
                    ]
                }
            )

    def test_input_symbols_must_be_unique(self):
        with pytest.raises(ValidationError, match="symbols must be unique"):
            DerivedChannels.model_validate(
                {
                    "derived_channels": [
                        {
                            "device": "U_ChamberVac",
                            "variable": "Pressure",
                            "expression": "a + a",
                            "inputs": [
                                {
                                    "symbol": "a",
                                    "device": "U_DaqPad1",
                                    "variable": "Analog Input 10",
                                },
                                {
                                    "symbol": "a",
                                    "device": "U_DaqPad1",
                                    "variable": "Analog Input 11",
                                },
                            ],
                        }
                    ]
                }
            )

    def test_unknown_field_fails_loudly(self):
        with pytest.raises(ValidationError, match="units"):
            DerivedChannels.model_validate(
                {
                    "derived_channels": [
                        {
                            "device": "U_ChamberVac",
                            "variable": "Pressure",
                            "expression": "v",
                            "inputs": [
                                {
                                    "symbol": "v",
                                    "device": "U_DaqPad1",
                                    "variable": "Analog Input 10",
                                }
                            ],
                            "units": "Torr",
                        }
                    ]
                }
            )
