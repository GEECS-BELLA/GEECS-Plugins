"""Model tests for derived CA gateway channels."""

import pytest
from pydantic import ValidationError

from geecs_schemas import DerivedChannels


def make_document() -> DerivedChannels:
    return DerivedChannels.model_validate(
        {
            "derived_channels": [
                {
                    "device": "TargetChamberPressure",
                    "variable": "Pressure",
                    "expression": "10**(v - 6)",
                    "inputs": [
                        {
                            "symbol": "v",
                            "device": "U_VacuumGauge",
                            "variable": "AI_mean.Channel 0",
                        }
                    ],
                    "egu": "Torr",
                    "precision": 6,
                }
            ]
        }
    )


class TestDerivedChannels:
    def test_round_trip(self):
        document = make_document()
        again = DerivedChannels.model_validate(document.model_dump(mode="json"))
        assert again == document

    def test_cross_device_requires_stale_after(self):
        with pytest.raises(ValidationError, match="stale_after is required"):
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

    def test_cross_device_with_stale_after_is_allowed(self):
        document = DerivedChannels.model_validate(
            {
                "derived_channels": [
                    {
                        "device": "LaserPermit",
                        "variable": "OK",
                        "expression": "pressure < 1e-5 and ready > 0",
                        "inputs": [
                            {
                                "symbol": "pressure",
                                "device": "TargetChamberPressure",
                                "variable": "Pressure",
                            },
                            {
                                "symbol": "ready",
                                "device": "Amp4Shutter",
                                "variable": "Ready",
                            },
                        ],
                        "stale_after": 2.0,
                    }
                ]
            }
        )

        [channel] = document.derived_channels
        assert channel.is_cross_device
        assert channel.source_devices == {"TargetChamberPressure", "Amp4Shutter"}
        assert channel.stale_after == 2.0

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
