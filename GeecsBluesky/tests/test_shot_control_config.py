"""Tests for the ShotControlConfig model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from geecs_bluesky.models.shot_control import ShotControlConfig, ShotControlState

# The real U_DG645_ShotControl YAML shape
_INFO = {
    "device": "U_DG645_ShotControl",
    "variables": {
        "Trigger.ExecuteSingleShot": {
            "OFF": "",
            "SCAN": "",
            "STANDBY": "",
            "SINGLESHOT": "on",
        },
        "Trigger.Source": {
            "OFF": "Single shot external rising edges",
            "SCAN": "External rising edges",
            "STANDBY": "External rising edges",
            "SINGLESHOT": "",
        },
    },
}


def test_from_information_validates_dict() -> None:
    cfg = ShotControlConfig.from_information(_INFO)
    assert isinstance(cfg, ShotControlConfig)
    assert cfg.device == "U_DG645_ShotControl"
    assert "Trigger.Source" in cfg.variables


def test_from_information_passthrough_and_none() -> None:
    cfg = ShotControlConfig.from_information(_INFO)
    assert ShotControlConfig.from_information(cfg) is cfg
    assert ShotControlConfig.from_information(None) is None


def test_from_information_empty_is_no_shot_control() -> None:
    """Empty/blank config (e.g. Bella's `{}` YAML) → None, not a crash."""
    assert ShotControlConfig.from_information({}) is None


def test_from_information_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        ShotControlConfig.from_information({"device": "X", "variables": {}, "junk": 1})


def test_values_for_state_skips_empty_noops() -> None:
    cfg = ShotControlConfig.from_information(_INFO)
    # SCAN: only Trigger.Source has a non-empty value
    assert cfg.values_for_state("SCAN") == {"Trigger.Source": "External rising edges"}
    # SINGLESHOT: only ExecuteSingleShot fires
    assert cfg.values_for_state(ShotControlState.SINGLESHOT) == {
        "Trigger.ExecuteSingleShot": "on"
    }
    # OFF arms single-shot mode on the source
    assert cfg.values_for_state("OFF") == {
        "Trigger.Source": "Single shot external rising edges"
    }


def test_defines_state() -> None:
    cfg = ShotControlConfig.from_information(_INFO)
    assert cfg.defines_state(ShotControlState.SINGLESHOT)
    assert cfg.defines_state("SCAN")
    # A state nobody has a non-empty value for is "not defined"
    cfg2 = ShotControlConfig(
        device="X",
        variables={"V": {"SCAN": "on", "SINGLESHOT": ""}},
    )
    assert cfg2.defines_state("SCAN")
    assert not cfg2.defines_state("SINGLESHOT")
    assert not cfg2.defines_state("STANDBY")


def test_armed_state_for_full_power_single_shot() -> None:
    """An ARMED state (jet on + single-shot source) drives full-power single-shot."""
    cfg = ShotControlConfig.from_information(
        {
            "device": "U_DG645_ShotControl",
            "variables": {
                "Amplitude.Ch AB": {"SCAN": "4.0", "STANDBY": "0.5", "ARMED": "4.0"},
                "Trigger.Source": {
                    "SCAN": "External rising edges",
                    "ARMED": "Single shot external rising edges",
                },
                "Trigger.ExecuteSingleShot": {"SINGLESHOT": "on"},
            },
        }
    )
    assert cfg.defines_state(ShotControlState.ARMED)
    # ARMED = jet on (full amplitude) + single-shot source (stops free-run)
    assert cfg.values_for_state(ShotControlState.ARMED) == {
        "Amplitude.Ch AB": "4.0",
        "Trigger.Source": "Single shot external rising edges",
    }
    # SINGLESHOT fires; amplitude is left at the ARMED value (no-op here)
    assert cfg.values_for_state(ShotControlState.SINGLESHOT) == {
        "Trigger.ExecuteSingleShot": "on"
    }


def test_state_enum_values() -> None:
    assert ShotControlState.SINGLESHOT.value == "SINGLESHOT"
    assert {s.value for s in ShotControlState} == {
        "OFF",
        "SCAN",
        "STANDBY",
        "SINGLESHOT",
        "ARMED",
    }
