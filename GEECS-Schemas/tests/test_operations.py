"""OperatorPolicy denies by default and grants by family, device and role; JournalEntry round-trips."""

import pytest
from pydantic import ValidationError

from geecs_schemas import JournalEntry, OperatorPolicy

POLICY = OperatorPolicy.model_validate(
    {
        "grants": [
            {"operators": ["ann"], "families": ["motor_record"]},
            {"operators": ["bob"], "devices": ["fps1"], "roles": ["current_sp"]},
        ]
    }
)


def test_the_empty_policy_permits_nothing():
    assert not OperatorPolicy().permits("ann", "m1", "motor_record", "twf")


@pytest.mark.parametrize(
    ("who", "device", "family", "role", "allowed"),
    [
        ("ann", "SimMotor1", "motor_record", "twf", True),  # by family, every role
        ("ann", "fps1", "fastps", "current_sp", False),  # not her family
        ("bob", "fps1", "fastps", "current_sp", True),  # by device and role
        ("bob", "fps1", "fastps", "output_sp", False),  # not his role
    ],
)
def test_a_grant_names_operator_scope_and_role(who, device, family, role, allowed):
    assert POLICY.permits(who, device, family, role) is allowed


def test_a_grant_without_a_scope_is_refused():
    with pytest.raises(ValidationError, match="families or devices"):
        OperatorPolicy.model_validate({"grants": [{"operators": ["ann"]}]})


def test_a_journal_row_round_trips_through_json():
    row = {
        "time": "2026-10-28T21:14:03Z",
        "operator": "ann",
        "device": "fps1",
        "role": "current_sp",
        "value": 1.5,
        "outcome": "refused",
    }
    entry = JournalEntry.model_validate(row)
    assert JournalEntry.model_validate_json(entry.model_dump_json()) == entry
