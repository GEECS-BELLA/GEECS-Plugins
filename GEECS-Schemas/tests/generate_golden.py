"""Regenerate the golden snapshots in ``tests/golden/`` from the fixtures.

Run after an *intentional* schema or converter change, then review the diff:

    poetry run python GEECS-Schemas/tests/generate_golden.py
"""

import json
from pathlib import Path

from geecs_schemas.convert import (
    convert_shot_control,
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
    profile = convert_shot_control(FIXTURES / "shot_control/HTU-Normal.yaml")
    write("htu_trigger_profile.json", profile.model_dump(mode="json"))


if __name__ == "__main__":
    main()
