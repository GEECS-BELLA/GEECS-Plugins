"""Record ``devicetype_variable`` rows into ``tests/fixtures/devicetype_variables.json``.

The fixture is the offline stand-in the parity test in
``tests/test_device_streams.py`` checks the capture declaration
(``geecs_core.db.device_streams``) against.  Run this on the lab network,
with the usual ``config.ini`` DB credentials, from ``GEECS-Core/``::

    poetry run python scripts/record_devicetype_variables.py "ThorlabsWFS"

Names are the DB's ``devicetype`` spellings; with none given, every
devicetype already in the fixture is re-recorded.  Rows are ``name``,
``variabletype`` and the ``choice`` table's text — type-level only, no
per-instance merge — sorted by name.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from geecs_core.db import GeecsDb

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "devicetype_variables.json"
)


def main(argv: list[str] | None = None) -> int:
    """Record the named devicetypes (default: those already in the fixture) and rewrite it."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "devicetype", nargs="*", help="DB devicetype names to (re)record"
    )
    args = parser.parse_args(argv)

    data: dict = (
        json.loads(FIXTURE.read_text()) if FIXTURE.exists() else {"devicetypes": {}}
    )
    names = args.devicetype or list(data["devicetypes"])
    if not names:
        parser.error("no devicetypes named and the fixture is empty")
    for name in names:
        rows = GeecsDb.get_devicetype_variables(name)
        if not rows:
            print(
                f"no devicetype_variable rows for {name!r} — check the spelling",
                file=sys.stderr,
            )
            return 1
        data["devicetypes"][name] = rows
        print(f"{name}: {len(rows)} rows")
    data = {
        "_recorded": (
            "devicetype_variable rows (name, variabletype, choice text) per "
            "devicetype, read from the GEECS DB by "
            "scripts/record_devicetype_variables.py, last "
            f"{dt.date.today().isoformat()}; the offline stand-in for "
            "tests/test_device_streams.py. Re-run that script when a "
            "devicetype's variables change or a table entry is added."
        ),
        "devicetypes": data["devicetypes"],
    }
    FIXTURE.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {FIXTURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
