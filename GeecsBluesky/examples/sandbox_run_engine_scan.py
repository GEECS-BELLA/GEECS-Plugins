"""Run a GeecsBluesky scan entirely against localhost fake hardware.

Usage
-----
From the ``GeecsBluesky`` package directory:

    poetry run python examples/sandbox_run_engine_scan.py
"""

from __future__ import annotations

from pprint import pprint

from geecs_bluesky.testing import run_fake_step_scan


def main() -> None:
    """Run the sandbox scan and print a compact document summary."""
    result = run_fake_step_scan()
    events = result.event_docs

    print(f"Run uid: {result.start_doc['uid']}")
    print(f"Plan: {result.start_doc['plan_name']}")
    print(f"Events: {len(events)}")
    print(f"Exit status: {result.stop_doc['exit_status']}")

    if events:
        print("\nFirst event data:")
        pprint(events[0]["data"], sort_dicts=True)


if __name__ == "__main__":
    main()
