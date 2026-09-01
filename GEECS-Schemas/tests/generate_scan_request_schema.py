"""Regenerate the committed ScanRequest JSON Schema artifact.

Run after an *intentional* change to the ScanRequest vocabulary, then review
the diff and commit the updated artifact:

    poetry run python GEECS-Schemas/tests/generate_scan_request_schema.py

This is the same output the no-drift guard (``tests/test_schema_export.py``)
checks, so a green ``pytest`` after regenerating means the artifact is in
step.
"""

from geecs_schemas.schema_export import write_artifact


def main() -> None:
    """Write ``docs/geecs_schemas/scan_request.schema.json`` from the schema."""
    written = write_artifact()
    print(f"wrote {written}")


if __name__ == "__main__":
    main()
