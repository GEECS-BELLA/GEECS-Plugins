"""Regenerate every committed JSON Schema artifact (the export registry).

Run after an *intentional* change to a published schema, then review the
diff and commit the updated artifacts:

    poetry run python GEECS-Schemas/tests/generate_schema_artifacts.py

This is the same output the no-drift guard (``tests/test_schema_export.py``)
checks for every :data:`geecs_schemas.schema_export.EXPORTED_SCHEMAS` entry,
so a green ``pytest`` after regenerating means the artifacts are in step.
"""

from geecs_schemas.schema_export import write_artifacts


def main() -> None:
    """Write ``docs/geecs_schemas/<name>.schema.json`` for every registry entry."""
    for written in write_artifacts():
        print(f"wrote {written}")


if __name__ == "__main__":
    main()
