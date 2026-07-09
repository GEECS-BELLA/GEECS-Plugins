"""Regenerate the committed schema-reference docs page from the schemas.

Run after an *intentional* change to any schema field description, then review
the diff and commit the updated page:

    poetry run python GEECS-Schemas/tests/generate_schema_reference.py

This is the same output the no-drift guard (``tests/test_schema_reference.py``)
checks, so a green ``pytest`` after regenerating means the page is in step.
"""

from geecs_schemas.docgen import write_page


def main() -> None:
    """Write ``docs/geecs_schemas/schema_reference.md`` from the schemas."""
    written = write_page()
    print(f"wrote {written}")


if __name__ == "__main__":
    main()
