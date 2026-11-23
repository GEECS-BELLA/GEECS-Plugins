"""EntryCreate, the logbook's create body: the one model that ignores unknown keys."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from geecs_schemas.log_entry import EntryCreate


def test_an_unknown_key_is_accepted_and_dropped() -> None:
    """A client one field ahead of the logbook can still write; the field is not kept."""
    body = EntryCreate.model_validate(
        {"day": "2026-07-12", "author": "Ada", "body_md": "", "added_later": 1}
    )
    assert "added_later" not in body.model_dump()
    assert not hasattr(body, "added_later")


def test_its_own_fields_still_validate() -> None:
    with pytest.raises(ValidationError):
        EntryCreate.model_validate({"day": "12 July", "author": "Ada", "body_md": ""})
