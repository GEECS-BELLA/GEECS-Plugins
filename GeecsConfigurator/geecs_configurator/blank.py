"""A new document's starting draft, and where this box may put it.

The draft is the consumer's: the portal builds a recipe's blank from the
analysis core's registry (its input, no steps, the default measure for
those frames, ``output_name`` the name), so what opens already binds.
When the portal is not configured or does not answer, the draft is the
schema's fewest fields (no measure chosen: the registry is the core's) and
the route says so. Nothing here writes: a draft is saved through the save
route's create, with its dry run, its tier and its stem rule.
"""

from __future__ import annotations

import json
import re
from pathlib import PurePosixPath
from typing import Any

from geecs_schemas import SCHEMA_REGISTRY

from geecs_configurator.ownership import Ownership
from geecs_configurator.store import DocumentStore, NotFound
from geecs_configurator.validation import stem_twins

#: A draft as JSON.
Json = dict[str, Any]  # Any: a document as JSON, validated by its kind's model
#: A new document's name: its file stem (the copy's rule).
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def parse_fields(text: str) -> Json:
    """The input's fields beyond its kind, as the start form sends them (a JSON object); ``ValueError`` in words."""
    try:
        parsed = json.loads(text) if text else {}
    except ValueError as exc:
        raise ValueError(f"fields is not JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("fields is a JSON object (the input's fields beyond its kind)")
    return parsed


def schema_blank(kind: str, name: str, input_kind: str, device: str, fields: Json) -> Json:  # fmt: skip
    """The schema's fewest-field draft (the portal's absent): raises ``ValidationError`` when it does not validate."""
    if kind == "analysis_group":
        document: Json = {"name": name, "analyzers": []}
    else:
        document = {"schema_version": 3, "device": device or name, "output_name": name, "input": {"kind": input_kind, **fields}, "steps": []}  # fmt: skip
    SCHEMA_REGISTRY[kind].model_validate(document)
    return document


def taken(store: DocumentStore, kind: str, doc_id: str) -> str:
    """Why ``doc_id`` cannot be a new document (a file there, its stem another's in the analyzers tree); ``""`` when free."""
    try:
        store.read(kind, None, doc_id)
    except NotFound:
        pass
    else:
        return f"{doc_id} already exists: open it, or choose another name"
    twins = stem_twins(store, store.new_path(kind, None, doc_id))
    if twins:
        return f"{PurePosixPath(doc_id).stem} is already {twins[0]}'s name; a run loads a diagnostic by its name, so names are unique across the analyzers tree"  # fmt: skip
    return ""


def folders(store: DocumentStore, ownership: Ownership, kind: str) -> list[str]:
    """The folders of the kind's tree a new document may go in: this box's facility folders and the subfolders in them that hold the kind's documents."""
    found = {str(PurePosixPath(i).parent) for i in store.files(kind, None)} - {"."}
    out = []
    for folder in sorted(set(ownership.analysis_folders) | found):
        try:
            rel = store.new_path(kind, None, f"{folder}/_.yaml")
        except NotFound:
            continue
        if ownership.owns_file(store.root, rel):
            out.append(folder)
    return out
