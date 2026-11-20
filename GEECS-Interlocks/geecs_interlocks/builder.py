"""The builder's draft: an interlocks document being edited, apart from the one running.

Every edit the Builder tab makes goes through :class:`Draft`, which holds
the document as plain data and changes nothing on disk or in the engine
until :meth:`Draft.save`. :meth:`Draft.document` validates the whole draft
the way the server will read it, so a draft that saves is a draft that
loads. Nothing here imports Qt.

Grouping is one operation: :meth:`Draft.add_check` adds a check to an
existing interlock (joining its group) or starts a new one, and
:meth:`Draft.move_check` moves a check between interlocks. An interlock
whose last check leaves it is removed — an interlock always holds at
least one condition.
"""

from __future__ import annotations

import copy
import difflib
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from geecs_schemas import InterlockSet

from geecs_interlocks.catalog import named_in

#: A check as the draft holds it: one condition's fields, ``kind`` first.
Check = dict[str, Any]


def to_yaml(document: InterlockSet) -> str:
    """*document* as the file the server reads: defaults left out, ``schema_version`` first."""
    data = document.model_dump(mode="json", exclude_defaults=True)
    data.pop("schema_version", None)
    ordered = {"schema_version": document.schema_version, **data}
    return yaml.safe_dump(ordered, sort_keys=False, allow_unicode=True, width=100)


class Draft:
    """An editable copy of *document*.

    Parameters
    ----------
    document :
        The interlocks the draft starts from (and :meth:`revert` returns to).
    """

    def __init__(self, document: InterlockSet) -> None:
        self._saved = document
        self._data = self._plain(document)

    @staticmethod
    def _plain(document: InterlockSet) -> dict[str, Any]:
        return document.model_dump(mode="json", exclude_defaults=True)

    # -- reading --------------------------------------------------------------

    @property
    def interlocks(self) -> list[dict[str, Any]]:
        """The interlocks as plain data, in order (a copy: edit through the methods)."""
        return copy.deepcopy(self._data.get("interlocks", []))

    def names(self) -> list[str]:
        """Every interlock's name, in order."""
        return [i["name"] for i in self._data.get("interlocks", [])]

    def check(self, name: str, index: int) -> Check:
        """Check *index* of interlock *name* (a copy)."""
        return copy.deepcopy(self._entry(name)["conditions"][index])

    @property
    def dirty(self) -> bool:
        """Whether the draft differs from the document it started from."""
        return self._data != self._plain(self._saved)

    def document(self) -> InterlockSet:
        """The draft validated as the server reads it.

        Raises
        ------
        pydantic.ValidationError
            The draft is not a valid interlocks document.
        """
        return InterlockSet.model_validate(self._data)

    def watched(self) -> dict[str, dict[str, list[str]]]:
        """``device → variable → interlock names`` of every check (the panels' markers).

        A device watched only for staleness appears with no variables.
        """
        out: dict[str, dict[str, list[str]]] = {}
        for entry in self._data.get("interlocks", []):
            single = InterlockSet.model_validate(
                {"interlocks": [{**entry, "enabled": True}]}
            )
            for device, variables in named_in(single).items():
                per_device = out.setdefault(device, {})
                for variable in variables:
                    per_device.setdefault(variable, []).append(entry["name"])
        return out

    def ports(self) -> dict[int, list[str]]:
        """``UDP port → interlock names`` of every external check."""
        out: dict[int, list[str]] = {}
        for entry in self._data.get("interlocks", []):
            for c in entry["conditions"]:
                if c.get("kind") == "external":
                    out.setdefault(int(c["port"]), []).append(entry["name"])
        return out

    # -- editing --------------------------------------------------------------

    def _entries(self) -> list[dict[str, Any]]:
        return self._data.setdefault("interlocks", [])

    def _entry(self, name: str) -> dict[str, Any]:
        for entry in self._entries():
            if entry["name"] == name:
                return entry
        raise KeyError(f"no interlock named {name!r}")

    def add_check(
        self, check: Mapping[str, Any], to: str | None = None, *, new_name: str = ""
    ) -> tuple[str, int]:
        """Add *check* to interlock *to*, or (``to=None``) to a new one named *new_name*.

        Returns
        -------
        tuple
            ``(interlock name, check index)`` of the added check.

        Raises
        ------
        KeyError
            *to* names no interlock.
        ValueError
            A new interlock's name is empty or taken.
        """
        if to is None:
            name = new_name.strip()
            if not name:
                raise ValueError("a new interlock needs a name")
            if name in self.names():
                raise ValueError(f"an interlock named {name!r} exists already")
            self._entries().append({"name": name, "conditions": [dict(check)]})
            return name, 0
        conditions = self._entry(to)["conditions"]
        conditions.append(dict(check))
        return to, len(conditions) - 1

    def replace_check(self, name: str, index: int, check: Mapping[str, Any]) -> None:
        """Replace check *index* of *name* with *check*."""
        self._entry(name)["conditions"][index] = dict(check)

    def remove_check(self, name: str, index: int) -> None:
        """Remove check *index* of *name* (and *name* with it, if it was the last)."""
        entry = self._entry(name)
        del entry["conditions"][index]
        if not entry["conditions"]:
            self._entries().remove(entry)

    def move_check(self, name: str, index: int, to: str) -> tuple[str, int]:
        """Move check *index* of *name* into interlock *to*; its new place."""
        if name == to:
            return name, index
        target = self._entry(to)  # raises before anything moves
        check = self._entry(name)["conditions"][index]
        self.remove_check(name, index)
        target["conditions"].append(check)
        return to, len(target["conditions"]) - 1

    def ungroup(self, name: str) -> list[str]:
        """Split *name* into one interlock per check, in its place; the new names."""
        entry = self._entry(name)
        if len(entry["conditions"]) < 2:
            return [name]
        entries = self._entries()
        at = entries.index(entry)
        taken = set(self.names()) - {name}
        parts, new_names = [], []
        for n, check in enumerate(entry["conditions"], start=1):
            candidate = f"{name} {n}"
            while candidate in taken:
                candidate += "'"
            taken.add(candidate)
            new_names.append(candidate)
            part = {k: v for k, v in entry.items() if k != "conditions"}
            parts.append({**part, "name": candidate, "conditions": [check]})
        entries[at : at + 1] = parts
        return new_names

    def rename(self, name: str, new_name: str) -> None:
        """Rename interlock *name* (refusing an empty or taken name)."""
        new_name = new_name.strip()
        if not new_name:
            raise ValueError("an interlock needs a name")
        if new_name != name and new_name in self.names():
            raise ValueError(f"an interlock named {new_name!r} exists already")
        self._entry(name)["name"] = new_name

    def set_enabled(self, name: str, enabled: bool) -> None:
        """Arm (``True``) or disarm interlock *name*."""
        entry = self._entry(name)
        if enabled:
            entry.pop("enabled", None)
        else:
            entry["enabled"] = False

    def set_settle(self, name: str, seconds: float) -> None:
        """Set how long *name* must stay clear before it reports OK."""
        if seconds < 0:
            raise ValueError("the settle time cannot be negative")
        entry = self._entry(name)
        if seconds:
            entry["settle_s"] = seconds
        else:
            entry.pop("settle_s", None)

    def remove(self, name: str) -> None:
        """Delete interlock *name*."""
        self._entries().remove(self._entry(name))

    def move(self, name: str, offset: int) -> None:
        """Move interlock *name* up (negative) or down in the list."""
        entries = self._entries()
        at = entries.index(self._entry(name))
        to = max(0, min(len(entries) - 1, at + offset))
        entries.insert(to, entries.pop(at))

    def revert(self) -> None:
        """Throw away every edit since the last save."""
        self._data = self._plain(self._saved)

    # -- saving ---------------------------------------------------------------

    def diff(self, path: Path | None) -> str:
        """The unified diff from the file at *path* (empty: none) to the draft."""
        new = to_yaml(self.document())
        old = path.read_text(encoding="utf-8") if path and path.exists() else ""
        lines = difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=str(path or "(new)"),
            tofile="draft",
        )
        return "".join(lines)

    def save(self, path: Path) -> InterlockSet:
        """Validate the draft and write it to *path*, replacing the file at once.

        The file is written beside *path* and renamed over it, so a reader
        never sees half a file. The draft then counts as saved.

        Raises
        ------
        pydantic.ValidationError
            The draft is not valid; nothing is written.
        """
        document = self.document()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.stem}-", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(to_yaml(document))
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        self._saved = document
        self._data = self._plain(document)
        return document


__all__ = ["Check", "Draft", "to_yaml"]
