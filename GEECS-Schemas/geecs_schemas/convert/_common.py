"""Shared plumbing for legacy-YAML → schema converters.

Converters accept either an already-parsed ``dict`` or a filesystem path to a
YAML file.  Path input needs PyYAML, which is *not* a dependency of this
package — it is imported lazily so dict-based conversion stays dependency
free.  All converters fail loudly through :class:`SchemaConversionError`,
always naming the exact keys they could not map.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Union

LegacyDocument = Union[dict, "Path", str]


class SchemaConversionError(ValueError):
    """A legacy config contains something the converter cannot map.

    The message always names the offending file/entry and the exact keys or
    values that have no representation in the new schema, so nothing is ever
    dropped silently.
    """


def load_legacy(source: LegacyDocument) -> dict:
    """Return the legacy document as a dict, reading YAML if given a path.

    Parameters
    ----------
    source : dict or Path or str
        A parsed legacy document, or a path to its YAML file.

    Returns
    -------
    dict
        The parsed document (an empty file parses to ``{}``).

    Raises
    ------
    ImportError
        If a path is given but PyYAML is not installed.
    SchemaConversionError
        If the file does not parse to a mapping.
    """
    if isinstance(source, dict):
        return source
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Loading legacy configs from a path requires PyYAML "
            "(pip install pyyaml); alternatively pass an already-parsed dict."
        ) from exc
    path = Path(source)
    loaded = yaml.safe_load(path.read_text())
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise SchemaConversionError(
            f"{path}: expected a YAML mapping at the top level, got "
            f"{type(loaded).__name__}."
        )
    return loaded


def require_known_keys(document: dict, known: Iterable[str], context: str) -> None:
    """Fail loudly if *document* has keys the converter does not understand.

    Parameters
    ----------
    document : dict
        The legacy document or sub-document to check.
    known : iterable of str
        Every key the converter knows how to map (or deliberately drop).
    context : str
        Human-readable location for the error message.

    Raises
    ------
    SchemaConversionError
        Naming each unknown key.
    """
    unknown = sorted(set(document) - set(known))
    if unknown:
        raise SchemaConversionError(
            f"{context}: cannot map unknown key(s) {unknown} — the converter "
            f"understands {sorted(known)}."
        )


def source_name(source: LegacyDocument, fallback: str) -> str:
    """Derive a config name from a path's stem, or use *fallback* for dicts.

    Parameters
    ----------
    source : dict or Path or str
        The converter input.
    fallback : str
        Name to use when the input is a dict.

    Returns
    -------
    str
        A name suitable for the converted model's ``name`` field.
    """
    if isinstance(source, dict):
        return fallback
    return Path(source).stem


def as_wire_value(value: Any) -> str:
    """Render a legacy scalar exactly as it would travel the GEECS wire.

    Parameters
    ----------
    value : Any
        Legacy YAML scalar (str, int, float, bool).

    Returns
    -------
    str
        The verbatim string form.
    """
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(value)
