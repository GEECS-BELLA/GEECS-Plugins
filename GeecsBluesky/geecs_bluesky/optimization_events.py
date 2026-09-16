"""Import-light optimization stream column names, shared by worker and clients."""

from urllib.parse import quote, unquote


def optimization_column(prefix: str, name: str) -> str:
    """Encode a name for event-model keys and Tiled SQL column identifiers.

    Use URI escaping with ``~`` in place of ``%`` (forbidden by Tiled).
    Escape URI's always-safe dot, hyphen and tilde too: event-model forbids
    dots, SQL forbids double hyphens, and tilde is our escape marker.
    """
    encoded = quote(name, safe=":_")
    for literal, escape in (("~", "%7E"), (".", "%2E"), ("-", "%2D")):
        encoded = encoded.replace(literal, escape)
    return f"{prefix}:{encoded.replace('%', '~')}"


def optimization_name(encoded: str) -> str:
    """Decode the name component of an optimization stream column."""
    return unquote(encoded.replace("~", "%"))
