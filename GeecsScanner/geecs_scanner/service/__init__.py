"""The scanner's service layer: pure Python over the queue client.

Nothing in this package imports FastAPI.  :class:`ScannerService` takes the
client, the resolver and the stream cache it is given, and every method
returns a Pydantic model or raises :class:`ScannerError`.  The web layer
is three lines per route over it; tests drive it directly.
"""

__all__ = ["ProgressCache", "ScannerError", "ScannerService"]


def __getattr__(name: str):
    """Keep the disposable trajectory process independent of the queue client."""
    from importlib import import_module

    modules = {
        "ProgressCache": "streams",
        "ScannerError": "errors",
        "ScannerService": "scanner",
    }
    if name in modules:
        return getattr(import_module(f"{__name__}.{modules[name]}"), name)
    raise AttributeError(name)
