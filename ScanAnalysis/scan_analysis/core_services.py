"""Build the collaborators a measure names as its service, from this host's config.

A core measure may need something the core is not allowed to do itself: the
``frog`` measure runs Kane's FROG.dll, an external 32-bit program. The core
names the service; this module is the host side that builds it — from the
client ``config.ini``, a facility value — so the core never reads a path,
starts a process or learns whether the DLL runs natively or under Wine.

A service travels to every pool worker once (pickled with the prepared
inputs), so what is built here must pickle; ``FrogDllRetrieval`` is a few
paths and a command prefix.
"""

from __future__ import annotations

from typing import Callable

from geecs_analysis.registry import MeasureSpec, measure_definition

__all__ = ["SERVICE_FACTORIES", "services_for"]


def _frog_retriever() -> object:
    """The FROG.dll retrieval, configured by ``[Paths] frog_*`` in config.ini."""
    from image_analysis.algorithms.frog_dll_retrieval import FrogDllRetrieval

    return FrogDllRetrieval.from_config()


#: Service name (a measure's registration) → the factory that builds it here.
SERVICE_FACTORIES: dict[str, Callable[[], object]] = {"frog": _frog_retriever}


def services_for(measure: MeasureSpec) -> dict[str, object]:
    """The bound services ``measure`` needs, built from this host's config.

    Empty for a measure that names none. A service this host cannot build
    (no factory, or a factory whose configuration is missing) raises here,
    when the run is prepared, before any shot is read.
    """
    name = measure_definition(measure).service
    if name is None:
        return {}
    factory = SERVICE_FACTORIES.get(name)
    if factory is None:
        raise LookupError(
            f"The {measure.kind!r} measure needs service {name!r}, "
            "which this host does not provide"
        )
    return {name: factory()}
