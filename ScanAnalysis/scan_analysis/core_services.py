"""Build the collaborators a measure names as its service, from this host's config.

A core measure may need something the core is not allowed to do itself: the
``frog`` measure runs Kane's FROG.dll, the ``haso`` measure Imagine Optic's
WaveKit — external programs. The core names the service; this module is the
host side that builds it — from the client ``config.ini``, a facility value,
and from the scan being analyzed — so the core never reads a path, starts a
process or learns whether the program runs natively or under Wine.

A factory takes the run's device data directory (``None`` for a context-free
preview): the WaveKit engine needs one ``.himg`` header of the sensor, which
it takes from that scan's capture stack. A service travels to every pool
worker once (pickled with the prepared inputs), so what is built here must
pickle; both services are a few paths, a command prefix and a header.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from geecs_analysis.registry import MeasureSpec, measure_definition
from geecs_data_utils.io.himg_stack import stack_header

from scan_analysis.core_source import stack_required

__all__ = ["SERVICE_FACTORIES", "services_for"]


def _frog_retriever(data_dir: Optional[Path]) -> object:
    """The FROG.dll retrieval, configured by ``[Paths] frog_*`` in config.ini."""
    from image_analysis.algorithms.frog_dll_retrieval import FrogDllRetrieval

    return FrogDllRetrieval.from_config()


def _haso_wavekit(data_dir: Optional[Path]) -> object:
    """The WaveKit engine (``[Paths] wavekit_*``) with the scan's sensor header."""
    from image_analysis.algorithms.haso_wavekit import HasoWaveKit

    if data_dir is None:
        raise LookupError(
            "the haso service needs the scan's capture stack for the sensor's "
            ".himg header (no scan folder given)"
        )
    return HasoWaveKit.from_config(stack_header(stack_required(data_dir)))


#: Service name (a measure's registration) → the factory that builds it here,
#: given the run's device data directory (``None`` without a scan).
SERVICE_FACTORIES: dict[str, Callable[[Optional[Path]], object]] = {
    "frog": _frog_retriever,
    "haso": _haso_wavekit,
}


def services_for(
    measure: MeasureSpec, *, data_dir: Optional[Path] = None
) -> dict[str, object]:
    """The bound services ``measure`` needs, built from this host's config.

    Empty for a measure that names none. A service this host cannot build
    (no factory, or a factory whose configuration is missing) raises here,
    when the run is prepared, before any shot is read. ``data_dir`` is the
    run's device data directory, for a service built from the scan itself.
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
    return {name: factory(data_dir)}
