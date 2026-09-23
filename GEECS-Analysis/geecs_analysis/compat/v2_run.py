"""Streaming v2 execution over explicit shot groups and a caller-owned loader.

The host resolves files, grouping and output destinations. This runner only
loads the requested group and evaluates the compiled recipe. It writes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterable, Iterator, Mapping

from geecs_analysis.compat.v2 import V2Recipe, analyze_v2
from geecs_analysis.pipeline import bind_inputs

if TYPE_CHECKING:
    import numpy as np
    from geecs_data_utils.frames import Frame, ShotMeta
    from geecs_analysis.measurement import Measurement


@dataclass(frozen=True)
class ShotGroup:
    """One shot or bin, retaining all member rows for legacy scalar propagation."""

    key: int
    shots: tuple[int, ...]

    def __post_init__(self) -> None:
        """Own a nonempty, unique sequence of positive shot numbers."""
        shots = tuple(self.shots)
        if not shots or any(type(shot) is not int or shot < 1 for shot in shots):
            raise ValueError("A group requires positive integer shot numbers")
        if len(set(shots)) != len(shots):
            raise ValueError("A group cannot repeat a shot number")
        object.__setattr__(self, "shots", shots)


@dataclass(frozen=True)
class LoadFailure:
    """A source failed to load a member; the bin may still have usable members."""

    shot: int
    message: str


@dataclass(frozen=True)
class UnitResult:
    """One explicit outcome, including failed loads and analysis failure.

    ``group.shots`` is the legacy scalar-write membership. ``loaded_shots`` is
    the actual contribution set, which can be smaller after source failures.
    Neither set is silently inferred from the other. The caller decides where
    to log failures and how to persist/display successful measurements.
    """

    group: ShotGroup
    loaded_shots: tuple[int, ...]
    measurement: Measurement | None
    load_failures: tuple[LoadFailure, ...] = ()
    error: str | None = None


def run_units(
    recipe: V2Recipe,
    groups: Iterable[ShotGroup],
    load: Callable[[int], np.ndarray],
    *,
    average_before_analysis: bool = False,
    inputs: Mapping[str, Frame] | None = None,
    shot_metadata: Mapping[int, ShotMeta] | None = None,
) -> Iterator[UnitResult]:
    """Yield ordered outcomes, loading at most one group's raw arrays at a time.

    Per-shot mode requires single-member groups. Per-bin mode averages native
    arrays before v2 scaling/processing, preserving numpy's legacy dtype and
    mean (not nanmean) semantics. Bad loads are excluded; the original bin's
    full scalar-write membership survives. Incompatible raw shapes or analysis
    failures yield an explicit unsuccessful outcome and later groups continue.

    Required frame bindings are snapshotted before the first load. The loader
    belongs to the source host; no paths, config reads, writes or renderer
    state live here. Loading is sequential in declared member order so the
    floating-point reduction order is reproducible.
    """
    import numpy as np
    from geecs_data_utils.frames import ShotMeta

    bound = bind_inputs(recipe.analysis.steps, inputs)
    metadata = dict(shot_metadata or {})
    if any(
        not isinstance(identity, ShotMeta) or identity.shot_number != number
        for number, identity in metadata.items()
    ):
        raise ValueError("Shot metadata must match its shot-number key")
    for group in groups:
        if not average_before_analysis and len(group.shots) != 1:
            raise ValueError("Per-shot execution requires single-member groups")
        loaded = []
        arrays = []
        failures = []
        for shot in group.shots:
            try:
                data = load(shot)
                if not isinstance(data, np.ndarray):
                    raise TypeError("Source must return a native ndarray")
            except Exception as exc:
                failures.append(LoadFailure(shot, str(exc)))
            else:
                loaded.append(shot)
                # A streaming source may reuse its read buffer on the next
                # call. Retain this shot's native precision and values now.
                arrays.append(data.copy())
        measurement = None
        error = None
        if not arrays:
            error = "No loadable inputs in group"
        else:
            try:
                raw = np.mean(arrays, axis=0) if average_before_analysis else arrays[0]
                shot = None
                if not average_before_analysis:
                    number = group.shots[0]
                    shot = metadata.get(number, ShotMeta(recipe.device, number))
                measurement = analyze_v2(raw, recipe, shot=shot, inputs=bound)
            except Exception as exc:
                error = str(exc)
        # Release native arrays before suspension. Only the owned Measurement
        # and lightweight outcome survive while the sink handles this group.
        arrays.clear()
        data = raw = None
        yield UnitResult(group, tuple(loaded), measurement, tuple(failures), error)
