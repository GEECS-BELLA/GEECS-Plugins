"""Pulsed-wire kicks of a magnet line from the drift plateaus of one trace.

A pulsed-wire bench records the wire deflection versus time; time maps to
position along the magnet line and the trace is the first field integral:
flat in the drift gaps, changing inside each element. The measure means
the trace inside each drift window (``plateau_i``) and reports each
element's kick as the plateau after it minus the plateau before it.
Scalar names are index-based (``plateau_0`` is the window before element
1, ``kick_1`` element 1's kick) so their meanings do not depend on the
element names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal, Mapping, Optional

from pydantic import Field, model_validator

from geecs_analysis.registry import MeasureSpec, SpecModel, measure

if TYPE_CHECKING:
    from geecs_data_utils.frames import Frame
    from geecs_analysis.measurement import Measurement


class PulsedWireWindow(SpecModel):
    """One drift-plateau region of the trace, in the trace's axis units."""

    start: float = Field(
        description="Start of the drift plateau, in the trace's axis units (inclusive)."
    )
    end: float = Field(
        description="End of the drift plateau, in the trace's axis units (inclusive)."
    )

    @model_validator(mode="after")
    def _ordered(self) -> PulsedWireWindow:
        if not self.start < self.end:
            raise ValueError("A pulsed-wire window needs start < end")
        return self


class PulsedWireElement(SpecModel):
    """One magnet of the line, between two drift windows."""

    name: str = Field(
        min_length=1,
        description="Label for the element (unique); scalars are named by index, not by this.",
    )
    length: Optional[float] = Field(
        None,
        gt=0,
        description=(
            "Element length in any unit; set, the measure also emits "
            "kick_per_length_i. Unset emits no per-length scalar."
        ),
    )


#: The default (example) line: a triplet; the numbers are placeholders.
_DEFAULT_WINDOWS = (
    PulsedWireWindow(start=0.0, end=0.4e-3),
    PulsedWireWindow(start=1.0e-3, end=1.4e-3),
    PulsedWireWindow(start=2.0e-3, end=2.4e-3),
    PulsedWireWindow(start=3.0e-3, end=3.4e-3),
)
_DEFAULT_ELEMENTS = (
    PulsedWireElement(name="Q1", length=1.0),
    PulsedWireElement(name="Q2", length=1.0),
    PulsedWireElement(name="Q3", length=1.0),
)


def _scalar_names(n_windows: int, lengths: tuple[Optional[float], ...]) -> list[str]:
    """Every scalar a line of ``n_windows`` windows and these lengths emits."""
    names = [f"plateau_{i}" for i in range(n_windows)]
    names += [f"kick_{i}" for i in range(1, n_windows)]
    names += [
        f"kick_per_length_{i}"
        for i, length in enumerate(lengths, start=1)
        if length is not None
    ]
    return names


def _default_docs() -> dict[str, str]:
    """Meanings of the default triplet's scalars (the documented index range)."""
    docs = {}
    last = len(_DEFAULT_WINDOWS) - 1
    for i in range(len(_DEFAULT_WINDOWS)):
        where = (
            "before element 1"
            if i == 0
            else f"after element {i}"
            if i == last
            else f"between elements {i} and {i + 1}"
        )
        docs[f"plateau_{i}"] = (
            f"Mean trace value inside drift window {i} ({where}), in trace units"
        )
    for i in range(1, len(_DEFAULT_WINDOWS)):
        docs[f"kick_{i}"] = (
            f"Kick of element {i}: plateau_{i} minus plateau_{i - 1}, in trace units"
        )
        docs[f"kick_per_length_{i}"] = (
            f"kick_{i} divided by element {i}'s length, in trace units per length unit"
        )
    return docs


class PulsedWireSpec(MeasureSpec):
    """Drift windows and the elements between them.

    ``scalar_docs`` documents the default triplet's index range; a longer
    line emits the same families (``plateau_i``, ``kick_i``,
    ``kick_per_length_i``) at higher indices with the same meanings.
    """

    kind: Literal["pulsed_wire"] = "pulsed_wire"
    scalar_docs: ClassVar[Mapping[str, str]] = _default_docs()
    windows: tuple[PulsedWireWindow, ...] = Field(
        _DEFAULT_WINDOWS,
        description=(
            "Drift-plateau regions in the trace's axis units (seconds for a "
            "scope time axis), ordered and non-overlapping, at least two: "
            "window 0 before element 1, window i between elements i and i+1, "
            "the last after the last element. The defaults are placeholders "
            "for a triplet; set them from the trace."
        ),
    )
    elements: tuple[PulsedWireElement, ...] = Field(
        _DEFAULT_ELEMENTS,
        description=(
            "The magnets in beam order, one fewer than the windows, names "
            "unique. The default triplet and its unit lengths are placeholders."
        ),
    )

    @model_validator(mode="after")
    def _consistent(self) -> PulsedWireSpec:
        windows = self.windows
        if len(windows) < 2:
            raise ValueError("A pulsed-wire measure needs at least two windows")
        for i, (before, after) in enumerate(zip(windows[:-1], windows[1:])):
            if not before.end < after.start:
                raise ValueError(
                    f"Pulsed-wire windows {i} and {i + 1} overlap or are out of order"
                )
        if len(self.elements) != len(windows) - 1:
            raise ValueError(
                f"{len(windows)} windows bound {len(windows) - 1} elements, "
                f"not {len(self.elements)}"
            )
        names = [element.name for element in self.elements]
        if len(set(names)) != len(names):
            raise ValueError("Pulsed-wire element names must be unique")
        return self

    def emitted_scalars(self) -> frozenset[str]:
        """Plateaus per window, kicks per element, per-length kicks per length."""
        return frozenset(
            _scalar_names(len(self.windows), tuple(e.length for e in self.elements))
        )


@measure(PulsedWireSpec, ndim={1})
def pulsed_wire(frame: Frame, spec: PulsedWireSpec) -> Measurement:
    """Mean each drift window and difference neighbours into element kicks.

    A window with no samples, or only nonfinite ones, gives a NaN plateau
    (and NaN kicks on both sides of it) with a note naming the window; the
    processed frame is returned unchanged.
    """
    from geecs_analysis.algorithms.pulsed_wire import kicks, plateau_means
    from geecs_analysis.measurement import Measurement

    if frame.data.ndim != 1:
        raise ValueError("Pulsed-wire measurement requires a trace")
    plateaus, notes = plateau_means(
        frame.axes[0].values,
        frame.data,
        [(w.start, w.end) for w in spec.windows],
    )
    scalars = {f"plateau_{i}": value for i, value in enumerate(plateaus)}
    for i, (kick, element) in enumerate(zip(kicks(plateaus), spec.elements), start=1):
        scalars[f"kick_{i}"] = kick
        if element.length is not None:
            scalars[f"kick_per_length_{i}"] = kick / element.length
    return Measurement(scalars=scalars, frame=frame, notes=tuple(notes))
