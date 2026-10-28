"""The portal's resources beyond a shot's pixels: traces and figure PNGs.

A device's shot as pixels — the stack / native / vendor tier ladder, the
event-row joins and the display PNG — lives in GEECS-Data-Utils
(`geecs_data_utils.io.shot_images`, `geecs_data_utils.io.display_png`),
shared with every other viewer.  What stays here is the portal's own:

Not every stack is pixels.  Since GeecsPvaGateway 0.13 the same file
plugin also captures a device's ARRAY variables — scope traces and
spectra — and those are x-vs-y data:
:func:`stack_content` names which a device's stack holds and
:func:`load_shot_trace` reads one shot of an array stack, so a caller
never renders a ``(16384, 2)`` lineout as a two-pixel-wide image.
:func:`figure_png` turns an object-API matplotlib figure into PNG bytes.

Strictly read-only: nothing on the scans path is ever created (repo
scan-folder invariant).
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from geecs_data_utils.io.array1d import (
    Data1DConfig,
    Data1DResult,
    Data1DType,
    read_1d_data,
)
from geecs_data_utils.io.scan_stack import (
    ShotRef,
    frame_index_for_acq_timestamp,
    stack_content_kind,
)
from geecs_data_utils.io.shot_images import DeviceKind, device_kind

logger = logging.getLogger(__name__)


def figure_png(fig, *, tight: bool = False) -> bytes:
    """Object-API figure → PNG bytes (never pyplot; safe on the threadpool).

    ``tight`` crops to the drawn content (``bbox_inches="tight"``), the way
    the analysis sink saves a run's figures; a preview of a document uses it
    so the image matches the product file the run would write.
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", **({"bbox_inches": "tight"} if tight else {}))
    return buffer.getvalue()


@dataclass(frozen=True)
class ShotTrace:
    """One resolved shot of an ARRAY stack — x-vs-y, or a refusal.

    The ``ShotImage`` of the trace path.  ``result`` carries the
    values the way every other 1D consumer in the repo receives them
    (:class:`~geecs_data_utils.io.array1d.Data1DResult`: an Nx2 array
    plus whatever the file could say about its axes), already trimmed to
    the trace's true length.
    """

    kind: str  # "stack" | "missing"
    result: Optional[Data1DResult] = None
    #: ``"lineout"`` or ``"waveform"`` — see ``stack_content_kind``.
    content: str = ""
    path: Optional[Path] = None
    reason: str = ""


#: The one reader config the portal uses for capture stacks: the stack
#: itself says whether a frame is a lineout or a waveform, and the axis
#: columns are the reader's defaults.
_TRACE_CONFIG = Data1DConfig(data_type=Data1DType.PVA_STACK)


def stack_content(probe: DeviceKind) -> str:
    """What a probed device's shots hold: ``"image"``, ``"lineout"`` or ``"waveform"``.

    Takes the :func:`device_kind` probe the caller already has, so the
    gallery costs one directory scan as before.  Anything that is not a
    readable array stack reads as ``"image"`` — the answer that keeps a
    caller on the path it has always taken, so a damaged or unfamiliar
    stack degrades to the existing rendering rather than to nothing.
    """
    if probe.kind != "stack" or probe.path is None:
        return "image"
    try:
        return stack_content_kind(probe.path)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        logger.warning("cannot classify stack %s: %s", probe.path, exc)
        return "image"


def load_shot_trace(
    scan_folder: Path,
    device: str,
    shot: int,
    acq_timestamp: Optional[float] = None,
    devices: Optional[list[str]] = None,
) -> ShotTrace:
    """Resolve one device shot of an ARRAY capture stack as x-vs-y.

    The trace twin of :func:`load_shot_array`, and deliberately thinner:
    array stacks are small, so there is no frame cache and no native-file
    tier — a device whose shots are traces has no per-shot file fallback.

    The shot→frame join is the same one the image path uses — its
    index-only form
    (:func:`~geecs_data_utils.io.scan_stack.frame_index_for_acq_timestamp`,
    the same ``_joined_index`` arithmetic, never a second copy), because
    the reader addresses the frame by ``ShotRef`` and reading it here
    would read it twice.  Without a timestamp it falls back to
    ``shot - 1`` exactly as the image path does.

    Parameters
    ----------
    scan_folder : Path
        The run's existing ``scans/ScanNNN`` folder.
    device : str
        Device subfolder name; validated against the folder listing.
    shot : int
        1-based shot number.
    acq_timestamp : float, optional
        The event row's device ``acq_timestamp`` (LabVIEW-epoch double).
    devices : list of str, optional
        An already-computed :func:`image_devices` listing.

    Returns
    -------
    ShotTrace
        The trimmed trace, or the refusal and its reason.
    """
    probe = device_kind(scan_folder, device, devices=devices)
    content = stack_content(probe)
    if content == "image":
        return ShotTrace(
            kind="missing",
            path=probe.path,
            reason=probe.reason or "device has no array capture stack",
        )
    if shot < 1:
        return ShotTrace(kind="missing", path=probe.path, reason="bad shot")
    stack = probe.path
    try:
        if acq_timestamp is not None:
            index = frame_index_for_acq_timestamp(stack, acq_timestamp)
            if index is None:
                return ShotTrace(
                    kind="missing",
                    path=stack,
                    content=content,
                    reason="no stack frame for this shot",
                )
        else:
            index = shot - 1
        result = read_1d_data(ShotRef(stack, index), _TRACE_CONFIG)
    # Same enumeration the image path degrades on: a malformed stack must
    # 404, never 500.
    except (IndexError, KeyError, OSError, TypeError, ValueError) as exc:
        return ShotTrace(
            kind="missing", path=stack, content=content, reason=f"stack: {exc}"
        )
    return ShotTrace(kind="stack", result=result, content=content, path=stack)
