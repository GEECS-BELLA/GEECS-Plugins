"""Object-API matplotlib figure → PNG bytes.

A device's shot — pixels or a trace, the stack / native / vendor tier
ladder, the event-row joins and the display PNG — lives in GEECS-Data-Utils
(`geecs_data_utils.io.shot_images`, `geecs_data_utils.io.display_png`),
shared with every other viewer.
"""

from __future__ import annotations

import io


def figure_png(fig, *, tight: bool = False) -> bytes:
    """Object-API figure → PNG bytes (never pyplot; safe on the threadpool).

    ``tight`` crops to the drawn content (``bbox_inches="tight"``), the way
    the analysis sink saves a run's figures; a preview of a document uses it
    so the image matches the product file the run would write.
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", **({"bbox_inches": "tight"} if tight else {}))
    return buffer.getvalue()
