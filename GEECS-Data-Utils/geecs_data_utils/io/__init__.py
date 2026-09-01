"""Low-level readers and decoders for native GEECS data.

This subpackage owns generic ``path -> numpy.ndarray`` file readers that are not
tied to any analysis logic. They provide a shared foundation for ImageAnalysis,
post-run analysis tools, and Bluesky external-asset handlers, none of which
should depend on the higher-level ``image_analysis`` package just to load a
file from disk. It also provides :func:`decode_imaq_image_string`, which decodes
an in-memory NI IMAQ "Flatten Image to String" payload received live over the
device TCP stream (not a file).
"""

from geecs_data_utils.io.array1d import (
    Data1DConfig,
    Data1DResult,
    Data1DType,
    read_1d_data,
)
from geecs_data_utils.io.images import (
    average_frames,
    decode_imaq_image_string,
    load_image_from_h5,
    read_imaq_image,
    read_imaq_png_image,
    read_tsv_file,
)
from geecs_data_utils.io.scan_stack import (
    LABVIEW_EPOCH_OFFSET,
    ShotRef,
    find_stack_file,
    is_stack_file,
    read_shot,
    read_stack_timestamps,
)

__all__ = [
    "Data1DConfig",
    "Data1DResult",
    "Data1DType",
    "LABVIEW_EPOCH_OFFSET",
    "ShotRef",
    "average_frames",
    "decode_imaq_image_string",
    "find_stack_file",
    "is_stack_file",
    "load_image_from_h5",
    "read_1d_data",
    "read_imaq_image",
    "read_imaq_png_image",
    "read_shot",
    "read_stack_timestamps",
    "read_tsv_file",
]
