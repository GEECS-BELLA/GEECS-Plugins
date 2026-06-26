"""Low-level readers and decoders for native GEECS data.

This subpackage owns generic ``path -> numpy.ndarray`` file readers that are not
tied to any analysis logic. They provide a shared foundation for ImageAnalysis,
post-run analysis tools, and Bluesky external-asset handlers, none of which
should depend on the higher-level ``image_analysis`` package just to load a
file from disk. It also provides :func:`decode_imaq_image_string`, which decodes
an in-memory NI IMAQ "Flatten Image to String" payload received live over the
device TCP stream (not a file).
"""

from geecs_data_utils.io.images import (
    decode_imaq_image_string,
    load_image_from_h5,
    read_imaq_image,
    read_imaq_png_image,
    read_tsv_file,
)

__all__ = [
    "decode_imaq_image_string",
    "load_image_from_h5",
    "read_imaq_image",
    "read_imaq_png_image",
    "read_tsv_file",
]
