"""Low-level readers and decoders for native GEECS data.

This subpackage owns generic ``path -> numpy.ndarray`` file readers that are not
tied to any analysis logic. They provide a shared foundation for ImageAnalysis,
post-run analysis tools and the web surfaces, none of which
should depend on the higher-level ``image_analysis`` package just to load a
file from disk. It also provides :func:`decode_imaq_image_string`, which decodes
an in-memory NI IMAQ "Flatten Image to String" payload received live over the
device TCP stream (not a file), and — in :mod:`geecs_data_utils.io.arrays` — the
decoders for the three array payload shapes devices push the same way.
"""

from geecs_data_utils.io.array1d import (
    Data1DConfig,
    Data1DResult,
    Data1DType,
    read_1d_data,
)
from geecs_data_utils.io.arrays import (
    WAVEFORM_ATTRIBUTE_KEYS,
    WAVEFORM_ATTRIBUTE_SUFFIXES,
    WAVEFORM_AXIS_KEYS,
    DecodedArray,
    decode_array_payload,
    decode_csv_values,
    decode_labview_waveform,
    decode_nested_pairs,
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
    frame_index_for_acq_timestamp,
    is_stack_file,
    read_shot,
    parse_attribute_name,
    read_stack_attributes,
    read_stack_timestamps,
    stack_content_kind,
    stack_scalar_variables,
)

__all__ = [
    "Data1DConfig",
    "Data1DResult",
    "Data1DType",
    "DecodedArray",
    "LABVIEW_EPOCH_OFFSET",
    "WAVEFORM_ATTRIBUTE_KEYS",
    "WAVEFORM_ATTRIBUTE_SUFFIXES",
    "WAVEFORM_AXIS_KEYS",
    "ShotRef",
    "average_frames",
    "decode_array_payload",
    "decode_csv_values",
    "decode_imaq_image_string",
    "decode_labview_waveform",
    "decode_nested_pairs",
    "find_stack_file",
    "frame_index_for_acq_timestamp",
    "is_stack_file",
    "load_image_from_h5",
    "read_1d_data",
    "read_imaq_image",
    "read_imaq_png_image",
    "read_shot",
    "parse_attribute_name",
    "read_stack_attributes",
    "read_stack_timestamps",
    "stack_content_kind",
    "stack_scalar_variables",
    "read_tsv_file",
]
