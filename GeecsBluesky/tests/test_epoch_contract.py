"""The wire and file formats agree without linking their foundational packages."""

from geecs_core.db.variable_types import LABVIEW_EPOCH_OFFSET as WIRE_EPOCH
from geecs_data_utils.io.scan_stack import LABVIEW_EPOCH_OFFSET as FILE_EPOCH


def test_wire_and_file_epoch_offsets_agree():
    assert WIRE_EPOCH == FILE_EPOCH
