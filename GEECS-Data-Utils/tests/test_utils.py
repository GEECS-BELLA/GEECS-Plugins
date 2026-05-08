from pathlib import Path
from unittest import TestCase

from geecs_data_utils.utils import timestamp_from_filename, timestamp_from_string


class TimestampTestCase(TestCase):
    def test_timestamp_from_string(self):
        assert timestamp_from_string("_3827756096.897.anything") == 3827756096.897

    def test_timestamp_from_filename(self):
        file = Path(
            r"Z:\data\Undulator\Y2025\04-Apr\25_0417\scans\Scan013"
            r"\UC_TC_Phosphor\UC_TC_Phosphor_3827756096.897.png"
        )
        assert timestamp_from_filename(file) == 3827756096.897

    def test_timestamp_from_string_no_match(self):
        with self.assertRaises(ValueError):
            timestamp_from_string("no_timestamp_here.png")
