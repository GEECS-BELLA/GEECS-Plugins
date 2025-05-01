from unittest import TestCase

from pathlib import Path
from geecs_python_api.tools.files.timestamping import timestamp_from_filename, timestamp_from_string


class TDMSTestCase(TestCase):
    def test_extract_timestamp_from_filename(self):
        assert timestamp_from_string("_3827756096.897.anything") == 3827756096.897

        file = Path("Z:\\data\\Undulator\\Y2025\\04-Apr\\25_0417\\scans\\Scan013\\UC_TC_Phosphor\\UC_TC_Phosphor_3827756096.897.png")
        assert timestamp_from_filename(file) == 3827756096.897
        