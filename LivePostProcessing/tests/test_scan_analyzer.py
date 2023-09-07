from unittest import TestCase
from livepostprocessing.scan_analyzer import ScanAnalyzer

class TestScanWatch(TestCase):
    def test_init(self):
        scan_analyzer = ScanAnalyzer()
