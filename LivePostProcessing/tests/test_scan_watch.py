from unittest import TestCase
from livepostprocessing.scan_watch import ScanWatch

class TestScanWatch(TestCase):
    def test_init(self):
        scan_watch = ScanWatch('23_0713')
