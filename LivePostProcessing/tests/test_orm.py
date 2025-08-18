import unittest
from pathlib import Path
from livepostprocessing.orm import Scan

class TestORM(unittest.TestCase):
    def test_init(self):
        scan = Scan("23_0518", 27, experiment_data_folder=Path(__file__).parent / 'data' / 'experiment')
        self.assertEqual(len(scan.scalar_data), 260)

if __name__ == "__main__":
    unittest.main()
