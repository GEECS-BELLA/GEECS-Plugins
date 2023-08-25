from __future__ import annotations

import unittest
from ..phasicsdensity.phasics_density_analysis import PhasicsImageAnalyzer

class PhasicsDensityAnalysisTestCase(unittest.TestCase):
    def test_init(self):
        pia = PhasicsImageAnalyzer()

if __name__ == "__main__":
    unittest.main()
