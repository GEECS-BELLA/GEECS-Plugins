"""
Fri 7/28/2023

Made a function to parse through multiple inequalities with arbitrary components and return the indices that satisfy all
inequalities.  This script just tests that it is working as intended.

@Chris
"""
from __future__ import annotations
import unittest

import numpy as np
import image_analysis.analyzers.U_HiResMagCam.OnlineAnalysisModules.CedossMathTools as MathTools

class TestMultiInequality(unittest.TestCase):
    
    def test_multi_inequality(self):

        testarr1 = np.linspace(1, 100, 100)
        value1 = 50
        value2 = 70
        testarr2 = np.linspace(80, 40, 100)

        inputList = [[testarr1, '>', value1],
                    [testarr1, '<', value2],
                    [testarr2, '>=', testarr1]]  # [testarr2, '>', value2]]

        trial1 = MathTools.GetInequalityIndices(inputList)

        self.assertTrue(np.all(trial1 == np.arange(50, 57)))

if __name__ == '__main__':
    unittest.main()
