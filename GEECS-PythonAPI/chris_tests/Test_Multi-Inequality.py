"""
Fri 7/28/2023

Made a function to parse through multiple inequalities with arbitrary components and return the indices that satisfy all
inequalities.  This script just tests that it is working as intended.

@Chris
"""

import sys
import numpy as np

sys.path.insert(0, "../")
import online_analysis.HTU.OnlineAnalysisModules.CedossMathTools as MathTools


testarr1 = np.linspace(1, 100, 100)
value1 = 50
value2 = 70
testarr2 = np.linspace(80, 40, 100)

inputList = [[testarr1, '>', value1],
             [testarr1, '<', value2],
             [testarr2, '>=', testarr1]]  # [testarr2, '>', value2]]

trial1 = MathTools.GetInequalityIndices(inputList)

print(trial1)
