from scan_analysis.base import AnalyzerInfo as Info
from scan_analysis.analyzers.Thomson.PW_test_analysis import PWTestAnalysis

controlroom_analyzers = [
    Info(analyzer_class=PWTestAnalysis,
         requirements={'OR':['CAM-CR-Beampointing1', 'CAM-HPD-MultiPlane1']}),
]