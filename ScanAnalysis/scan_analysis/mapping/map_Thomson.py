from scan_analysis.base import AnalyzerInfo as Info
from scan_analysis.analyzers.common.array2D_scan_analysis import Array2DScanAnalysis
from image_analysis.offline_analyzers.Thomson.HTTC14Analysis.HTT_C14_Analysis import HTTC14
thomson_analyzers = [
    # Info(analyzer_class=HTTMagSpecAnalysis,
    #      requirements={'AND':['HTT-C23_1_MagSpec1', 'HTT-C23_2_MagSpec2', 'HTT-C23_3_MagSpec3', 'HTT-C23_4_MagSpec4']}),
    Info(analyzer_class=Array2DScanAnalysis,
         requirements={'HTT-C14_1_ebeamprofile'},
         device_name='HTT-C14_1_ebeamprofile',
         image_analyzer_class=HTTC14,
         )
]
