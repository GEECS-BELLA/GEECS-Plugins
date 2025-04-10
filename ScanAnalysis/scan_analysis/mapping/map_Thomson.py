from scan_analysis.base import AnalyzerInfo as Info

from scan_analysis.analyzers.Thomson.htt_magspec_analysis import HTTMagSpecAnalysis

thomson_analyzers = [
    Info(analyzer_class=HTTMagSpecAnalysis,
         requirements={'AND':['HTT-C23_1_MagSpec1', 'HTT-C23_2_MagSpec2', 'HTT-C23_3_MagSpec3', 'HTT-C23_4_MagSpec4']}),
]