from scan_analysis.base import ScanAnalyzerInfo as Info

from scan_analysis.analyzers.Thomson.htt_magspec_analysis import HTTMagSpecAnalyzer

thomson_analyzers = [
    Info(analyzer_class=HTTMagSpecAnalyzer,
         requirements={'AND':['HTT-C23_1_MagSpec1', 'HTT-C23_2_MagSpec2', 'HTT-C23_3_MagSpec3', 'HTT-C23_4_MagSpec4']}),
]
