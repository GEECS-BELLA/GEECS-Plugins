from scan_analysis.base import AnalyzerInfo as Info
from scan_analysis.analyzers.Thomson.camera_image_analysis import CameraImageAnalysis
from scan_analysis.analyzers.Thomson.htt_magspec_analysis import HTTMagSpecAnalysis
from scan_analysis.analyzers.Thomson.htt_c14_analysis import HttC14EbeamProfiler
thomson_analyzers = [
    # Info(analyzer_class=HTTMagSpecAnalysis,
    #      requirements={'AND':['HTT-C23_1_MagSpec1', 'HTT-C23_2_MagSpec2', 'HTT-C23_3_MagSpec3', 'HTT-C23_4_MagSpec4']}),
    Info(analyzer_class=HttC14EbeamProfiler,
         requirements={'HTT-C14_1_ebeamprofile'},
         device_name='HTT-C14_1_ebeamprofile'),
]
