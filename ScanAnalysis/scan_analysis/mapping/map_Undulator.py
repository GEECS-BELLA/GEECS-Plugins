from scan_analysis.base import AnalyzerInfo as Info

from scan_analysis.analyzers.Undulator.MagSpecStitcherAnalysis import MagSpecStitcherAnalysis
from scan_analysis.analyzers.Undulator.CameraImageAnalysis import CameraImageAnalysis
from scan_analysis.analyzers.Undulator.VisaEBeamAnalysis import VisaEBeamAnalysis
from scan_analysis.analyzers.Undulator.Rad2SpecAnalysis import Rad2SpecAnalysis
from scan_analysis.analyzers.Undulator.HiResMagCamAnalysis import HiResMagCamAnalysis

undulator_analyzers = [
    Info(analyzer_class=MagSpecStitcherAnalysis,
         requirements={'AND': ['U_BCaveICT', 'U_BCaveMagSpec']},
         device_name='U_BCaveMagSpec'),
    Info(analyzer_class=VisaEBeamAnalysis,
         requirements={'OR': ['UC_VisaEBeam1', 'UC_VisaEBeam2', 'UC_VisaEBeam3', 'UC_VisaEBeam4',
                              'UC_VisaEBeam5', 'UC_VisaEBeam6', 'UC_VisaEBeam7', 'UC_VisaEBeam8']}),
    Info(analyzer_class=Rad2SpecAnalysis,
         requirements={'AND': ['U_BCaveICT', 'UC_UndulatorRad2',
                               {'OR': ['UC_VisaEBeam1', 'UC_VisaEBeam2', 'UC_VisaEBeam3', 'UC_VisaEBeam4',
                                       'UC_VisaEBeam5', 'UC_VisaEBeam6', 'UC_VisaEBeam7', 'UC_VisaEBeam8']}]}),
    Info(analyzer_class=CameraImageAnalysis,
         requirements={'UC_ALineEBeam3'},
         device_name='UC_ALineEBeam3'),
    Info(analyzer_class=HiResMagCamAnalysis,
         requirements={'AND': ['UC_HiResMagCam', 'U_BCaveICT']})
]