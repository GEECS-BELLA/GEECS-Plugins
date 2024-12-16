"""
Refurbished analysis for the Rad2 Spectrometer

Chris
"""
from __future__ import annotations

from typing import Optional
from scan_analysis.analyzers.Undulator.CameraImageAnalysis import CameraImageAnalysis
from geecs_python_api.controls.api_defs import ScanTag


class Rad2SpecAnalysis(CameraImageAnalysis):
    def __init__(self, scan_tag: ScanTag, skip_plt_show: bool = True):
        super().__init__(scan_tag=scan_tag, device_name='UC_UndulatorRad2', skip_plt_show=skip_plt_show)

    def run_analysis(self, config_options: Optional[str] = None):
        pass
