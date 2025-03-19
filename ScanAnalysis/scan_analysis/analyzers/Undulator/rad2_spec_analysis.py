"""
Refurbished analysis for the Rad2 Spectrometer

Chris
"""
from __future__ import annotations

from typing import Optional
from scan_analysis.analyzers.Undulator.camera_image_analysis import CameraImageAnalysis
from geecs_python_api.controls.api_defs import ScanTag


class Rad2SpecAnalysis(CameraImageAnalysis):
    def __init__(self, scan_tag: ScanTag, device_name: Optional[str] = None, skip_plt_show: bool = True):
        super().__init__(scan_tag=scan_tag, device_name='UC_UndulatorRad2', skip_plt_show=skip_plt_show)

    # def run_analysis(self, config_options: Optional[str] = None):
        # pass


if __name__ == "__main__":
    from geecs_python_api.analysis.scans.scan_data import ScanData

    tag = ScanData.get_scan_tag(year=2025, month=2, day=13, number=22, experiment_name='Undulator')
    analyzer = Rad2SpecAnalysis(scan_tag=tag, device_name='UC_UndulatorRad2', skip_plt_show=True)
    analyzer.run_analysis()
