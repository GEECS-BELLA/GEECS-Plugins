import os
from pathlib import Path
from typing import Optional, Union
from geecs_api.interface import GeecsDatabase
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.diagnostics import EBeamDiagnostics
from htu_scripts.analysis.screens_scan_analysis import screens_scan_analysis
from geecs_api.tools.scans.scan_images import ScanImages
from geecs_api.tools.interfaces.prompts import text_input
from geecs_api.tools.images.filtering import FiltersParameters
from geecs_api.tools.scans.scan_data import ScanData


def emq_alignment(ebd: EBeamDiagnostics):
    return
