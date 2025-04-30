# geecs_python_api/analysis/scans/scan_data.py
import warnings
from geecs_scan_data_utils.scan_data import ScanData as _ScanData

warnings.warn(
    "geecs_python_api.analysis.scans.scan_data.ScanData has moved to "
    "geecs_scan_data_utils.scan_data.ScanData; please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

class ScanData(_ScanData):
    """Stub for backward compatibility."""
    pass