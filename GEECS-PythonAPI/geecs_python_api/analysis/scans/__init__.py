import warnings

__all__ = ["ScanData"]

def __getattr__(name):
    if name == "ScanData":
        warnings.warn(
            "geecs_python_api.analysis.scans.scan_data.ScanData has moved to geecs_data_utils.scan_data.ScanData; "
            "please update your imports.",
            DeprecationWarning,
            stacklevel=2,
        )
        from geecs_data_utils import ScanData
        return ScanData
    raise AttributeError(f"module {__name__} has no attribute {name}")