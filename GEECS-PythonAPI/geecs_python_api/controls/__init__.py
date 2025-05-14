import warnings

__all__ = ["ScanTag"]

def __getattr__(name):
    if name == "ScanTag":
        warnings.warn(
            "geecs_python_api.controls.api_defs.ScanTag has moved to "
            "geecs_data_utils.scan_paths; please update your imports.",
            DeprecationWarning,
            stacklevel=2,
        )
        from geecs_data_utils import ScanTag
        return ScanTag
    raise AttributeError(f"module {__name__} has no attribute {name}")