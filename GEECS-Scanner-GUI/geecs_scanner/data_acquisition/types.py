import warnings
from geecs_data_utils import ScanConfig as _ScanConfig
from geecs_data_utils import ScanMode as _ScanMode

warnings.warn(
    "geecs_scanner.data_acquisition.types.ScanConfg/ScanMode have moved to "
    "geecs_data_utils.types; please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

class ScanConfig(_ScanConfig):
    """Stub for backward compatibility."""
    pass

class ScanMode(_ScanMode):
    """Stub for backward compatibility."""
    pass
