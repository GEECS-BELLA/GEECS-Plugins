"""
Deprecated Type Aliases for GEECS Scanner Data Acquisition.

This module provides backward compatibility for ScanConfig and ScanMode types
that have been moved to the geecs_data_utils package. It serves as a transitional
mechanism to help users update their import statements while maintaining
existing code functionality.

Key Features:
- Provides deprecation warnings for outdated import paths
- Maintains backward compatibility with existing code
- Redirects imports to the new location in geecs_data_utils

Design Principles:
- Minimize disruption during package refactoring
- Encourage gradual migration to new import paths
- Provide clear guidance for code updates

Typical Workflow:
1. Receive deprecation warning
2. Update import statements in dependent modules
3. Gradually phase out usage of this module

Dependencies:
- geecs_data_utils package
- warnings module for deprecation notifications

Example (Old Usage):
```python
from geecs_scanner.data_acquisition.types import ScanConfig, ScanMode
```

Example (New Usage):
```python
from geecs_data_utils.types import ScanConfig, ScanMode
```

Notes
-----
- This module will be removed in a future version
- Users should update their import statements
- The underlying types remain unchanged

See Also
--------
- geecs_data_utils.types : New location of ScanConfig and ScanMode
"""

import warnings
from geecs_data_utils import ScanConfig as _ScanConfig
from geecs_data_utils import ScanMode as _ScanMode

warnings.warn(
    "geecs_scanner.data_acquisition.types.ScanConfig/ScanMode have moved to "
    "geecs_data_utils.types; please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

# Backward compatibility: alias instead of subclass
ScanConfig = _ScanConfig
ScanMode = _ScanMode
