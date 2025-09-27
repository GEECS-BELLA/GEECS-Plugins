"""Type definitions and TypedDicts for the ImageAnalysis package.

Defines NewType aliases for NumPy arrays and Pint quantities used throughout the
codebase, as well as the :class:`AnalyzerResultDict` TypedDict describing the
structure of results returned by analyzers.
"""

from typing import NewType, TYPE_CHECKING, Any, Union, Optional

# exception to handle python 3.7
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

# Import numpy types for runtime use
try:
    from numpy.typing import NDArray
except ImportError:
    # Fallback for older numpy versions
    import numpy as np

    NDArray = np.ndarray

if TYPE_CHECKING:
    from pint import Quantity

    Array2D = NewType("Array2D", NDArray)

    QuantityArray = NewType("QuantityArray", Quantity)
    QuantityArray2D = NewType("QuantityArray2D", Quantity)
else:
    # Runtime definitions for when TYPE_CHECKING is False
    Array2D = NDArray
    QuantityArray = object
    QuantityArray2D = object


class AnalyzerResultDict(TypedDict):
    """TypedDict describing analyzer result dictionary."""

    processed_image: Optional[NDArray]
    analyzer_return_dictionary: Optional[dict[str, Union[int, float]]]
    analyzer_return_lineouts: Optional[NDArray]
    analyzer_input_parameters: Optional[dict[str, Any]]
