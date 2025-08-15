"""Type definitions and TypedDicts for the ImageAnalysis package.

Defines NewType aliases for NumPy arrays and Pint quantities used throughout the
codebase, as well as the :class:`AnalyzerResultDict` TypedDict describing the
structure of results returned by analyzers.
"""

from typing import NewType, TYPE_CHECKING

# exception to handle python 3.7
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

if TYPE_CHECKING:
    from typing import Any, Union, Optional

    from numpy.typing import NDArray
    from pint import Quantity

    Array2D = NewType("Array2D", NDArray)

    QuantityArray = NewType("QuantityArray", Quantity)
    QuantityArray2D = NewType("QuantityArray2D", Quantity)


class AnalyzerResultDict(TypedDict):
    """TypedDict describing analyzer result dictionary."""

    processed_image: Optional[NDArray]
    analyzer_return_dictionary: Optional[dict[str, Union[int, float]]]
    analyzer_return_lineouts: Optional[NDArray]
    analyzer_input_parameters: Optional[dict[str, Any]]
