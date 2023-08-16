from typing import NewType
from numpy.typing import NDArray
from pint import Quantity

Array2D = NewType("Array2D", NDArray)

QuantityArray = NewType("QuantityArray", Quantity)
QuantityArray2D = NewType("QuantityArray2D", Quantity)
