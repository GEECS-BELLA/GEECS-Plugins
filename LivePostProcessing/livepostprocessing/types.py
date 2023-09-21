from typing import NewType

from numpy.typing import NDArray
from pint import Quantity

QuantityArray = NewType("QuantityArray", Quantity)

Array2D = NewType("Array2D", NDArray)
QuantityArray2D = NewType("QuantityArray2D", Quantity)

# run ID is the date folder name in yy_mmdd format
RunID = NewType("RunID", str)
ScanNumber = NewType("ScanNumber", int)
ShotNumber = NewType("ShotNumber", int)

# a folder name such as DeviceName or DeviceName-Subject
ImageFolderName = NewType("ImageFolderName", str)
DeviceName = NewType("DeviceName", str)
MetricName = NewType("MetricName", str)

# Usually 'raw', but sometimes a camera can produce multiple images
ImageSubject = NewType("ImageSubject", str)

