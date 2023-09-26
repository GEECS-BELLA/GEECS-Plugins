# package-wide unit registry
from pint import UnitRegistry
ureg = UnitRegistry()
Quantity = Q_ = ureg.Quantity
