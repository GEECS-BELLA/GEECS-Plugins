# package-wide unit registry
import pint
ureg = pint.UnitRegistry()
pint.set_application_registry(ureg)
Quantity = Q_ = ureg.Quantity
