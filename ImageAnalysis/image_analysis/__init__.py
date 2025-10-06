"""Top-level package for ImageAnalysis.

Provides a package‑wide Pint unit registry and common quantity alias used
throughout the ImageAnalysis codebase. The module initializes a ``UnitRegistry``
instance and exposes it as ``ureg`` as well as a ``Quantity`` alias for easy
import by downstream modules.
"""

# package-wide unit registry
import pint
import logging

ureg = pint.UnitRegistry()
pint.set_application_registry(ureg)
Quantity = Q_ = ureg.Quantity

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
