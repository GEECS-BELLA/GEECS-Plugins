from pathlib import Path

# package-wide unit registry
from pint import UnitRegistry
ureg =  UnitRegistry()

GEECS_Plugins_folder = Path(__file__).parents[2]
