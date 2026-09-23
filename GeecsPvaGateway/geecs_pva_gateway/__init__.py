"""Distributed pvAccess gateway serving GEECS camera images as NTNDArray PVs."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("geecs-pva-gateway")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0+source"

from geecs_pva_gateway.config import DeviceSpec, PvaGatewayConfig
from geecs_pva_gateway.server import GeecsPvaGateway

__all__ = ["DeviceSpec", "GeecsPvaGateway", "PvaGatewayConfig", "__version__"]
