from dataclasses import dataclass
from typing import Optional, Union
from enum import Enum, auto

class ScanMode(Enum):
    STANDARD = auto()
    NOSCAN = auto()
    OPTIMIZATION = auto()

@dataclass
class ScanConfig:
    scan_mode: ScanMode = ScanMode.NOSCAN
    device_var: Optional[str] = None
    start: Union[int, float] = 0
    end: Union[int, float] = 1
    step: Union[int, float] = 1
    wait_time: float = 1.0
    additional_description: Optional[str] = None
    background: bool = False
