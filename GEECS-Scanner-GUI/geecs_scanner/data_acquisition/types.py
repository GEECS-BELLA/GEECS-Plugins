from dataclasses import dataclass
from typing import Optional, Union

@dataclass
class ScanConfig:
    device_var: str
    start: Union[int, float] = 0
    end: Union[int, float] = 1
    step: Union[int, float] = 1
    wait_time: float = 1.0
    additional_description: str = ''
