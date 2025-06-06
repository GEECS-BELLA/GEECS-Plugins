from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Dict, TYPE_CHECKING,Any
from enum import Enum

class ScanMode(str, Enum):
    STANDARD = 'standard'
    NOSCAN = 'noscan'
    OPTIMIZATION = 'optimization'
    BACKGROUND = 'background'

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
    optimizer_config_path: Optional[Union[str,Path]] = None
    optimizer_overrides: Optional[Dict[str, Any]] = field(default_factory=dict)
    evaluator_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)
