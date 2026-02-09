from dataclasses import dataclass
from typing import Any
from omegaconf import MISSING


@dataclass
class OptimizerConfig:
    _target_: str = MISSING
    _partial_: bool = True
    lr: float = 1e-3
    weight_decay: float = 0.0

@dataclass
class LossConfig:
    _target_: str = MISSING