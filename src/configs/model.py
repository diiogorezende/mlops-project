from dataclasses import dataclass
from typing import Any
from omegaconf import MISSING

@dataclass
class BackboneConfig:
    _target_: str = MISSING
    pretrained: bool = True
    in_channels: int = 1

@dataclass
class HeadConfig:
    _target_: str = MISSING
    num_classes: int = 10
    in_features: int = 512

@dataclass
class ModelConfig:
    _target_: str = MISSING
    backbone: BackboneConfig = MISSING
    head: HeadConfig = MISSING