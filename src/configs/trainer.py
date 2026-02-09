from dataclasses import dataclass
from typing import Any
from omegaconf import MISSING

@dataclass
class TrainerConfig:
    _target_: str = "pytorch_lightning.Trainer"
    max_epochs: int = 3
    accelerator: str = "auto"
    devices: int | str = "auto"
    log_every_n_steps: int = 1
    enable_checkpointing: bool = True