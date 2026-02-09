from dataclasses import dataclass
from omegaconf import MISSING
from typing import Any



@dataclass
class TaskConfig:
    _target_: str = MISSING
    model: Any = MISSING
    optimizer: Any = MISSING
    loss_fn: Any = MISSING


@dataclass
class ProjectConfig:
    project_name: str = "FashionMNIST_Project"
    version: str = "1.0.0"
    debug: bool = True

    data: Any = MISSING
    task: Any = MISSING
    trainer: Any = MISSING