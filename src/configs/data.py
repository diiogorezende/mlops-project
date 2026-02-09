from dataclasses import dataclass
from omegaconf import MISSING

@dataclass
class DataModuleConfig:
    _target_: str = MISSING
    data_dir: str = "./data"
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = False