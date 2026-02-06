from pydantic import BaseModel, Field

class DataModuleConfig(BaseModel):
    target: str = Field(alias="_target_")
    data_dir: str = "./data"
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = True