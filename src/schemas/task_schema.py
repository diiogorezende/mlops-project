from pydantic import BaseModel, Field, ConfigDict
from typing import Any
from src.schemas.model_schema import ModelConfig

class OptimizerConfig(BaseModel):
    target: str = Field(alias="_target_")
    # _partial_ permite informar parametros parciais sem passar o modelo completo no yaml
    partial: bool = Field(default=True, alias="_partial_")
    lr: float = 1e-3
    # Outros parametros do otimizador podem ser adicionados aqui
    model_config = ConfigDict(extra="allow")

class LossConfig(BaseModel):
    target: str = Field(alias="_target_")

class TaskConfig(BaseModel):
    target: str = Field(alias="_target_")
    # A task precisa conhecer o modelo, otimizador e loss
    model: ModelConfig
    optimizer: OptimizerConfig
    loss_fn: LossConfig