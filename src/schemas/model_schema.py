from pydantic import BaseModel, Field
from typing import Optional

# Esquema do backbone
class BackboneConfig(BaseModel):
    # O _targe_ diz qual classe Python deve ser instanciada
    target: str = Field(alias="_target_")
    pretrained: bool = False
    in_channels: int = 1

# Esquema para o head
class HeadConfig(BaseModel):
    target: str = Field(alias="_target_")
    num_classes: int = 10
    in_features: int = 512

# Esquema do modelo completo
class ModelConfig(BaseModel):
    target: str = Field(alias="_target_")
    backbone: BackboneConfig
    head: HeadConfig