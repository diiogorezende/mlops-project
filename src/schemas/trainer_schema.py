from pydantic import BaseModel, Field, ConfigDict

class TrainerConfig(BaseModel):
    target: str = Field(alias="_target_")
    max_epochs: int = 5
    accelerator: str = "auto" # Outras opcoes sao: "cpu", "gpu", "tpu"
    devices: int | str = "auto" # Pode ser um numero inteiro ou "auto"

    # Permitir campos extras do Lightning no modelo
    model_config = ConfigDict(extra="allow")