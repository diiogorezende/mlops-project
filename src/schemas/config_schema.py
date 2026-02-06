from pydantic import BaseModel, ConfigDict
from src.schemas.task_schema import TaskConfig
from src.schemas.datamodule_schema import DataModuleConfig
from src.schemas.trainer_schema import TrainerConfig

class ConfigSchema(BaseModel):
    """
    Define a estrutura esperada para o arquivo config.yaml
    Se houver algum campo faltante ou incorreto, o Pydantic gera um erro.
    """

    project_name: str
    version: str
    debug: bool = False

    task: TaskConfig
    data: DataModuleConfig
    trainer: TrainerConfig

    model_config = ConfigDict(arbitrary_types_allowed=True)