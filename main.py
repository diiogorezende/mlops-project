import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from src.schemas.config_schema import ConfigSchema

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Funcao principal que busca os arquivos YAML, combina esses arquivos
    em um objeto DictConfig e injeta esse objeto na funcao principal.

    Args:
        cfg (DictConfig): Objeto de configuracao que contem todas as configuracoes
                          carregadas dos arquivos YAML.
    """
    # Converter o DictConfig para um dicionario normal do Python
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    # Valida o dicionario usando o schema definido no Pydantic
    config = ConfigSchema(**cfg_dict)
    print("Configurações carregadas e validadas com sucesso:")
    print(config.model_dump_json(indent=4))

    # Instancia o DataModule
    data_config_dict = config.data.model_dump(by_alias=True)
    data_module = hydra.utils.instantiate(data_config_dict)

    # Instancia a Task
    print("\nInstanciando a Task (modelo, otimizador, funcao de perda)")
    task_conf = config.task.model_dump(by_alias=True)
    task = hydra.utils.instantiate(task_conf)

    # Instancia o Trainer
    print("\nInstanciando o Trainer do PyTorch Lightning")
    trainer_conf = config.trainer.model_dump(by_alias=True)
    trainer = hydra.utils.instantiate(trainer_conf)

    print("\nIniciando o treinamento")
    trainer.fit(model=task, datamodule=data_module)

    print("\nIniciando o teste")
    trainer.test(model=task, datamodule=data_module)


if __name__ == "__main__":
    main()