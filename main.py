import hydra
from hydra.core.config_store import ConfigStore
import torch
from omegaconf import DictConfig, OmegaConf
from src.configs.project import ProjectConfig


cs = ConfigStore.instance()
cs.store(name="base_config", node=ProjectConfig)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Funcao principal que carrega as configuracoes, instancia os componentes do projeto e inicia o treinamento e teste.

    Args:
        cfg (DictConfig): Objeto de configuracao carregado pelo Hydra, contendo todas as configuracoes.
    """
    print("Configurações carregadas e validadas com sucesso:")
    print(OmegaConf.to_yaml(cfg))

    # Instancia o DataModule
    datamodule = hydra.utils.instantiate(cfg.data)

    # Instancia a Task
    print("\nInstanciando a Task (modelo, otimizador, funcao de perda)")
    task = hydra.utils.instantiate(cfg.task)

    # Instancia o Trainer
    print("\nInstanciando o Trainer do PyTorch Lightning")
    trainer = hydra.utils.instantiate(cfg.trainer)

    print("\nIniciando o treinamento")
    trainer.fit(model=task, datamodule=datamodule)

    print("\nIniciando o teste")
    trainer.test(model=task, datamodule=datamodule)


if __name__ == "__main__":
    main()