'''
Original Code from: https://github.com/PaulBorneP/Xview2_Strong_Baseline/tree/master

Edits:
 - modified to run final evaluation on wind_hold on last epoch weights
 - commented out prediction outputs

'''

import os

import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf
import wandb

# modified to run final evaluation on wind_hold

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> pl.Trainer:
    pl.seed_everything(cfg.seed)
    config = OmegaConf.to_container(cfg, resolve=True)
    data_module = hydra.utils.instantiate(cfg.data)
    network = hydra.utils.instantiate(cfg.network)
    wandb.init(
        project=cfg.logger.project,
        config=config,
        group=cfg.group,
        name=cfg.name,
    )
    wandb.watch(network, log_freq=1000)
    trainer_partial = hydra.utils.instantiate(cfg.trainer)
    logger = hydra.utils.instantiate(cfg.logger)
    logger.log_hyperparams(config)
    trainer = trainer_partial(logger=logger)
    trainer.fit(network, data_module)

    # NEW: Final evaluation on test_dirs (wind_hold) using last epoch weights
    print("Running final evaluation on wind_hold using last epoch weights")

    trainer.test(model=network, datamodule=data_module, ckpt_path=None)
    #trainer.predict(network, data_module)

    return trainer


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    main()
