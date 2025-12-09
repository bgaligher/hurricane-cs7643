'''
Adapted Code from: https://github.com/PaulBorneP/Xview2_Strong_Baseline/tree/master

This is a new script adapting main.py to run only a last evaluation on the wind_hold
by loading a checkpoint path defined in the sbatch
'''

import os
import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)
    data_module = hydra.utils.instantiate(cfg.data)
    ModelClass = hydra.utils.get_class(cfg.network._target_)
    trainer_cfg = hydra.utils.instantiate(cfg.trainer)
    trainer = trainer_cfg(logger=False)
    ckpt_path = os.environ.get("CKPT_PTH")
    if not ckpt_path:
        raise ValueError("Environment variable CKPT_PTH is not set")
    print(f"Evaluating Lightning checkpoint:{ckpt_path}")
    trainer.test(model=ModelClass, datamodule=data_module, ckpt_path=ckpt_path)

    return None

if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
