import os
from typing import Any, List

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from vqvae.data.datamodule import ImageNetDM  # noqa: F401
from vqvae.data.dataset import ImageNetDataset  # noqa: F401
from vqvae.pl_modules.pl_module import VQGANLitModule


def train(cfg: DictConfig) -> Any:
    """Main training routine for VQ-GAN.

    Args:
        cfg: The hydra configuration object.

    Returns:
        The output of the trainer's test function, if testing is enabled.
    """
    pl.seed_everything(cfg.train.seed)

    # Instantiate the datamodule
    train_dataset: torch.utils.data.Dataset = ImageNetDataset(cfg.dataset, train=True)  # noqa: F401
    val_dataset: torch.utils.data.Dataset = ImageNetDataset(cfg.dataset, train=True)  # noqa: F401
    datamodule: pl.LightningDataModule = ImageNetDM(
        datamodule_cfg=cfg.datamodule,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    # Instantiate the Lightning module
    model: pl.LightningModule = VQGANLitModule(cfg)

    # Instantiate callbacks from config
    callbacks: List[pl.Callback] = []
    if "callbacks" in cfg.train:
        for _, cb_conf in cfg.callbacks.items():
            callbacks.append(hydra.utils.instantiate(cb_conf))

    # Instantiate the logger from config
    logger = hydra.utils.instantiate(cfg.logger)

    # Instantiate the trainer
    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Start training
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.train.get("ckpt_path"))

    # Test the model
    if cfg.train.get("test_after_training", False):
        test_results = trainer.test(model=model, datamodule=datamodule)
        return test_results

    return None


@hydra.main(config_path="../../conf", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    """The main function to run the training.

    Args:
        cfg: The hydra configuration object.
    """
    os.environ["WANDB_DISABLE_CODE"] = "true"  # to avoid wandb saving code
    train(cfg)


if __name__ == "__main__":
    main()
