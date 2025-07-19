import hydra

# isort: off
import vqvae.quiet as _quiet  # must run before pl to set warning filters

# isort: on
import pytorch_lightning as pl
from omegaconf import DictConfig


def train(cfg: DictConfig) -> None:
    """Main training routine for VQ-GAN.

    Args:
        cfg: The hydra configuration object.

    Returns:
        The output of the trainer's test function, if testing is enabled.
    """
    pl.seed_everything(cfg.seed)

    # Instantiate the modules
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer)
    # Start training
    trainer.fit(model=model, datamodule=datamodule)


@hydra.main(config_path="../../conf", config_name="experiment", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """The main function to run the training.

    Args:
        cfg: The hydra configuration object.
    """
    train(cfg)


if __name__ == "__main__":
    main()
