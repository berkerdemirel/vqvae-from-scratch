from typing import Optional

import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset


class ImageNetDM(pl.LightningDataModule):
    """A PyTorch Lightning DataModule for the ImageNet dataset."""

    def __init__(
        self,
        datamodule_cfg: DictConfig,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
    ):
        """Initializes the DataModule.

        This DataModule follows a dependency injection pattern, where the
        datasets are created outside and passed in as arguments.

        Args:
            datamodule_cfg: The configuration for the datamodule (batch size, etc.).
            train_dataset: The dataset for the training set.
            val_dataset: The dataset for the validation set.
        """
        super().__init__()
        # `save_hyperparameters` is a Pytorch Lightning convenience function
        # that makes the arguments available as properties (e.g. self.hparams)
        # and logs them to the logger.
        self.save_hyperparameters(datamodule_cfg)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the training set."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=self.hparams.shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers if self.hparams.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the validation set."""
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers if self.hparams.num_workers > 0 else False,
        )
