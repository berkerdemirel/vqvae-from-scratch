import torch
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset

from vqvae.data.datamodule import ImageNetDM


def test_dataloaders():
    """Tests that the dataloaders can be created and have the correct batch size."""
    # 1. Create a dummy dataset
    dummy_dataset = TensorDataset(torch.randn(10, 3, 32, 32))

    # 2. Create a mock config for the datamodule
    dm_cfg = OmegaConf.create(
        {
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "shuffle": False,
        }
    )

    # 3. Instantiate ImageNetDM with the dummy dataset and mock config
    dm = ImageNetDM(
        datamodule_cfg=dm_cfg,
        train_dataset=dummy_dataset,
        val_dataset=dummy_dataset,
    )

    # 4. Get the dataloaders
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    # 5. Assert that the batch sizes are correct
    assert train_loader.batch_size == 2
    assert val_loader.batch_size == 2

    # 6. Check if a batch can be retrieved and has the correct shape
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    assert train_batch[0].shape == (2, 3, 32, 32)
    assert val_batch[0].shape == (2, 3, 32, 32)
