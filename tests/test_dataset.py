import os

import hydra
import pytest
import torch
from omegaconf import DictConfig

from vqvae.data.dataset import ImageNetDataset

# Load the hydra config to get the dataset path
with hydra.initialize(config_path="../conf", version_base=None):
    cfg: DictConfig = hydra.compose(config_name="default")

# Get the validation data path from the config
IMAGENET_VAL_PATH = cfg.dataset.dir_val

# Conditionally skip the test if the data path does not exist
requires_imagenet = pytest.mark.skipif(
    not os.path.exists(IMAGENET_VAL_PATH),
    reason=f"ImageNet validation data not found at {IMAGENET_VAL_PATH}",
)


@requires_imagenet
def test_imagenet_dataset_loads() -> None:
    """
    Tests that the ImageNetDataset can be instantiated and loads a sample
    from the actual validation data directory specified in the config.
    """
    # Instantiate the dataset for the validation set
    ds = ImageNetDataset(cfg.dataset, train=False)

    # Assert that the dataset is not empty
    assert len(ds) > 0, "Dataset should not be empty."

    # Assert that a sample has the correct shape
    sample = ds[0]
    assert isinstance(sample, torch.Tensor), "Sample should be a torch.Tensor."
    expected_shape = (3, cfg.dataset.resolution, cfg.dataset.resolution)
    assert sample.shape == expected_shape, f"Sample shape should be {expected_shape}, but got {sample.shape}."
