import hydra
from omegaconf import DictConfig


def test_config_loads_correctly() -> None:
    """Tests that the hydra configuration loads without errors."""
    with hydra.initialize(config_path="../conf", version_base=None):
        cfg: DictConfig = hydra.compose(config_name="default")
        print(cfg)

        # Check that the main config groups are present
        assert "dataset" in cfg
        assert "datamodule" in cfg
        assert "model" in cfg
        assert "pl" in cfg

        # Check for a specific key in each group to be extra sure
        assert "resolution" in cfg.dataset
        assert "batch_size" in cfg.datamodule
        assert "embedding_dim" in cfg.model
        assert "max_epochs" in cfg.pl
