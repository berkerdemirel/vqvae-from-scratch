import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../../conf", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    """Load and print Hydra configuration."""
    print(cfg)


if __name__ == "__main__":
    main()
