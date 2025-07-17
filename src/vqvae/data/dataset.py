from omegaconf import DictConfig
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


class ImageNetDataset(Dataset):
    """A wrapper for `torchvision.datasets.ImageFolder`.

    This dataset applies the correct transformations and returns only the image,
    which is required for VQ-GAN.
    """

    def __init__(self, cfg: DictConfig, train: bool = True):
        """Initializes the dataset.

        Args:
            cfg: The dataset configuration.
            train: Whether to use the training or validation set.
        """
        super().__init__()
        self.cfg = cfg
        root_dir = self.cfg.dir_train if train else self.cfg.dir_val

        if train and self.cfg.augment:
            # Training transforms with augmentations
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(self.cfg.resolution),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],  # ImageNet mean
                        std=[0.229, 0.224, 0.225],  # ImageNet std
                    ),
                ]
            )
        else:
            # Validation transforms without augmentations
            transform = transforms.Compose(
                [
                    transforms.Resize(self.cfg.resolution),
                    transforms.CenterCrop(self.cfg.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],  # ImageNet mean
                        std=[0.229, 0.224, 0.225],  # ImageNet std
                    ),
                ]
            )

        self.dataset = ImageFolder(root=root_dir, transform=transform)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        # ImageFolder returns (image, label), we only need the image for VQGAN
        image, _ = self.dataset[idx]
        return image
