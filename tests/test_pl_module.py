import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from vqvae.pl_modules.pl_module import VQGANLitModule


def test_pl_module_training_step(tmp_path):
    """
    Runs two train + two val steps with fast_dev_run to ensure the
    VQGANLitModule forward/optimisation passes are free of NaNs/inf.
    """
    # --- Comprehensive configuration (same as before) ---
    cfg = OmegaConf.create(
        {
            "model": {
                "image_channels": 3,
                "embedding_dim": 16,
                "codebook_size": 32,
                "beta": 0.25,
                "start_channels": 16,
            },
            "train": {
                "lr": 1e-4,
                "max_steps": 100,
                "d_steps": 1,
                "r1_every": 16,
            },
            "pl": {
                "loss": {
                    "recon_weight": 1.0,
                    "perceptual_weight": 1.0,
                    "vq_weight": 1.0,
                    "adversarial_weight": 0.5,
                    "r1_weight": 10.0,
                }
            },
        }
    )

    # --- LightningModule ---
    model = VQGANLitModule(cfg)

    # --- Dummy dataloader (4 samples → 2 batches of size 2) ---
    imgs = torch.randn(4, 3, 64, 64)
    loader = DataLoader(TensorDataset(imgs), batch_size=2)

    # --- Minimal trainer with fast_dev_run ---
    trainer = pl.Trainer(
        accelerator="cpu",  # <-- forces CPU
        fast_dev_run=2,  # 2 train + 2 val iterations total
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )

    # --- Run ---
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)

    # --- Assertions (losses are logged into callback_metrics) ---
    assert torch.isfinite(trainer.callback_metrics["train/total_g_loss"])
    assert torch.isfinite(trainer.callback_metrics["train/d_loss"])
