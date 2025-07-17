import torch
from omegaconf import DictConfig

from vqvae.modules.module import VQGAN_G


def test_vqgan_g_forward_pass_and_usage():
    """
    Tests the forward pass of the VQGAN_G model and ensures codebook usage.
    """
    # Configuration based on the new architecture
    cfg = DictConfig(
        {
            "image_channels": 3,
            "embedding_dim": 256,
            "codebook_size": 1024,
            "beta": 0.25,
            "start_channels": 128,
        }
    )

    # Instantiate the model
    model = VQGAN_G(cfg)
    model.train()  # Set to train mode to update usage stats

    # Create a dummy input tensor
    batch_size = 2
    image_size = 256
    input_tensor = torch.randn(batch_size, cfg.image_channels, image_size, image_size)

    # Perform a forward pass
    with torch.no_grad():
        x_recon, vq_loss, indices = model(input_tensor)

    # Assert output shapes
    assert x_recon.shape == input_tensor.shape, "Reconstructed image shape is incorrect."
    assert torch.isfinite(vq_loss).all(), "VQ loss should be a finite scalar tensor."

    # The downsampling factor is 2^3 = 8
    latent_size = image_size // 8
    assert indices.shape == (
        batch_size,
        latent_size,
        latent_size,
    ), "Indices shape is incorrect."

    # Check codebook usage
    assert model.quantizer.code_usage.sum() > 0, "Codebook usage should be updated."
    assert model.quantizer.code_usage.nonzero().numel() > 0, "Some codes should have been used."

    print("VQGAN_G forward pass and code usage test passed successfully!")
    print(f"Reconstructed shape: {x_recon.shape}, VQ Loss: {vq_loss.item()}")
    print(f"Indices shape: {indices.shape}")
    print(f"Number of used codes: {model.quantizer.code_usage.nonzero().numel()}")
