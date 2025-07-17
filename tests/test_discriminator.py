from types import SimpleNamespace

import torch

from vqvae.modules.module import PatchDiscriminator


def test_discriminator_forward_pass():
    """
    Tests the forward pass of the PatchDiscriminator model based on the
    latest architecture with spectral normalization.
    """
    # Use SimpleNamespace for lightweight config as requested
    cfg = SimpleNamespace(image_channels=3, start_channels=128)
    n_layers = 4
    disc = PatchDiscriminator(cfg, n_layers=n_layers)

    # Create a dummy input tensor
    batch_size = 2
    image_size = 256
    x = torch.randn(batch_size, cfg.image_channels, image_size, image_size)

    # Perform a forward pass
    output = disc(x)

    # Calculate the expected output shape based on the architecture
    # Input: 256
    # Layer 0 (s=2): (256 - 4 + 2)/2 + 1 = 128
    # Layer 1 (s=2): (128 - 4 + 2)/2 + 1 = 64
    # Layer 2 (s=2): (64 - 4 + 2)/2 + 1 = 32
    # Layer 3 (s=2): (32 - 4 + 2)/2 + 1 = 16
    # Layer 4 (s=1): (16 - 4 + 2)/1 + 1 = 15
    # Final Layer (s=1): (15 - 4 + 2)/1 + 1 = 14
    expected_size = 14
    expected_shape = (batch_size, 1, expected_size, expected_size)

    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"
    assert torch.isfinite(output).all(), "Discriminator output should be finite."

    print("PatchDiscriminator forward pass test passed successfully!")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
