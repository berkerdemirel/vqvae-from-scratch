import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class VectorQuantizer(nn.Module):
    """Standard Vector Quantizer module.

    This module is a key component of VQ-VAE and VQ-GAN, responsible for
    mapping continuous encoder outputs to a discrete set of codes from a
    learned codebook.

    This implementation is based on the original VQ-VAE paper
    (https://arxiv.org/abs/1711.00937).
    """

    def __init__(self, cfg: DictConfig):
        """Initializes the VectorQuantizer module.

        Args:
            cfg: The configuration for the model, containing parameters like
                 codebook_size, embedding_dim, and beta (commitment cost).
        """
        super().__init__()
        self.num_embeddings = cfg.codebook_size
        self.embedding_dim = cfg.embedding_dim
        self.beta = cfg.beta

        # Initialize the codebook embeddings
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.embedding_dim, 1.0 / self.embedding_dim)

        # Buffer for monitoring codebook usage
        self.register_buffer("code_usage", torch.zeros(self.num_embeddings), persistent=False)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantizes the input tensor and computes the loss.

        Args:
            z: The continuous latent tensor from the encoder.
               Shape: (B, C, H, W)

        Returns:
            A tuple containing:
            - The quantized latent tensor.
            - The total loss for the VQ layer (codebook + commitment loss).
            - The indices of the chosen codebook vectors.
        """
        # Reshape z from (B, C, H, W) to (B*H*W, C)
        b, c, h, w = z.shape
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)

        # Calculate distances using torch.cdist for efficiency and stability
        with torch.amp.autocast(str(z.device), enabled=False):
            distances = torch.cdist(z_flat, self.embedding.weight, p=2).pow(2)
            if distances.dtype in (torch.bfloat16, torch.float16):
                distances = distances + 1e-4  # numerical guard

        # Find the closest codebook vectors
        min_encoding_indices = torch.argmin(distances, dim=1)
        z_q_flat = self.embedding(min_encoding_indices)

        # Reshape quantized vectors back to the original input shape
        z_q = z_q_flat.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        # Use straight-through estimator
        z_q_st = z + (z_q - z).detach()

        # VQ-VAE losses
        codebook_loss = F.mse_loss(z_q, z.detach())
        commitment_loss = F.mse_loss(z, z_q.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss

        # Update code usage for monitoring
        if self.training:
            self.code_usage.index_add_(
                0, min_encoding_indices, torch.ones_like(min_encoding_indices, dtype=torch.float)
            )

        return z_q_st, vq_loss, min_encoding_indices.view(b, h, w)

    # Optional: per‑epoch reset (call from trainer hook)
    def reset_usage(self):
        self.code_usage.zero_()


# ---------- building blocks ----------
class GNConv(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(
            nn.GroupNorm(min(32, in_ch), in_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, k, s, p),
        )


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = GNConv(in_ch, out_ch)
        self.conv2 = GNConv(out_ch, out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.skip(x) + self.conv2(self.conv1(x))


class SelfAttention(nn.Module):
    def __init__(self, ch, n_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(ch, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(ch)

    def forward(self, x):
        b, c, h, w = x.shape
        z = x.view(b, c, -1).permute(0, 2, 1)  # (b, hw, c)
        z = self.ln(z)
        z, _ = self.attn(z, z, z, need_weights=False)
        z = z.permute(0, 2, 1).view(b, c, h, w)
        return x + z


def Down(in_ch, out_ch):  # stride‑2 conv
    return nn.Conv2d(in_ch, out_ch, 4, 2, 1)


def Up(in_ch, out_ch):  # transposed conv
    return nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)


# ---------- encoder & decoder ----------
class Encoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        ch = cfg.start_channels  # 128 by default
        ch_mult = [1, 2, 4, 4]  # per level
        n_res = 3  # blocks per level

        layers = [nn.Conv2d(cfg.image_channels, ch, 3, 1, 1)]
        in_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(n_res):
                layers.append(ResBlock(in_ch, out_ch))
                in_ch = out_ch
            if i < len(ch_mult) - 1:  # no downsample after last level
                layers.append(Down(in_ch, in_ch))

        layers.append(SelfAttention(in_ch))
        layers.append(GNConv(in_ch, cfg.embedding_dim, k=1, p=0))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        ch = cfg.start_channels
        ch_mult = [4, 4, 2, 1]
        n_res = 3

        layers = [nn.Conv2d(cfg.embedding_dim, ch * ch_mult[0], 3, 1, 1)]
        in_ch = ch * ch_mult[0]
        layers.append(SelfAttention(in_ch))

        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(n_res):
                layers.append(ResBlock(in_ch, out_ch))
                in_ch = out_ch
            if i < len(ch_mult) - 1:
                layers.append(Up(in_ch, in_ch // 2))
                in_ch = in_ch // 2

        layers.append(GNConv(in_ch, cfg.image_channels, k=3))
        layers.append(nn.Tanh())  # for normalized images
        self.net = nn.Sequential(*layers)

    def forward(self, z_q):
        return self.net(z_q)


class VQGAN_G(nn.Module):
    """The complete VQ-GAN Generator model."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.encoder = Encoder(cfg)
        self.quantizer = VectorQuantizer(cfg)
        self.decoder = Decoder(cfg)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the full VQ-GAN generator.

        Args:
            x: The input image tensor.

        Returns:
            A tuple containing:
            - The reconstructed image.
            - The VQ loss.
            - The codebook indices.
        """
        z = self.encoder(x)
        z_q, vq_loss, indices = self.quantizer(z)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, indices
