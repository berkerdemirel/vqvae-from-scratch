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
            z_flat_fp32 = z_flat.float()
            embedding_fp32 = self.embedding.weight.float()
            distances = torch.cdist(z_flat_fp32, embedding_fp32, p=2).pow(2)
            # distances = torch.cdist(z_flat, self.embedding.weight, p=2).pow(2)
            # if distances.dtype in (torch.bfloat16, torch.float16):
            # distances = distances + 1e-4  # numerical guard

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
    def __init__(self, channels: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        z = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        z = self.norm(z)
        z, _ = self.attn(z, z, z, need_weights=False)
        z = z.transpose(1, 2).reshape(b, c, h, w)
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
        # ch_mult = [1, 2, 4, 4]  # per level
        ch_mult = [1, 1, 2, 2, 4]  # five levels
        n_res = 2  # blocks per level

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
        ch = cfg.start_channels  # 128
        ch_mult = [4, 2, 2, 1, 1]  # mirror of encoder
        n_res = 2  # mirror of encoder

        layers = [nn.Conv2d(cfg.embedding_dim, ch * ch_mult[0], 3, 1, 1)]
        in_ch = ch * ch_mult[0]  # 512 when ch=128

        layers.append(SelfAttention(in_ch))  # 16×16 attention

        # main up‑sampling stack
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(n_res):
                layers.append(ResBlock(in_ch, out_ch))
                in_ch = out_ch
            if i < len(ch_mult) - 1:  # add 4 up‑samplers
                next_out_ch = ch * ch_mult[i + 1]
                layers.append(Up(in_ch, next_out_ch))
                in_ch = next_out_ch

        # final 3‑×‑3 conv to RGB and tanh
        layers.append(GNConv(in_ch, cfg.image_channels, k=3))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
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


class PatchDiscriminator(nn.Module):
    """Spectral norm PatchGAN discriminator compatible with VQGAN.

    Produces a 1 channel logits map (B,1,H',W').
    """

    def __init__(self, cfg: DictConfig, n_layers: int = 3, use_batchnorm: bool = False):
        super().__init__()

        def sn_conv(in_ch, out_ch, k=4, s=2, p=1, apply_bn=False):
            conv = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, k, s, p))
            layers = [conv]
            if apply_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers  # list of layers

        layers = []

        # -------- first layer (no BN) --------
        in_ch = cfg.image_channels
        out_ch = cfg.start_channels  # 64 or 128
        layers.extend(sn_conv(in_ch, out_ch, apply_bn=False))

        # -------- intermediate layers ---------
        for i in range(1, n_layers + 1):
            in_ch = out_ch
            out_ch = min(out_ch * 2, 512)  # cap channels if you wish
            stride = 1 if i == n_layers else 2
            layers.extend(
                sn_conv(
                    in_ch,
                    out_ch,
                    k=4,
                    s=stride,
                    p=1,
                    apply_bn=use_batchnorm,
                )
            )

        # -------- final 1×1 conv to logits -----
        layers.append(nn.utils.spectral_norm(nn.Conv2d(out_ch, 1, kernel_size=4, stride=1, padding=1)))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
