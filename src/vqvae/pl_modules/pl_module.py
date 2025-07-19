import lpips
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from omegaconf import DictConfig

from vqvae.modules.module import VQGAN_G, PatchDiscriminator


class VQGANLitModule(pl.LightningModule):
    """PyTorch Lightning module for training the VQ-GAN.

    This module encapsulates the entire training logic, including the generator
    and discriminator, loss functions, and optimization steps.
    """

    def __init__(self, cfg: DictConfig):
        """Initializes the VQGANLitModule.

        Args:
            cfg: The configuration object containing all hyperparameters.
        """
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        self.generator = VQGAN_G(cfg.model)
        self.discriminator = PatchDiscriminator(cfg.model)

        # Loss functions
        self.perceptual_loss = lpips.LPIPS(net="vgg").eval().requires_grad_(False)
        self.disc_start = getattr(cfg.pl.loss, "disc_start", 250_001)  # default = paper

        self.automatic_optimization = False  # We need manual optimization for GANs

    # ------------------------------ device hook ------------------------------
    def setup(self, stage=None):
        self.perceptual_loss.to(self.device)

    # ------------------------------ optimizers --------------------------------
    def configure_optimizers(self):
        """Configures the optimizers and learning rate schedulers."""
        lr = self.cfg.pl.lr
        opt_g = torch.optim.AdamW(self.generator.parameters(), lr=lr, betas=self.cfg.pl.betas, weight_decay=0.0)
        opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=lr, betas=self.cfg.pl.betas, weight_decay=0.0)

        max_steps = self.cfg.pl.max_steps
        sch_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=max_steps, eta_min=lr * 0.1)
        sch_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=max_steps, eta_min=lr * 0.1)

        return (
            [opt_g, opt_d],
            [
                {"scheduler": sch_g, "interval": "step"},
                {"scheduler": sch_d, "interval": "step"},  # ⬅
            ],
        )

    def training_step(self, batch, batch_idx):
        """Manual‑opt training step for VQ‑GAN.

        – LPIPS in no‑grad
        – Grad‑stabilised discriminator (hinge + optional R1)
        – Optionally several D steps per G step
        """
        # ---------------- unpack ----------------
        if isinstance(batch, (list, tuple)):
            images = batch[0]  # take the tensor
        else:
            images = batch
        opt_g, opt_d = self.optimizers()
        sch_g, sch_d = self.lr_schedulers()

        # ====================================================
        # 1. ────────────  GENERATOR  ────────────────────────
        # ====================================================
        self.toggle_optimizer(opt_g)
        self.discriminator.requires_grad_(False)  # freeze D

        x_hat, vq_loss, _ = self.generator(images)

        recon_loss = F.l1_loss(x_hat, images)

        with torch.no_grad():  # LPIPS is frozen
            p_loss = self.perceptual_loss(x_hat, images).mean()

        # ---- skip GAN loss until disc_start ----
        if self.global_step < self.disc_start:
            g_adv = torch.zeros(1, device=self.device)
            train_disc = False
        else:
            g_adv = -self.discriminator(x_hat).mean()
            train_disc = True

        total_g = (
            self.cfg.pl.loss.recon_weight * recon_loss
            + self.cfg.pl.loss.perceptual_weight * p_loss
            + self.cfg.pl.loss.vq_weight * vq_loss
            + self.cfg.pl.loss.adversarial_weight * g_adv
        )

        self.manual_backward(total_g)
        # unused_g = [n for n, p in self.named_parameters() if p.requires_grad and p.grad is None]
        # if unused_g:
        #     print(f"[rank {self.global_rank}]  G‑unused →", unused_g[:10])
        g_grad_norm = torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
        opt_g.step()
        opt_g.zero_grad()
        sch_g.step()
        self.untoggle_optimizer(opt_g)
        self.discriminator.requires_grad_(True)  # unfreeze D

        # Logging
        self.log_dict(
            {
                "train/total_g_loss": total_g,
                "train/recon": recon_loss,
                "train/percept": p_loss,
                "train/vq": vq_loss,
                "train/g_adv": g_adv,
                "train/grad_norm_g": g_grad_norm,
                "lr": opt_g.param_groups[0]["lr"],
            },
            prog_bar=True,
            sync_dist=True,
        )

        if batch_idx == 0 and self.logger is not None:
            grid = torchvision.utils.make_grid(
                torch.cat([images[:4], x_hat[:4]], 0), nrow=4, normalize=True, value_range=(-1, 1)
            )
            # Lightning‑native, works for WandbLogger, TensorBoardLogger, CSVLogger…
            self.logger.log_image(key="recon", images=[grid], step=self.global_step)

        # ====================================================
        # 2. ─────────── DISCRIMINATOR  ──────────────────────
        # ====================================================
        if train_disc:

            d_steps = getattr(self.cfg.train, "d_steps", 1)
            r1_every = getattr(self.cfg.train, "r1_every", 16)
            r1_weight = self.cfg.pl.loss.get("r1_weight", 0.0)

            for _ in range(d_steps):
                self.toggle_optimizer(opt_d)

                logits_real = self.discriminator(images)
                logits_fake = self.discriminator(x_hat.detach())

                d_loss = 0.5 * (F.relu(1.0 - logits_real).mean() + F.relu(1.0 + logits_fake).mean())

                # ── R1 gradient penalty every r1_every steps ──
                if r1_weight > 0 and (self.global_step % r1_every == 0):
                    # images.requires_grad_()
                    real = images.detach().requires_grad_(True)
                    real_logits = self.discriminator(real)
                    grad_real = torch.autograd.grad(
                        outputs=real_logits.sum(), inputs=real, create_graph=False, retain_graph=False
                    )[0]
                    r1_penalty = grad_real.pow(2).view(images.size(0), -1).sum(1).mean()
                    d_loss = d_loss + r1_weight * r1_penalty

                self.manual_backward(d_loss)
                d_grad_norm = torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                # unused_d = [n for n, p in self.named_parameters() if p.requires_grad and p.grad is None]
                # if unused_d:
                #     print(f"[rank {self.global_rank}]  D‑unused →", unused_d[:10])

                opt_d.step()
                opt_d.zero_grad()
                self.untoggle_optimizer(opt_d)

            sch_d.step()

            self.log_dict(
                {
                    "train/d_loss": d_loss,
                    "train/grad_norm_d": d_grad_norm,
                },
                prog_bar=True,
                sync_dist=True,
            )

        return None  # total_g.detach()

    def validation_step(self, batch, batch_idx):
        """Performs a single validation step."""
        if isinstance(batch, (list, tuple)):
            images = batch[0]  # take the tensor
        else:
            images = batch
        x_hat, vq_loss, _ = self.generator(images)

        recon_loss = F.l1_loss(x_hat, images)
        p_loss = self.perceptual_loss(x_hat, images).mean()

        self.log_dict(
            {
                "val/vq_loss": vq_loss,
                "val/p_loss": p_loss,
                "val/recon_loss": recon_loss,
            },
            prog_bar=True,
            sync_dist=True,
            on_epoch=True,
        )
        return x_hat

    def on_validation_epoch_end(self):
        """Resets the codebook usage at the end of each validation epoch."""
        if hasattr(self.generator.quantizer, "reset_usage"):
            self.generator.quantizer.reset_usage()
