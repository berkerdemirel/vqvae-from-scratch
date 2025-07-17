In this project we will implement VQGANs for both generative task and tokenization of images. please use pytorch lightning, hydra and wandb for logging project mgmt etc.


# TODO for `vqvae-from-scratch`

---

## 0. Repo hygiene

*Start by pruning the parts of ****grok‑ai/nn‑template**** that we don’t need.*

**Delete / ignore:** `app/`, `dvc.yaml`, `.dvc/`, `docs/`, `mkdocs.yml`, Hugging‑Face helpers, PyPI / Pages workflows (`release.yml`, `pages.yml`), `cli.py`, `nn_core/registry*`, template hooks.

---

## 1  Environment (with **uv**)

```bash
# 1. install uv (one‑time)
pipx install uv   # or: pip install uv

# 2. create + activate env
uv venv .venv
source .venv/bin/activate   # fish/zsh variants apply

# 3. core deps
uv pip install torch torchvision torchaudio \   # select CUDA wheel
               pytorch-lightning hydra-core wandb lpips einops tqdm
# opt‑ins
uv pip install jinja2 rich pre‑commit pytest

# 4. lock
uv pip freeze > requirements.lock

# 5. git hooks
pre-commit install
```

---

## 2  Configuration files (`configs/`)

> Each YAML is addressable as `cfg.<section>` in constructors.

### 2.1 `dataset.yaml`

```yaml
dir_train: /nfs/scistore19/locatgrp/bdemirel/data/imagenet/train
dir_val:   /nfs/scistore19/locatgrp/bdemirel/data/imagenet/val
resolution: 256          # output size
augment:
  horizontal_flip: true
  random_crop: true
```

### 2.2 `datamodule.yaml`

```yaml
batch_size:  256
num_workers: 8
pin_memory:  true
shuffle:     true
```

### 2.3 `model.yaml` (VQGAN)

```yaml
embedding_dim:   256   # z_dim
codebook_size:   1024  # K
beta:            0.25  # commitment cost
n_res_blocks:    2
channel_base:    128
channel_max:     512
f_downsample:    16     # 4 downsamples (256→16)
lambda_rec:      1.0
lambda_perc:     1.0
lambda_gan:      0.1
```

### 2.4 `pl.yaml` (trainer)

```yaml
max_epochs:        300
precision:        16   # mixed‑precision
accelerator:      auto
lr:               2e-4   # G & E
lr_d:             2e-4   # D
betas:            [0.5, 0.9]
gradient_clip_val: 1.0
checkpoint_every_n: 1
```

### 2.5 `defaults.yaml`

```yaml
defaults:
  - dataset: dataset
  - datamodule: datamodule
  - model: model
  - pl: pl
```

---

## 3  Project scaffolding (`src/`)

| file            | responsibility                                                                         |
| --------------- | -------------------------------------------------------------------------------------- |
| `dataset.py`    | `ImageNetDataset(cfg.dataset)` → returns `torch.utils.data.Dataset` with augmentations |
| `datamodule.py` | `ImageNetDM(cfg)` → `LightningDataModule`                                              |
| `model.py`      | `VQGAN_G`, `PatchDiscriminator`, `VectorQuantizerEMA`                                  |
| `pl_module.py`  | `VQGAN(cfg)` (`LightningModule`) – losses, optimizers, logging                         |
| `utils.py`      | LPIPS wrapper, EMA, `count_params`, init helpers                                       |

> **Constructor rule:** each class receives its dedicated `cfg.<section>` object only, e.g. `ImageNetDataset(cfg)` gets **dataset** cfg, not the full tree.

---

## 4  Unit tests (smoke‑level)

Create `tests/` with one test per component.

```python
# tests/test_dataset.py
from omegaconf import OmegaConf
from src.dataset import ImageNetDataset


def test_dataset_len():
    cfg = OmegaConf.create(
        {"dir_train": "dummy", "dir_val": "dummy", "resolution": 256, "augment": {}}
    )
    ds = ImageNetDataset(cfg, train=False)
    assert len(ds) == 0  # placeholder until data mounted
```

```python
# tests/test_datamodule.py
from omegaconf import OmegaConf
from src.datamodule import ImageNetDM


def test_dataloaders():
    dm = ImageNetDM(
        OmegaConf.create(
            {"batch_size": 2, "num_workers": 0, "pin_memory": False, "shuffle": False}
        )
    )
    dm.setup(None)
    loader = dm.train_dataloader()
    assert loader.batch_size == 2
```

```python
# tests/test_model.py
import torch
from omegaconf import OmegaConf
from src.model import VQGAN_G


def test_forward():
    cfg = OmegaConf.create(
        {
            "embedding_dim": 256,
            "codebook_size": 1024,
            "channel_base": 64,
            "channel_max": 256,
            "f_downsample": 16,
        }
    )
    net = VQGAN_G(cfg)
    x = torch.randn(1, 3, 256, 256)
    rec, _, _ = net(x)
    assert rec.shape == x.shape
```

```python
# tests/test_pl_module.py
import torch
from omegaconf import OmegaConf
from src.pl_module import VQGAN

cfg = OmegaConf.create(
    {
        "model": {
            "embedding_dim": 256,
            "codebook_size": 1024,
            "channel_base": 64,
            "channel_max": 256,
            "f_downsample": 16,
            "beta": 0.25,
            "lambda_rec": 1.0,
            "lambda_perc": 1.0,
            "lambda_gan": 0.1,
        },
        "pl": {"lr": 1e-4, "lr_d": 1e-4, "betas": [0.5, 0.9]},
    }
)

plm = VQGAN(cfg)


def test_step():
    imgs = torch.randn(2, 3, 256, 256)
    out = plm.training_step(imgs, 0, 0)
    assert out is not None
```

Add a minimal GitHub Action `python -m pytest -q` to ensure tests run on pushes.

---

## 5  Documentation

- Replace template `README.md` with:
  - quick environment setup
  - `python train.py` usage (Hydra overrides example)
  - expected wandb dashboard screenshots (optional)
- We drop MkDocs—plain markdown is sufficient.

---

## 6  Milestones / execution order

1. **Configs complete** → verify `OmegaConf.load` works.
2. Implement **dataset.py** → unit test passes.
3. Implement **datamodule.py** → loaders iterate a batch.
4. Implement **model.py** (encoder/decoder, VQ, disc) → forward test green.
5. Implement **pl\_module.py** (losses, optims, logging) → smoke train step test.
6. Add CI with pytest; ensure GPU optional path.
7. Run reconstruction‑only training; evaluate FID.
8. Add perceptual + GAN losses, tune λ schedule.
9. Freeze VQGAN; export checkpoint; log as wandb artifact.
10. Train autoregressive transformer on code indices (separate module).

---
