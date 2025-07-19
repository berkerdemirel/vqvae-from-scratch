# VQ-GAN From Scratch

<p align="center">
    <a href="https://github.com/berkerdemirel/vqvae-from-scratch/actions/workflows/test_suite.yml"><img alt="CI" src=https://img.shields.io/github/workflow/status/berkerdemirel/vqvae-from-scratch/Test%20Suite/main?label=main%20checks></a>
    <a href="https://berkerdemirel.github.io/vqvae-from-scratch"><img alt="Docs" src=https://img.shields.io/github/deployments/berkerdemirel/vqvae-from-scratch/github-pages?label=docs></a>
    <a href="https://github.com/grok-ai/nn-template"><img alt="NN Template" src="https://shields.io/badge/nn--template-0.4.0-emerald?style=flat&labelColor=gray"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.11-blue.svg"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

This repository contains a from-scratch implementation of a Vector-Quantized Generative Adversarial Network (VQ-GAN). The project is built using PyTorch Lightning for the training pipeline and Hydra for flexible configuration management.

The goal is to provide a clear, well-structured, and easy-to-follow implementation of the VQ-GAN architecture, complete with data modules, model components, and training logic.

## Development Installation

To set up the development environment, you will need `git` and `uv`.

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:berkerdemirel/vqvae-from-scratch.git
    cd vqvae-from-scratch
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    uv venv
    source .venv/bin/activate
    ```

3.  **Install the project in editable mode with development dependencies:**
    ```bash
    uv pip install -e ".[dev]"
    ```

4.  **Set up pre-commit hooks:**
    ```bash
    pre-commit install
    ```

## Running Tests

To ensure everything is set up correctly, run the test suite:

```bash
# Run all pre-commit checks
pre-commit run --all-files

# Run the Python tests
pytest -v
```

## Training

The training is managed by `src/vqvae/main.py` and configured via YAML files in the `conf/` directory using Hydra.

To start a training run with the default configuration, simply run:
```bash
python src/vqvae/main.py
```

### Customizing Training Runs

You can easily override any configuration parameter from the command line. For example:

-   **Change the number of epochs:**
    ```bash
    python src/vqvae/main.py train.trainer.max_epochs=50
    ```

-   **Use a different logger (e.g., WandB):**
    The default logger is configured in `conf/train.yaml`. You can create new logger configs in `conf/train/logger/` and switch between them. For example, to use a hypothetical `wandb.yaml` logger config:
    ```bash
    python src/vqvae/main.py train/logger=wandb
    ```

-   **Modify model parameters:**
    ```bash
    python src/vqvae/main.py model.embedding_dim=512
    ```
