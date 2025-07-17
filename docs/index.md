# vqvae-from-scratch

<p align="center">
    <a href="https://github.com/berkerdemirel/vqvae-from-scratch/actions/workflows/test_suite.yml"><img alt="CI" src=https://img.shields.io/github/workflow/status/berkerdemirel/vqvae-from-scratch/Test%20Suite/main?label=main%20checks></a>
    <a href="https://berkerdemirel.github.io/vqvae-from-scratch"><img alt="Docs" src=https://img.shields.io/github/deployments/berkerdemirel/vqvae-from-scratch/github-pages?label=docs></a>
    <a href="https://github.com/grok-ai/nn-template"><img alt="NN Template" src="https://shields.io/badge/nn--template-0.4.0-emerald?style=flat&labelColor=gray"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.11-blue.svg"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

vqvae-from-scratch implementation


## Installation

```bash
pip install git+ssh://git@github.com/berkerdemirel/vqvae-from-scratch.git
```


## Quickstart

[comment]: <> (> Fill me!)


## Development installation

Setup the development environment:

```bash
git clone git@github.com:berkerdemirel/vqvae-from-scratch.git
cd vqvae-from-scratch
conda env create -f env.yaml
conda activate vqvae
pre-commit install
```

Run the tests:

```bash
pre-commit run --all-files
pytest -v
```


### Update the dependencies

Re-install the project in edit mode:

```bash
pip install -e .[dev]
```
