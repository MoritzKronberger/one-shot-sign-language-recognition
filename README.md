# One Shot Recognition of Sign Language Hand Landmarks

Data Science project (Prof. Schöler) MIN WiSe 2023/24

## Getting Started

### Installation

If the project's dependencies should be installed in a separate virtual environment,
create and activate a new virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Dependencies can be installed using pip:

```bash
pip install -r requirements.txt
```

### Development

For a better development experience in VisualStudio Code, install the recommended extension listed in [extensions.json](./.vscode/extensions.json).

Static type checking on the project's core functionality can be performed using:

```bash
mypy core/*.py
```

Linting for the core functionality can be performed with:

```bash
flake8 core
```

## Project Structure

The project is divided into various notebooks for all of the different experiments.

The notebooks are meant to be viewed in the following order:

1. [introduction.ipynb](./introduction.ipynb): background information on the project and basic data exploration
2. [embeddings.ipynb](./embeddings.ipynb): conventional sign language classification and embeddings visualization
3. [contrastive_loss.ipynb](./contrastive_loss.ipynb): One-Shot Learning experiments using Contrastive Loss
4. [triplet_loss.ipynb](./triplet_loss.ipynb): One-Shot Learning experiments using Triplet Loss
5. [n_pair_loss.ipynb](./n_pair_loss.ipynb): One-Shot Learning experiments using Multi-Class N-Pair Loss
6. [prototype_comparison.ipynb](./prototype_comparison.ipynb): comparison of benefits from support prototypes across all One-Shot Learning techniques

Core functionality used in multiple notebooks is separated in the [core module](./core) and imported.
