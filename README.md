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
