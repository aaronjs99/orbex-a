# ORBEX-A Installation Guide

## Quick Start

```bash
# Clone and install
git clone https://github.com/aaronjohnsabu1999/orbex-a.git
cd orbex-a
pip install -e .

# Verify
python -c "import orbexa; print(orbexa.__version__)"
```

## Prerequisites

- Python >= 3.8
- pip >= 21.0

## Installation Options

### Option 1: Basic Install (GEKKO solver only)
```bash
pip install -e .
```

### Option 2: With CasADi Solver
```bash
pip install -e ".[casadi]"
```

### Option 3: With 3D Visualization (Mayavi)
```bash
pip install -e ".[visualization]"
```

### Option 4: Full Installation
```bash
pip install -e ".[all]"
```

### Option 5: Using Shell Script
```bash
chmod +x install_dependencies.sh
./install_dependencies.sh           # Basic
./install_dependencies.sh --casadi  # With CasADi
./install_dependencies.sh --full    # Everything
```

## Solver Configuration

Edit `config/default.yaml` to switch between solvers:

```yaml
solver:
  backend: "gekko"  # Options: "gekko", "casadi", "scipy"
```

## Conda Environment (Recommended)

```bash
conda create -n orbexa python=3.9
conda activate orbexa
pip install -e .
```

## Troubleshooting

### Mayavi Build Fails
Mayavi requires VTK. On Ubuntu:
```bash
sudo apt-get install libvtk7-dev
pip install vtk mayavi
```

### Keyboard Package Permission Error
```bash
sudo pip install keyboard
# OR run scripts with sudo
```

## Running Simulations

```bash
python run.py --help
python run.py  # Default simulation
```
