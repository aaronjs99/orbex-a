# ORBEX-A: Orbital Rendezvous and Capture Experiment with Adaptive Tube MPC

**ORBEX-A** is a Python-based simulator for spacecraft performing cooperative orbital rendezvous and capture tasks. The system implements Adaptive Dynamic Tube Model Predictive Control (ADTMPC) for the capture of tumbling targets.

This repository contains simulation code, configuration files, visualizations, and evaluation data for the system described in the IEEE Aerospace Conference paper:  
“Capturing Tumbling Objects in Orbit with Adaptive Tube Model Predictive Control” (Aaron John Sabu and Brett T Lopez).

## Features

- Introduction of Adaptive Tube-based MPC (ADTMPC)
- Chaser-target orbital dynamics and estimation
- Distributed task allocation and multi-agent coordination
- High-resolution trajectory visualizations

## File Structure

```
.
├── config/                         # YAML-based configuration for scenarios
├── paper/                          # Paper assets and figures
├── src/
│   └── orbexa/                     # Core simulation modules and MPC logic
├── tests/                          # Automated test suite
├── INSTALL.md                      # Detailed installation guide
├── LICENSE                         # GPL-2.0 license
├── README.md                       # Project documentation
├── install_dependencies.sh         # Helper script for dependency installation
├── pyproject.toml                  # Project metadata and dependencies
├── requirements.txt                # Pip requirements file
├── run.py                          # Entrypoint for running ADTMPC simulations
└── setup.py                        # Legacy setup script
```

## Getting Started

You can install dependencies and run a test scenario. For more detailed installation options (e.g., CasADi, Mayavi), see [INSTALL.md](INSTALL.md).

```bash
conda create -n orbexa python=3.9
conda activate orbexa
pip install -e .
python run.py
```

This will run a sample adaptive MPC simulation.

## License

GPLv2 License. See `LICENSE` for details.

## Citation

If you use this work, please cite:

```
@inproceedings{johnsabu2025orbexa,
  title={Capturing Tumbling Objects in Orbit with Adaptive Tube Model Predictive Control},
  author={Aaron John Sabu and Brett T. Lopez},
  booktitle={IEEE Aerospace Conference},
  year={2025}
}
```
