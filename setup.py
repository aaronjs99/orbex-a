#!/usr/bin/env python3
"""setup.py: package metadata and build entry for pip and ROS 2 ament_python.

All metadata lives here, not in ``pyproject.toml``, because colcon's symlink
build runs ``setup.py develop`` in a build directory that links ``setup.py``,
``setup.cfg``, the packages and the data files, but not ``pyproject.toml``.
``scripts/`` is installed as the ``orbexa`` package. The ament resource marker
and ``package.xml`` are installed so ``colcon build`` registers the package.

Copyright: Aaron John Sabu
"""

import re
from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
VERSION = re.search(
    r'__version__ = "([^"]+)"', (ROOT / "scripts" / "__init__.py").read_text()
).group(1)
CERTIFICATION = ["python-flint>=0.8,<0.10", "zstandard==0.25.0"]

setup(
    name="orbexa",
    version=VERSION,
    description="Adaptive dynamic tube MPC for multi-chaser capture and disposal "
    "of a tumbling spacecraft",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Aaron John Sabu, Brett T. Lopez",
    author_email="aaronjs@g.ucla.edu",
    url="https://github.com/aaronjs99/orbex-a",
    license="GPL-2.0-only",
    python_requires=">=3.10",
    install_requires=["numpy", "scipy", "gekko", "matplotlib", "pyyaml"],
    extras_require={
        "casadi": ["casadi"],
        "video": ["imageio-ffmpeg"],
        "studio": ["mistune>=3,<4"],
        "certification": CERTIFICATION,
        "all": [
            "casadi",
            "imageio-ffmpeg",
            "mistune>=3,<4",
            *CERTIFICATION,
        ],
    },
    package_dir={"orbexa": "scripts"},
    packages=["orbexa", *("orbexa." + name for name in find_packages("scripts"))],
    package_data={
        "orbexa.figures": [
            "studio/*",
            "studio/vendor/*",
            "studio/vendor/three/*",
            "studio/vendor/katex/*",
            "studio/vendor/katex/fonts/*",
        ]
    },
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/orbexa"]),
        ("share/orbexa", ["package.xml"]),
    ],
    entry_points={
        "console_scripts": [
            "orbexa-run-nonlinear = orbexa.cli.run_nonlinear:main",
            "orbexa-serve-studio = orbexa.cli.serve_studio:main",
            "orbexa-figures = orbexa.cli.figures:main",
        ]
    },
)
