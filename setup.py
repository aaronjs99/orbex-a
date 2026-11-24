#!/usr/bin/env python3
"""setup.py: setuptools entry for pip and ROS 2 ament_python builds.

``pyproject.toml`` holds the package metadata. This file maps ``scripts/`` to
the ``orbexa`` package and installs the ament resource marker and
``package.xml`` so ``colcon build`` registers the package.

Copyright: Aaron John Sabu
"""

from setuptools import find_packages, setup

setup(
    packages=["orbexa", *("orbexa." + name for name in find_packages("scripts"))],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/orbexa"]),
        ("share/orbexa", ["package.xml"]),
    ],
)
