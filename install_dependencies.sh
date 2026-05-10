#!/bin/bash
# /***********************************************************
# *                                                         *
# * Copyright (c) 2022                                      *
# *                                                         *
# * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
# * University of California, Los Angeles                   *
# *                                                         *
# * Authors: Aaron John Sabu                                *
# * Contact: {aaronjs, btlopez}@ucla.edu                    *
# *                                                         *
# ***********************************************************/

# ORBEX-A Installation Script
# Usage: ./install_dependencies.sh [--full|--casadi|--viz]

set -e

echo "========================================="
echo "  ORBEX-A Dependency Installation"
echo "========================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python >= $REQUIRED_VERSION is required (found $PYTHON_VERSION)"
    exit 1
fi

echo "Python version: $PYTHON_VERSION ✓"
echo ""

# Parse arguments
INSTALL_CASADI=false
INSTALL_VIZ=false

for arg in "$@"; do
    case $arg in
        --full)
            INSTALL_CASADI=true
            INSTALL_VIZ=true
            ;;
        --casadi)
            INSTALL_CASADI=true
            ;;
        --viz)
            INSTALL_VIZ=true
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --full     Install all optional dependencies (CasADi + Mayavi)"
            echo "  --casadi   Install CasADi solver backend"
            echo "  --viz      Install Mayavi 3D visualization (requires VTK)"
            echo "  --help     Show this message"
            exit 0
            ;;
    esac
done

# Create virtual environment if specified
if [ "$CREATE_VENV" = true ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
fi

# Install core dependencies
echo "Installing core dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install CasADi if requested
if [ "$INSTALL_CASADI" = true ]; then
    echo ""
    echo "Installing CasADi solver..."
    pip install casadi>=3.5.0
fi

# Install Mayavi if requested
if [ "$INSTALL_VIZ" = true ]; then
    echo ""
    echo "Installing Mayavi visualization (this may take a while)..."
    pip install mayavi
fi

# Install package in editable mode
echo ""
echo "Installing ORBEX-A package..."
pip install -e .

echo ""
echo "========================================="
echo "  Installation Complete!"
echo "========================================="
echo ""
echo "Verify installation with:"
echo "  python -c \"import orbexa; print(orbexa.__version__)\""
echo ""
echo "Run a simulation with:"
echo "  python run.py --help"
echo ""
