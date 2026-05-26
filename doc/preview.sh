#!/bin/bash
#
# This script compiles the project documentation using MkDocs and starts a local
# HTTP server to preview it.

set -vex

# Configuration
TARGET_DIR="third_party/py/dgf"
CONFIG_PATH="doc/mkdocs.yml"
VENV_PATH="$HOME/dgf_doc_venv"
ADDRESS="0.0.0.0:8889"

# Create the venv in the home directory if it doesn't exist
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment at $VENV_PATH..."
    python3 -m venv "$VENV_PATH"
fi

# Navigate to the project directory
echo "Moving to $TARGET_DIR..."
cd "$TARGET_DIR" || { echo "Error: Directory $TARGET_DIR not found!"; exit 1; }

# Activate using the absolute path
source "$VENV_PATH/bin/activate"

# Install/Update requirements
echo "Updating dependencies..."
pip install --upgrade pip
pip install -r doc/requirements.txt

# Serve the documentation
echo "Starting MkDocs on $ADDRESS..."
mkdocs serve -a "$ADDRESS" -f "$CONFIG_PATH"
