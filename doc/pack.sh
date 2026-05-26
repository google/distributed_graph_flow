#!/bin/bash
#
# This script compiles the documentation using MkDocs and packages it into a
# standalone .zip file.

set -vex

# Configuration
TARGET_DIR="third_party/py/dgf"
CONFIG_PATH="doc/mkdocs.yml"
VENV_PATH="$HOME/dgf_doc_venv"
ZIP_BASE_NAME="$HOME/dgf_docs"

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

# Create a temporary directory for building
TEMP_DIR=$(mktemp -d)
BUILD_DIR="$TEMP_DIR/dgf_docs"
mkdir "$BUILD_DIR"
echo "Created temporary build directory: $BUILD_DIR"

# Build the documentation
echo "Building MkDocs..."
mkdocs build -f "$CONFIG_PATH" -d "$BUILD_DIR"

# Pack it into a zip file in the user home directory
echo "Packing documentation into ${ZIP_BASE_NAME}.zip..."
# Remove old zip if exists
rm -f "${ZIP_BASE_NAME}.zip"

(cd "$TEMP_DIR" && zip -r "${ZIP_BASE_NAME}.zip" dgf_docs)

# Clean up the temporary directory
rm -rf "$TEMP_DIR"

echo "Done. Documentation packed to ${ZIP_BASE_NAME}.zip"
