#!/bin/bash

# Set the path to Python 3.9 from Homebrew
PYTHON39_PATH="/opt/homebrew/opt/python@3.9/bin/python3.9"

# Check if Python 3.9 is installed
if [ ! -f "$PYTHON39_PATH" ]; then
    echo "Python 3.9 is not found at $PYTHON39_PATH"
    echo "Please install it using:"
    echo "  brew install python@3.9"
    exit 1
fi

echo "Found Python 3.9 at $PYTHON39_PATH"

# Check if the environment already exists and remove it if it does
if [ -d "env-anemll-bench" ]; then
    echo "Found existing env-anemll-bench environment. Removing it..."
    rm -rf env-anemll-bench
    echo "Existing environment removed."
fi

# Create a virtual environment with Python 3.9
echo "Creating a fresh virtual environment with Python 3.9..."
"$PYTHON39_PATH" -m venv env-anemll-bench

# Activate the virtual environment
echo "Activating the virtual environment..."
source env-anemll-bench/bin/activate

# Verify Python version
python_version=$(python --version)
echo "Using $python_version"

# Copy the installation files to the new environment
echo "Copying installation files to the new environment..."
cp install_dependencies.sh env-anemll-bench/
cp requirements.txt env-anemll-bench/

echo ""
echo "Python 3.9 virtual environment created successfully!"
echo ""
echo "To activate the environment, run:"
echo "  source env-anemll-bench/bin/activate"
echo ""
echo "Then run the installation script:"
echo "  cd env-anemll-bench"
echo "  ./install_dependencies.sh"
echo ""
echo "After installation, you can run your scripts with Python 3.9" 