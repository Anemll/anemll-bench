#!/bin/bash

# Resolve Python 3.9 from Homebrew dynamically (handles keg-only installs)
# Prioritize native ARM64 Homebrew for Apple Silicon
if [ -x "/opt/homebrew/bin/brew" ]; then
    # Use native ARM64 Homebrew first
    BREW_PREFIX=$(/opt/homebrew/bin/brew --prefix python@3.9 2>/dev/null)
    if [ -n "$BREW_PREFIX" ] && [ -x "$BREW_PREFIX/bin/python3.9" ]; then
        PYTHON39_PATH="$BREW_PREFIX/bin/python3.9"
        echo "Using native ARM64 Homebrew Python 3.9"
    fi
elif command -v brew >/dev/null 2>&1; then
    # Fallback to default brew (may be x86_64 under Rosetta)
    BREW_PREFIX=$(brew --prefix python@3.9 2>/dev/null)
    if [ -n "$BREW_PREFIX" ] && [ -x "$BREW_PREFIX/bin/python3.9" ]; then
        PYTHON39_PATH="$BREW_PREFIX/bin/python3.9"
        echo "Using default Homebrew Python 3.9 (may be x86_64)"
    fi
fi

# Fallbacks for common Homebrew prefixes (Apple Silicon and Intel)
if [ -z "$PYTHON39_PATH" ]; then
    if [ -x "/opt/homebrew/opt/python@3.9/bin/python3.9" ]; then
        PYTHON39_PATH="/opt/homebrew/opt/python@3.9/bin/python3.9"
    elif [ -x "/usr/local/opt/python@3.9/bin/python3.9" ]; then
        PYTHON39_PATH="/usr/local/opt/python@3.9/bin/python3.9"
    fi
fi

# Final fallback: any python3.9 on PATH
if [ -z "$PYTHON39_PATH" ] && command -v python3.9 >/dev/null 2>&1; then
    PYTHON39_PATH="$(command -v python3.9)"
fi

# Check if Python 3.9 is installed
if [ -z "$PYTHON39_PATH" ] || [ ! -x "$PYTHON39_PATH" ]; then
    echo "Python 3.9 was not found."
    echo ""
    echo "For Apple Silicon Macs, install native ARM64 Python:"
    echo "  /opt/homebrew/bin/brew install python@3.9"
    echo ""
    echo "For Intel Macs or if native Homebrew is not available:"
    echo "  brew install python@3.9"
    echo ""
    echo "If already installed (keg-only), it will live at:"
    echo "  ARM64: \"$(/opt/homebrew/bin/brew --prefix python@3.9 2>/dev/null)/bin/python3.9\""
    echo "  x86_64: \"$(brew --prefix python@3.9 2>/dev/null)/bin/python3.9\""
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

# Verify Python version and architecture
python_version=$(python --version)
python_arch=$(python -c "import platform; print(platform.machine())")
echo "Using $python_version"
echo "Architecture: $python_arch"

# Warn if using x86_64 on Apple Silicon
if [[ "$python_arch" == "x86_64" ]] && [[ "$(uname -m)" == "arm64" ]]; then
    echo ""
    echo "⚠️  WARNING: You're using x86_64 Python on Apple Silicon!"
    echo "   This will prevent access to the Apple Neural Engine (ANE)."
    echo "   For proper ANE support, use native ARM64 Python:"
    echo "   /opt/homebrew/bin/brew install python@3.9"
    echo ""
fi

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