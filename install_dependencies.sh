#!/bin/bash

set -euo pipefail

# Prefer project venv Python if present; otherwise fallback to python3/python
PY="python3"
if [ -x "./env-anemll-bench/bin/python" ]; then
    PY="./env-anemll-bench/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
elif command -v python >/dev/null 2>&1; then
    PY="python"
fi
PIP="$PY -m pip"

echo "Using Python: $($PY -c 'import sys; print(sys.executable)')"

# Upgrade pip inside the chosen interpreter (avoids system Python 2)
$PIP install --upgrade pip

# Detect version (major.minor) without bc
PYTHON_VERSION=$($PY -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Detected Python version: $PYTHON_VERSION"

# Gentle warning if not 3.9
if [ "${PYTHON_VERSION%%.*}" != "3" ] || [ "${PYTHON_VERSION#*.}" != "9" ]; then
    echo "⚠️ WARNING: ANEMLL is designed to work best with Python 3.9.x"
    echo "Proceeding with $PYTHON_VERSION; some combinations may require manual tweaks."
fi

# Install PyTorch based on Python version with fallbacks for compatibility
MAJOR=${PYTHON_VERSION%%.*}
MINOR=${PYTHON_VERSION#*.}

if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 13 ]; then
    echo "Python $PYTHON_VERSION detected. Stable PyTorch wheels may be unavailable. Skipping torch install by default."
    echo "If you need nightly, run (inside this environment):"
    echo "  $PIP install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu"
elif [ "$MAJOR" -eq 3 ] && [ "$MINOR" -eq 9 ]; then
    echo "Python 3.9 detected. Installing PyTorch 2.2.2 for maximum compatibility with CoreMLTools on Sequoia..."
    $PIP install "torch==2.2.2" "torchvision==0.17.2" "torchaudio==2.2.2"
elif [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ] && [ "$MINOR" -le 12 ]; then
    echo "Installing PyTorch (stable) for Python $PYTHON_VERSION..."
    $PIP install "torch>=2.5,<2.6" torchvision torchaudio || {
        echo "PyTorch 2.5.x not available, falling back to 2.2.2..."
        $PIP install "torch==2.2.2" "torchvision==0.17.2" "torchaudio==2.2.2"
    }
else
    echo "Installing PyTorch 2.2.2 for Python $PYTHON_VERSION (fallback for compatibility)..."
    $PIP install "torch==2.2.2" "torchvision==0.17.2" "torchaudio==2.2.2"
fi

# Install coremltools and the rest of dependencies
$PIP install "coremltools>=8.2"
# Install the rest of the dependencies (pin NumPy <2 for py3.9 stability)
$PIP install -r requirements.txt

# Verify PyTorch (if installed) and coremltools
$PY - <<'PYEND'
try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'MPS available: {getattr(getattr(torch, "backends", object()), "mps", object()).is_available() if hasattr(getattr(torch, "backends", object()), "mps") else False}')
except Exception as e:
    print(f'PyTorch not installed or failed to import: {e}')
try:
    import coremltools
    print(f'CoreMLTools version: {coremltools.__version__}')
except Exception as e:
    print(f'coremltools import failed: {e}')
PYEND

echo "Installation complete!"