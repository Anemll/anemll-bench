# ANEMLL-Bench

## ⚠️ Attention: macOS 15.x is required! ⚠️

This alpha release requires macOS 15. We plan to update support for older OS versions in the next update.

## Overview
ANEMLL-Bench  (pronounced like "animal-bench") is a benchmarking tool specifically designed to measure and evaluate the performance of machine learning models on Apple's Neural Engine (ANE). It provides comprehensive metrics including inference time and memory bandwidth utilization (GB/s) to help researchers and developers optimize their models for Apple Silicon.

This alpha release requires macOS 15. We plan to update support for older OS versions in the next update. Currently, only Memory bandwidth (GB/s) is benchmarked in this release.

ANEMLL-Bench is part on ANEMLL Open Source Project [anemll.com](https://anemll.com)

## 📊 [View Benchmark Results](./Results.MD) 📊

[![Apple Silicon Performance Comparison](./reports/chip_comparison_llama_lm_head.png?v=20250309_v3)](./Results.MD)

Check out our latest [benchmark results](./Results.MD) comparing performance across different Apple Silicon chips (M1, M2, M4 series).

<div align="center">
  <h2>📊 Help Us Build a Comprehensive Benchmark Database! 📊</h2>
  <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #007bff; margin: 20px 0; color: #333333;">
    <h3 style="color: #000000;">🚨 PLEASE SUBMIT YOUR BENCHMARK RESULTS FOR: 🚨</h3>
    <table align="center" style="color: #333333;">
      <tr>
        <td align="center"><strong style="color: #000000;">M2 Series</strong></td>
        <td align="center"><strong style="color: #000000;">M3 Series</strong></td>
        <td align="center"><strong style="color: #000000;">M4 Series</strong></td>
      </tr>
      <tr>
        <td>
          ✓ M2<br>
          ✓ M2 PRO<br>
          ✓ M2 MAX<br>
          ✓ M2 ULTRA
        </td>
        <td>
          ✓ M3<br>
          ✓ M3 PRO<br>
          ✓ M3 MAX<br>
          ✓ M3 ULTRA
        </td>
        <td>
          ✓ M1 ULTRA<br>
          ✓ M4<br>
          ✓ M4 PRO<br>
          ✓ M4 MAX
        </td>
      </tr>
    </table>
    <p style="color: #333333;"><em>📧 Submit results to: <a href="mailto:realanemll@gmail.com" style="color: #0366d6;">realanemll@gmail.com</a> or <a href="https://github.com/Anemll/anemll-bench/issues/new" style="color: #0366d6;">open an issue</a></em></p>
  </div>
</div>

![Sample Benchmark Results](./assets/sample.png)

[**Jump to Quick Start →**](#quick-start)

## Compatibility Notice

⚠️ **Important**: This project is designed to work with **Python 3.9-3.11** and has known compatibility issues with Python 3.13+.

### Python Version Compatibility

- **Recommended**: Python 3.9.x
- **Compatible**: Python 3.10-3.12 (may have minor issues)
- **Not Compatible**: Python 3.13+ (has significant compatibility issues with PyTorch 2.5.0)

### PyTorch Version Compatibility

- **Required for ANEMLL**: PyTorch 2.5.0
- **Issue with Python 3.13+**: PyTorch 2.5.0 is not available for Python 3.13+
- **Workaround for Python 3.13+**: Use PyTorch 2.6.0, but expect potential compatibility issues with coremltools

## Additional Requirements

- macOS with Apple Silicon
- Xcode Command Line Tools installed
- Homebrew (for installing Python 3.9)

## Quick Start

To get started quickly with platform-specific optimized models:

```bash
# Create a virtual environment
python -m venv env-anemll-bench

source env-anemll-bench/bin/activate

pip install -r requirements.txt

pip install -e .

# Download all optimized models for your macOS version
python examples/sync_models.py

# Benchmark all available models and generate a report
python examples/benchmark_all_models.py

# Run a benchmark with a specific pre-optimized model
python examples/load_platform_models.py --model llama_lm_head

# Use existing local models without checking online (prevents downloads)
python examples/benchmark_all_models.py --use-local --no-sync
```

This will automatically download and prepare all the optimized models for your specific macOS version. The models are stored in `~/.cache/anemll-bench/` and are ready to use immediately.

After running benchmarks, check out the [benchmark results](./Results.MD) to see how your device compares to other Apple Silicon chips.

## Features
- Benchmark models on Apple Neural Engine and compare with CPU/GPU performance
- Measure inference time and memory bandwidth utilization (GB/s)
- Download and convert models from Hugging Face to CoreML format
- Automatically collect system information (Mac model, CPU details, memory)
- Generate comprehensive HTML reports with visualizations
- Upload and share reports via multiple services (GitHub Gist, JSONBin, Pastebin)
- (future) Easy-to-use API for integrating new models
- Automatic downloading of platform-specific optimized models (macOS 15.x+)
- Robust model size detection for accurate throughput calculation

## Setup Instructions

### Option 1: Using Python 3.9 (Recommended)

We provide a script to create a Python 3.9 virtual environment:

```bash
# Make the script executable
chmod +x create_python39_env.sh

# Run the script
./create_python39_env.sh

# Activate the environment
source env-anemll-bench/bin/activate

# Install dependencies
cd env-anemll-bench
./install_dependencies.sh
cd ..
pip install -e .

#download models
python examples/sync_models.py

#run banhcmarks
python examples/benchmark_all_models.py

```

### Option 2: Using Your Current Python Version

If you want to use your current Python version:

```bash
# Make the script executable
chmod +x install_dependencies.sh

# Run the script
./install_dependencies.sh
```

> **Note**: This may result in compatibility issues if you're using Python 3.13+. See the [Troubleshooting](#troubleshooting) section for common issues and solutions.

## Installation

### Prerequisites

#### Xcode Command Line Tools

ANEMLL-Bench requires Xcode Command Line Tools to be installed on macOS, as they provide essential development libraries and compilers needed for the build process.

To check if Xcode Command Line Tools are installed:

```bash
xcode-select -p
```

If the command returns a path (e.g., `/Library/Developer/CommandLineTools` or `/Applications/Xcode.app/Contents/Developer`), then the tools are installed.

If not installed, you can install them by running:

```bash
xcode-select --install
```


### Verifying Installation

To verify your installation, run the system info command:

```bash
python -m anemll_bench --system-info
```

This should display information about your system, including whether you have Apple Silicon and Neural Engine available.

### Automatic Benchmarking of All Models

You can easily benchmark all available platform-specific models with a single command:

```bash
# Benchmark all models with default settings (300 iterations)
python examples/benchmark_all_models.py

# Customize the benchmarking process
python examples/benchmark_all_models.py --runs 500 --sequence-length 1 --output my_report.html

# Skip model synchronization and use only local models
python examples/benchmark_all_models.py --no-sync --use-local

# Generate a report without charts
python examples/benchmark_all_models.py --no-charts
```

This will automatically:
1. Download any missing models (unless `--no-sync` and `--use-local` are used)
2. Benchmark each available model for your macOS version 
3. Generate a comprehensive report with comparison metrics


### Checking for Online Model Updates

The package can check Hugging Face for updated model definitions:

```python
from anemll_bench.models import check_and_update_platform_models

# Check for updated model definitions online
check_and_update_platform_models()
```

You can also use the example scripts provided:

```bash
# Standard benchmarking (uses local models)
python examples/load_platform_models.py

# Check for updates online, then benchmark
python examples/load_platform_models.py --check-online

# Benchmark a specific model with online check
python examples/load_platform_models.py --model llama_lm_head --check-online --num-runs 50

# Check and update model definitions from Hugging Face
python examples/check_online_models.py
```

### Automatic Model Synchronization

The easiest way to get all required models is to run the sync script:

```