# ANEMLL-Bench

## Overview
ANEMLL-Bench is a benchmarking tool specifically designed to measure and evaluate the performance of machine learning models on Apple's Neural Engine (ANE). It provides comprehensive metrics including inference time and memory bandwidth utilization (GB/s) to help researchers and developers optimize their models for Apple Silicon.

This alpha release requires macOS 15. We plan to update support for older OS versions in the next update. Currently, only Memory bandwidth (GB/s) is benchmarked in this release.

## Quick Start

To get started quickly with platform-specific optimized models:

```bash
# Download and install the package
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

## Features
- Benchmark models on Apple Neural Engine and compare with CPU/GPU performance
- Measure inference time and memory bandwidth utilization (GB/s)
- Download and convert models from Hugging Face to CoreML format
- Automatically collect system information (Mac model, CPU details, memory)
- Generate comprehensive HTML reports with visualizations
- Upload and share reports via multiple services (GitHub Gist, JSONBin, Pastebin)
- Easy-to-use API for integrating new models
- Automatic downloading of platform-specific optimized models (macOS 15.x+)
- Robust model size detection for accurate throughput calculation

## Installation

### Prerequisites

### Setting Up Your Environment

#### Option 1: Using Python venv (Recommended)

```bash
# Create a virtual environment
python -m venv env-anemll-bench

# Activate the virtual environment
# On macOS
source env-anemll-bench/bin/activate


# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

#### Option 2: Using Conda

```bash
# Create a conda environment
conda create -n anemll-bench python=3.9

# Activate the environment
conda activate anemll-bench

# Install PyTorch
conda install pytorch -c pytorch

# Install remaining dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

#### Note for Apple Silicon Macs

For best performance on Apple Silicon (M1/M2/M3), ensure you're using Python 3.9+ and the ARM64 version of PyTorch:

```bash
# For Apple Silicon Macs
pip install torch==2.5.0 --extra-index-url https://download.pytorch.org/whl/nightly/cpu

# Then install other dependencies
pip install -r requirements.txt
```

### Installing from PyPI

Alternatively, you can install directly from PyPI:

```bash
pip install anemll-bench
```

### Verifying Installation

To verify your installation, run the system info command:

```bash
python -m anemll_bench --system-info
```

This should display information about your system, including whether you have Apple Silicon and Neural Engine available.


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

Follow the prompts to complete the installation.



## Usage

### Command Line Interface (CLI)

ANEMLL-Bench provides several command-line scripts to help you benchmark models quickly.

#### Benchmarking All Platform Models

```bash
# Benchmark all models with default settings (300 iterations)
python examples/benchmark_all_models.py

# Customize your benchmark
python examples/benchmark_all_models.py --runs 500 --sequence-length 256 --output my_report.html

# Benchmark a specific model
python examples/benchmark_all_models.py --model llama_lm_head

# Skip model synchronization (use only local models)
python examples/benchmark_all_models.py --no-sync --use-local

# Generate a report with comparison metrics
python examples/benchmark_all_models.py --no-charts
```

To see all CLI options:

```bash
python examples/benchmark_all_models.py --help
```

#### Benchmarking PyTorch Models

Run a quick benchmark with a Hugging Face model:

```bash
python -m examples.basic_benchmark --model microsoft/phi-2 --runs 50
```

Additional options:
```bash
python -m examples.basic_benchmark --model microsoft/phi-2 --sequence-length 256 --batch-size 1 --runs 50 --output my_report.html --upload --upload-service jsonbin
```

### Benchmarking with Configuration File

Run benchmarks for multiple models defined in a configuration file:

```bash
python -m examples.basic_benchmark --config examples/benchmark_config.json
```

### Python API for Basic Benchmarking

```python
from anemll_bench import Benchmark

# Initialize benchmark
benchmark = Benchmark()

# Benchmark a model
result = benchmark.benchmark_model(
    model=your_model,  # PyTorch model or CoreML model
    model_name="model_name",
    input_shape=[1, 3, 224, 224],  # Example shape for an image model
    backend="ANE",  # Use "ANE", "CPU", or "GPU"
    num_runs=300     # Default is 300 iterations
)

# Generate a report
benchmark.generate_report("benchmark_report.html")
```

### Python API Usage

The Python API gives you more control over the benchmarking process:

```python
from anemll_bench import benchmark

# Benchmark all models
results = benchmark.benchmark_all_platform_models(
    num_runs=300,                  # Number of iterations per model (default: 300)
    batch_size=1,                  # Batch size for inputs
    sequence_length=None,          # Sequence length (auto-detected from model)
    sync_first=True,               # Download missing models first
    include_charts=True,           # Include charts in the report
    output_path="my_report.html",  # Custom report path
    use_local_if_exists=True       # Use local models when available
)

# Benchmark a specific model
result = benchmark.benchmark_platform_model(
    model_name="llama_lm_head",
    num_runs=300,                  # Default: 300 iterations
    check_online=False,            # Skip checking for newer versions online
    force_redownload=False,        # Don't force re-downloading the model
    use_local_if_exists=True       # Use local model if it exists
)

# Create a benchmark report
benchmark.generate_platform_report(
    results, 
    output_path="reports/benchmark_report.html",
    include_charts=True
)
```

### Downloading and Benchmarking Models from Hugging Face

```python
from anemll_bench import Benchmark
from anemll_bench.models.model_loader import download_from_hf

# Download a model from Hugging Face
model = download_from_hf("microsoft/phi-2")

# Benchmark the model
benchmark = Benchmark()
result = benchmark.benchmark_model(
    model=model,
    model_name="Phi-2",
    input_shape=[1, 128, 2560],  # Adjust based on model architecture
    backend="CPU",
    num_runs=300
)

# Convert and benchmark on Apple Neural Engine
ane_result = benchmark.benchmark_model(
    model=model,
    model_name="Phi-2",
    input_shape=[1, 128, 2560],
    backend="ANE",  # This will trigger automatic CoreML conversion
    num_runs=300
)

# Generate and upload a report
report_url = benchmark.generate_report(
    output_path="reports/phi2_benchmark.html", 
    upload=True,
    upload_service="jsonbin"  # Options: "jsonbin", "gist", "pastebin"
)

print(f"Report uploaded to: {report_url}")
```

### Using Configuration File

You can also define models and benchmark settings in a JSON configuration file:

```python
from anemll_bench import Benchmark

# Initialize with config
benchmark = Benchmark(config_path="benchmark_config.json")

# Run all benchmarks defined in config
results = benchmark.run()

# Generate report
benchmark.generate_report()
```

Example config.json:
```json
{
  "models": [
    {
      "name": "Phi-2",
      "id": "microsoft/phi-2",
      "type": "pytorch",
      "input_shape": [1, 128, 2560],
      "backends": ["CPU", "ANE"],
      "num_runs": 50
    },
    {
      "name": "DistilBERT",
      "id": "distilbert-base-uncased",
      "type": "pytorch",
      "input_shape": [1, 128, 768],
      "backends": ["CPU", "ANE"],
      "num_runs": 50
    }
  ],
  "output": {
    "report_path": "reports/multi_model_benchmark.html",
    "upload": true,
    "upload_service": "jsonbin"
  }
}
```

## Report Sharing

Benchmark reports can be shared in several ways:

1. **Automatic Uploading**: Use the `--upload` flag to automatically upload reports to JSONBin, GitHub Gist, or Pastebin
2. **GitHub Pages**: Host your reports on GitHub Pages
3. **Direct Sharing**: Send the HTML file directly to colleagues

### Setting up Upload Services

To use the report uploading feature, you need to set environment variables for the respective services:

```bash
# For GitHub Gist
export GITHUB_TOKEN=your_github_personal_access_token

# For JSONBin
export JSONBIN_API_KEY=your_jsonbin_api_key

# For Pastebin
export PASTEBIN_API_KEY=your_pastebin_api_key
```

## Working with Your Own CoreML Models

If you have existing CoreML models (.mlmodel or .mlmodelc files), you can profile their performance directly:

```bash
# Profile a single CoreML model
python -m examples.profile_coreml --model path/to/your/model.mlmodel --iterations 300 --compare-cpu

# Profile all models in a directory
python -m examples.batch_profile --models-dir path/to/models/dir --iterations 300 --compare-cpu
```

### Direct CoreML API

You can also use the CoreML-specific functions from your Python code:

```python
from anemll_bench import Benchmark

# Initialize benchmark
benchmark = Benchmark()

# Benchmark a CoreML model file directly
result = benchmark.benchmark_coreml_file(
    model_path="path/to/your/model.mlmodel",
    num_runs=100,
    batch_size=1,
    sequence_length=512,  # For text models
    compute_unit="CPU_AND_NE"  # Uses both CPU and Neural Engine
)

# Also benchmark on CPU only for comparison
cpu_result = benchmark.benchmark_coreml_file(
    model_path="path/to/your/model.mlmodel",
    num_runs=100,
    compute_unit="CPU_ONLY"
)

# Calculate speedup
speedup = cpu_result.inference_time_ms / result.inference_time_ms
print(f"ANE Speedup: {speedup:.2f}x")

# Generate and upload report comparing both results
benchmark.generate_report(
    output_path="model_benchmark.html",
    upload=True,
    upload_service="jsonbin"
)
```

### Advanced CoreML Adapter

For more control over the CoreML benchmarking process, you can use the low-level functions:

```python
from anemll_bench.models.coreml_adapter import (
    load_coreml_model, 
    profile_coreml_model, 
    prepare_inputs, 
    get_model_size
)

# Load a CoreML model with specific compute unit
model = load_coreml_model("path/to/model.mlmodel", compute_unit="CPU_AND_NE")

# Create custom inputs (or let the system generate them)
inputs = prepare_inputs(model, batch_size=1, sequence_length=256)

# Profile the model
results = profile_coreml_model(
    model=model, 
    num_iterations=1000,
    inputs=inputs  # Optional - will be generated if not provided
)

# Print results
print(f"Inference time: {results['avg_inference_time_ms']:.2f} ms")
print(f"Throughput: {results['throughput_gbps']:.2f} GB/s")
print(f"TFLOPS: {results['tflops']:.4f}")
```

## Platform-Specific Model Support

ANEMLL-Bench now includes support for platform-specific optimized models. For macOS 15.x and higher, the following models are available:

- `llama_lm_head` - Optimized LLaMA model in mlmodelc format
- `llama_lm_head_lut6` - Optimized LLaMA model with LUT6 quantization in mlpackage format

These models are automatically downloaded from the Hugging Face repository when needed. Here's how to use them:

```python
from anemll_bench import Benchmark
from anemll_bench.models import list_available_platform_models

# List available platform-specific models
list_available_platform_models()

# Create a benchmark instance
benchmark = Benchmark()

# Benchmark a platform-specific model
result = benchmark.benchmark_platform_model(
    model_name="llama_lm_head",
    num_runs=100
)

# Generate a benchmark report
benchmark.generate_report(output_path="benchmark_report.html")
```

### Automatic Benchmarking of All Models

You can easily benchmark all available platform-specific models with a single command:

```bash
# Benchmark all models with default settings (300 iterations)
python examples/benchmark_all_models.py

# Customize the benchmarking process
python examples/benchmark_all_models.py --runs 500 --sequence-length 256 --output my_report.html

# Skip model synchronization and use only local models
python examples/benchmark_all_models.py --no-sync --use-local

# Generate a report without charts
python examples/benchmark_all_models.py --no-charts
```

This will automatically:
1. Download any missing models (unless `--no-sync` and `--use-local` are used)
2. Benchmark each available model for your macOS version 
3. Generate a comprehensive report with comparison metrics

You can also use the API to benchmark all models programmatically:

```python
from anemll_bench import Benchmark

benchmark = Benchmark()

# Benchmark all available models and generate a report
results = benchmark.benchmark_all_platform_models(
    num_runs=300,                  # Number of iterations per model (default: 300)
    batch_size=1,                  # Batch size for inputs
    sequence_length=None,          # Sequence length (auto-detected from model)
    sync_first=True,               # Download missing models first
    include_charts=True,           # Include charts in the report
    output_path="my_report.html",  # Custom report path
    use_local_if_exists=True       # Use local models when available
)

print(f"Benchmarked {len(results)} models")
```

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

```bash
# Sync all platform models for your macOS version
python examples/sync_models.py
```

This single command will:
1. Download the latest model definitions from Hugging Face
2. Identify which models are available for your macOS version
3. Download and unzip any missing models
4. Skip models that are already in your cache

After running this command, all optimized models will be ready to use without additional setup.

Additional sync options:

```bash
# Force update of meta.yalm before syncing
python examples/sync_models.py --force

# Quiet mode (less output)
python examples/sync_models.py -q
```

You can also synchronize models programmatically:

```python
from anemll_bench.models import sync_platform_models

# Sync all platform models (download what's missing)
results = sync_platform_models()

# Force update of meta.yalm before syncing
results = sync_platform_models(force_update=True)

print(f"Downloaded {results['models_downloaded']} models")
```

For advanced users, the cache management tool provides additional options:

```bash
# Sync all platform models
python examples/manage_cache.py sync

# Force meta.yalm update before syncing
python examples/manage_cache.py sync --force

# Output results in JSON format
python examples/manage_cache.py sync --json
```

### Managing the Model Cache

All downloaded models and metadata are stored in `~/.cache/anemll-bench/`. The cache can be managed using the provided utility:

```bash
# Display cache information
python examples/manage_cache.py info

# Display cache information in JSON format
python examples/manage_cache.py info --json

# Clear all models from the cache
python examples/manage_cache.py clear

# Clear a specific model from the cache
python examples/manage_cache.py clear --model llama_lm_head

# Clear the entire cache including metadata
python examples/manage_cache.py clear --all

# Update model definitions from Hugging Face
python examples/manage_cache.py update
```

You can also manage the cache programmatically:

```python
from anemll_bench.models import get_cache_info, clear_cache, CACHE_DIR

# Get information about the cache
cache_info = get_cache_info()
print(f"Cache directory: {CACHE_DIR}")
print(f"Total cache size: {cache_info['total_size_mb']:.2f} MB")

# Clear specific models
clear_cache(model_name="llama_lm_head")

# Clear the entire cache
clear_cache(include_meta=True)
```

## Understanding Performance Metrics

ANEMLL-Bench provides several key performance metrics to help you evaluate your models:

### Inference Time

The time it takes to perform a single forward pass of the model, measured in milliseconds (ms). This is calculated by averaging the time across multiple iterations (default: 300) to get a stable measurement.

### Memory Bandwidth Utilization (GB/s)

This metric measures how efficiently your model uses the available memory bandwidth. It is calculated by:

```
Throughput (GB/s) = Model Size (GB) / Inference Time (seconds)
```

The throughput calculation uses the actual model weights size to provide a more accurate representation of memory bandwidth utilization, especially on the Apple Neural Engine (ANE).

### TFLOPS Calculation (Currently Disabled)

The TFLOPS metric (Tera Floating Point Operations per Second) is temporarily disabled in reports as we work on implementing more accurate calculation methods for various model architectures. Future versions will re-enable this metric with improved precision.

### Model Size Detection

ANEMLL-Bench automatically detects model size by examining the weight files in both `.mlmodelc` and `.mlpackage` formats. This size is used when calculating memory bandwidth utilization.

## Documentation
For more detailed documentation, please refer to the [docs](./docs) directory.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
- This project is part of the ANEMLL (Artificial Neural Engine Machine Learning Library) initiative
- Special thanks to Apple for developing the CoreML toolchain