# ANEMLL-Bench

## Overview
ANEMLL-Bench is a benchmarking tool specifically designed to measure and evaluate the performance of machine learning models on Apple's Neural Engine (ANE). It provides comprehensive metrics including GB/s throughput and TFLOPS calculations to help researchers and developers optimize their models for Apple Silicon.

## Features
- Benchmark models on Apple Neural Engine and compare with CPU/GPU performance
- Measure inference time, memory usage, throughput (GB/s), and computational efficiency (TFLOPS)
- Download and convert models from Hugging Face to CoreML format
- Automatically collect system information (Mac model, CPU details, memory)
- Generate comprehensive HTML reports with visualizations
- Upload and share reports via multiple services (GitHub Gist, JSONBin, Pastebin)
- Easy-to-use API for integrating new models

## Installation

### Setting Up Your Environment

#### Option 1: Using Python venv (Recommended)

```bash
# Create a virtual environment
python -m venv env-anemll-bench

# Activate the virtual environment
# On macOS/Linux:
source env-anemll-bench/bin/activate
# On Windows:
# env-anemll-bench\Scripts\activate

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

## Usage

### Basic Command Line Usage

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
    num_runs=100
)

# Generate a report
benchmark.generate_report("benchmark_report.html")
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
    num_runs=50
)

# Convert and benchmark on Apple Neural Engine
ane_result = benchmark.benchmark_model(
    model=model,
    model_name="Phi-2",
    input_shape=[1, 128, 2560],
    backend="ANE",  # This will trigger automatic CoreML conversion
    num_runs=50
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
python -m examples.profile_coreml --model path/to/your/model.mlmodel --iterations 1000 --compare-cpu

# Profile all CoreML models in a directory (batch mode)
python -m examples.batch_profile --models-dir path/to/models/dir --iterations 100 --compare-cpu
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
    num_iterations=100,
    inputs=inputs  # Optional - will be generated if not provided
)

# Print results
print(f"Inference time: {results['avg_inference_time_ms']:.2f} ms")
print(f"Throughput: {results['throughput_gbps']:.2f} GB/s")
print(f"TFLOPS: {results['tflops']:.4f}")
```

## Documentation
For more detailed documentation, please refer to the [docs](./docs) directory.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
- This project is part of the ANEMLL (Artificial Neural Engine Machine Learning Library) initiative
- Special thanks to Apple for developing the CoreML toolchain 