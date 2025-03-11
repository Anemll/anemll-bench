# Dual Model Benchmarking

This feature allows you to benchmark two models simultaneously to measure how they perform when running in parallel on the Neural Engine.

## Overview

When running multiple ML models concurrently, resource contention can affect performance. The dual model benchmarking feature helps you:

- Measure individual performance of each model running alone
- Measure performance when both models run simultaneously
- Analyze efficiency and resource utilization
- Compare throughput and latency in isolation vs. parallel execution

## Requirements

- macOS with Apple Neural Engine (M1/M2/M3 series)
- Python 3.8+
- anemll-bench package

## Updating from Previous Versions

If you're updating from a previous version of the tool, follow these steps to ensure compatibility:

1. **Use the automated update script** (recommended):
   ```bash
   python examples/update_dual_benchmark.py
   ```
   This script will:
   - Update the meta.yalm file with the latest model information
   - Check if required models are available
   - Download any missing models automatically

   If you want to force re-downloading all models:
   ```bash
   python examples/update_dual_benchmark.py --force-resync
   ```

2. **Manual update**: If you prefer to update manually:
   ```bash
   # Update the meta.yalm file and download any missing/new models (recommended)
   python examples/sync_models.py --update
   
   # For faster downloads, use parallel mode
   python examples/sync_models.py --update --parallel
   
   # Or use these individual steps:
   # Update the meta.yalm file
   python examples/sync_models.py --force
   
   # Check which models are available for your platform
   python examples/list_platform_models.py
   
   # Download models
   python examples/sync_models.py
   ```

3. **Check cached models**: If you encounter any errors, check your model cache:
   ```bash
   python examples/manage_cache.py --status
   ```
   
   You may need to clear the cache if there are corrupted models:
   ```bash
   python examples/manage_cache.py --clear-models
   ```

## Basic Usage

Run a dual model benchmark with default settings:

```bash
python examples/benchmark_dual_models.py
```

This will benchmark the default models (`llama_lm_head` and `DeepHermes_lm_head`) with 300 runs each.

## Advanced Options

Customize the benchmark with these options:

```bash
python examples/benchmark_dual_models.py --runs 100 
```

## Understanding the Results

The benchmark will output several key metrics:

1. **Individual Performance**:
   - Inference time (ms) for each model running alone
   - Throughput (GB/s) for each model running alone

2. **Parallel Performance**:
   - Inference time (ms) for each model running simultaneously
   - Throughput (GB/s) for each model running simultaneously
   - Combined metrics showing overall system performance

3. **Combined Analysis**:
   - Total parallel execution time
   - Combined throughput
   - Sum of individual throughputs
   - Bandwidth utilization factor
   - Efficiency percentage

4. **HTML Report**:
   An interactive HTML report will be generated at `~/.cache/anemll-bench/reports/`

## Interpreting Efficiency

The efficiency percentage indicates how well the models share resources:

- **~100%**: Near-perfect resource sharing, minimal contention
- **~50%**: Significant resource contention, models competing for bandwidth
- **<50%**: Severe contention, consider running models sequentially

## Example Output

```
=== Dual Model Benchmarking Results ===

Individual Performance:
  - llama_lm_head: 19.35 ms, 54.25 GB/s
  - DeepHermes_lm_head: 19.42 ms, 54.07 GB/s

Parallel Performance:
  - llama_lm_head: 38.12 ms, 27.56 GB/s
  - DeepHermes_lm_head: 38.25 ms, 27.45 GB/s
  - Combined: 38.25 ms, 54.85 GB/s (total throughput)

Combined Analysis:
  - Total Parallel Execution Time: 0.77 seconds
  - Combined Throughput: 54.85 GB/s
  - Sum of Individual Throughputs: 108.32 GB/s
  - Bandwidth Utilization Factor: 0.51x
  - Efficiency: 50.64%
```

## Troubleshooting

If you encounter issues:

1. **Models not found**:
   - Run `python examples/sync_models.py --force` to update the meta.yalm file and download models

2. **Performance issues**:
   - Ensure no other intensive applications are running
   - Try rebooting your system to clear memory

3. **Report generation fails**:
   - Check disk space in your home directory
   - Ensure you have write permissions to `~/.cache/anemll-bench/reports/` 