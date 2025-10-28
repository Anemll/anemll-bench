# ANEMLL-Bench Results

This document presents benchmark results for various machine learning models on different Apple Silicon chips, focusing on Neural Engine (ANE) performance.

## ⚠️ ATTENTION: MORE M3 RESULTS NEEDED! ⚠️

**We now have M3 Max data but still need more M3 series benchmarks!** If you have access to any other M3 chip variant (M3, M3 Pro, or M3 Ultra), please consider running the benchmarks and submitting your results. Your contribution will help complete our cross-generation performance analysis.

*Submit results to: [realanemll@gmail.com](mailto:realanemll@gmail.com) or [open an issue](https://github.com/Anemll/anemll-bench/issues/new)*

## Overview

ANEMLL-Bench measures two primary metrics:
1. **Memory Bandwidth (GB/s)**: How Apple Chip Generation utilizes memory bandwidth (higher is better)
2. **Inference Time (ms)**: How quickly the model produces results (lower is better)

Higher memory bandwidth and lower inference time indicate better performance.

## Apple Silicon Performance Comparison

The chart below shows performance comparison across Apple Silicon generations for the `llama_lm_head` model:

![Apple Silicon Performance Comparison](./reports/chip_comparison_llama_lm_head.png?v=20251028_v5)

As shown in the visualization:
- **M4 Series** chips demonstrate approximately 2.3x higher memory bandwidth compared to M1 series
- **M3 Max** shows impressive memory bandwidth (2.2x over M1 series) and excellent inference performance (1.9x faster than M1)
- **Base M2** has slightly better bandwidth than M1 series but slightly worse inference time
- The M2 Max/Ultra and M4 base model show modest improvements, while M3 Max and high-end M4 variants represent significant leaps in performance
- **M5** shows solid performance with good memory bandwidth improvement (1.3x) and faster inference times (1.2x), positioning it competitively between M2 and M3 Max performance levels

## Detailed Benchmark Results

### llama_lm_head Model (Standard)

| Chip | Memory Bandwidth (GB/s) | Inference Time (ms) | Bandwidth Factor | Inference Factor |
|------|------------------------|---------------------|------------------|------------------|
| M1 | 60.87 | 7.52 | 1.1x | 1.0x |
| M1 Pro | 54.90 | 7.45 | 1.0x | 1.0x |
| M1 Max | 54.62 | 7.61 | 1.0x | 1.0x |
| M1 Ultra | 54.72 | 7.58 | 1.0x | 1.0x |
| M2 | 60.45 | 8.67 | 1.1x | 0.9x |
| M2 Max | 62.01 | 6.64 | 1.1x | 1.1x |
| M2 Ultra | 61.68 | 6.70 | 1.1x | 1.1x |
| M3 | 63.10 | 16.65 | 1.2x | 0.5x |
| M3 Max | 120.22 | 3.98 | 2.2x | 1.9x |
| M4 16GB MBP  | 64.18 | 6.45 | 1.2x | 1.2x |
| M4 Pro 24GB Mini| 126.36 | 3.85 | 2.3x | 2.0x |
| M4 Max | 118.88 | 3.87 | 2.2x | 2.0x |
| M5 | 70.21 | 6.10 | 1.3x | 1.2x |

### Key Observations

1. **Neural Engine Scaling**:
   - All M1 variants (Pro, Max, Ultra) show very similar performance, suggesting limited Neural Engine scaling in first-generation Apple Silicon
   - Similar pattern with M2 Ultra vs M2 Max
   - The base M2 chip has slightly better bandwidth than M1 series, but surprisingly worse inference time
   - M3 Max shows excellent memory bandwidth utilization and very good inference times, competitive with M4 series
   - M4 series demonstrates slightly better performance across both metrics compared to M3 Max
   - The base M4 16GB shows modest improvements (~1.2x) over M1 series, significantly underperforming both M3 Max and higher-end M4 variants
   - **M5 shows solid performance**: It achieves 1.3x memory bandwidth improvement over M1 series (70.21 GB/s) and 1.2x faster inference times (6.10 ms), positioning it competitively between M2 and M3 Max performance levels

2. **Memory Bandwidth Efficiency**:
   - M3 Max and high-end M4 series show ~2.2-2.3x improvement in memory bandwidth utilization
   - Base M2 shows only a minor improvement (1.1x) over M1 series
   - This indicates that the substantial architectural improvements in the Neural Engine started with the M3 generation
   - Entry-level M4 16GB shows ~1.2x improvement, suggesting memory configuration or thermal constraints may be factors
   - M5 achieves 1.3x memory bandwidth improvement, falling between M2 and M3 Max performance levels

3. **Inference Time Improvements**:
   - Base M2 actually performs slightly worse (0.9x) than M1 series for inference time
   - M2 Max/Ultra models show modest improvements (1.1x) over M1 series
   - M3 Max shows impressive inference performance (3.98 ms), nearly as fast as the best M4 chips
   - High-end M4 chips are only marginally faster (3.85-3.87 ms) than the M3 Max
   - This suggests the major architectural leap for Neural Engine inference performance occurred with the M3 generation
   - M4 represents a more incremental improvement over M3 Max for inference workloads
   - **M5 shows solid inference performance** (6.10 ms), achieving 1.2x improvement over M1 series, which aligns well with its memory bandwidth improvements and positions it competitively in the performance landscape

## Running Your Own Benchmarks

To reproduce these results or run benchmarks on your own device:

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Download optimized models for your macOS version
python examples/sync_models.py

# Run benchmarks on all available models
python examples/benchmark_all_models.py

# Generate visualization of results
python examples/plot_chip_comparison.py --save
```

## Contributing Results

We're building a comprehensive benchmark database across all Apple Silicon variants. Please consider submitting your benchmark results by:

1. Running the benchmarks using the instructions above
2. Opening an issue on our GitHub repository with your results
3. Or emailing your results to realanemll@gmail.com

When submitting results, please include:
- Your exact device model (e.g., "MacBook Pro 14" 2023, M3 Pro 12-core CPU, 18-core GPU")
- macOS version
- Any cooling modifications or environmental factors
- The complete benchmark report

## Analyzing Your Results

When analyzing your benchmark results, consider:

1. **Relative Performance**: How does your chip compare to others in the same family?
2. **Scaling Efficiency**: If you have a Pro/Max/Ultra variant, how efficiently does it scale?
3. **Model-Specific Performance**: Different model architectures may perform differently on the same hardware

## Future Work

We plan to expand our benchmarks to include:
- More diverse model architectures
- Power efficiency measurements (performance per watt)
- Sustained performance under thermal constraints
- Newer versions of CoreML and PyTorch

## Acknowledgements

Thanks to all contributors who have submitted benchmark results and helped improve ANEMLL-Bench. 
