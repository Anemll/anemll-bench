# ANEMLL-Bench Results

This document presents benchmark results for various machine learning models on different Apple Silicon chips, focusing on Neural Engine (ANE) performance.

## ⚠️ ATTENTION: MORE M3 RESULTS NEEDED! ⚠️

**We now have M3 Max data but still need more M3 series benchmarks!** If you have access to any other M3 chip variant (M3, M3 Pro, or M3 Ultra), please consider running the benchmarks and submitting your results. Your contribution will help complete our cross-generation performance analysis.

*Submit results to: [realanemll@gmail.com](mailto:realanemll@gmail.com) or [open an issue](https://github.com/Anemll/anemll-bench/issues/new)*

## Overview

The cross-chip comparison below reports `llama_lm_head` memory bandwidth. A historical latency column mixed `llama_lm_head_lut6` and full `llama_lm_head` artifacts, so it has been removed rather than presented as a cross-chip comparison.

## Apple Silicon Performance Comparison

The chart below shows performance comparison across Apple Silicon generations for the `llama_lm_head` model:

![Apple Silicon Performance Comparison](./reports/chip_comparison_llama_lm_head.png?v=20260923_m5u_v1)

As shown in the visualization:
- **M4 Series** chips demonstrate approximately 2.3x higher memory bandwidth compared to M1 series
- **M3 Max** shows impressive memory bandwidth (2.2x over M1 series)
- **Base M2** has slightly better bandwidth than M1 series
- The M2 Pro/Max/Ultra and M4 base model show modest improvements, while M3 Max and high-end M4 variants represent significant leaps in performance
- **M5** shows a 1.3x memory-bandwidth improvement over M1 series
- **M5 Max**, **M5 Ultra (36/80/256)**, and **M6 (Mac18,5, 32 GB)** are the three highest reported single-model bandwidth results, at 148.39 GB/s, 150.33 GB/s, and 152.01 GB/s respectively

## Detailed Benchmark Results

### llama_lm_head Model (Standard)

| Chip | Memory Bandwidth (GB/s) | Bandwidth Factor |
|------|------------------------|------------------|
| M1 | 60.87 | 1.1x |
| M1 Pro | 54.90 | 1.0x |
| M1 Max | 54.62 | 1.0x |
| M1 Ultra | 54.72 | 1.0x |
| M2 | 60.45 | 1.1x |
| M2 Pro | 61.93 | 1.1x |
| M2 Max | 62.01 | 1.1x |
| M2 Ultra | 61.68 | 1.1x |
| M3 | 63.10 | 1.2x |
| M3 Max | 120.22 | 2.2x |
| M4 | 64.18 | 1.2x |
| M4 Pro 24GB Mini| 126.36 | 2.3x |
| M4 Max | 118.88 | 2.2x |
| M5 | 70.21 | 1.3x |
| M5 Max | 148.39 | 2.7x |
| M5 Ultra (36C CPU / 80C GPU / 256 GB) | 150.33 | 2.7x |
| M6 (Mac18,5, 32 GB) | 152.01 | 2.8x |

### Verified `llama_lm_head` Single-Model Latency

| Chip | Inference Time (ms) | Model Size (MB) | Provenance |
|------|--------------------:|----------------:|------------|
| M5 Max | 7.08 | 1050.67 | Standalone benchmark report |
| M5 Ultra (36C CPU / 80C GPU / 256 GB) | 6.99 | Not stated | Contributor-supplied result; run count and warmup policy not stated |
| M6 (Mac18,5, 32 GB) | 6.91 | 1050.67 | Supplied benchmark report; run count and warmup policy not stated |

### Key Observations

1. **Neural Engine Scaling**:
   - All M1 variants (Pro, Max, Ultra) show very similar performance, suggesting limited Neural Engine scaling in first-generation Apple Silicon
   - Similar pattern with M2 Pro/Max/Ultra
   - The base M2 chip has slightly better bandwidth than M1 series
   - M3 Max and high-end M4 variants represent the first large bandwidth step in this dataset
   - The base M4 16GB shows modest improvements (~1.2x) over M1 series, significantly underperforming both M3 Max and higher-end M4 variants
   - **M5** reaches 70.21 GB/s, between the base M2/M3 tier and the higher-bandwidth variants
   - **M5 Max**, **M5 Ultra**, and **M6** reach 148.39 GB/s, 150.33 GB/s, and 152.01 GB/s respectively; parallel/combined dual-model values are not used in the cross-chip table

2. **Memory Bandwidth Efficiency**:
   - M3 Max and high-end M4 series show ~2.2-2.3x improvement in memory bandwidth utilization
   - Base M2 shows only a minor improvement (1.1x) over M1 series
   - This indicates that the substantial architectural improvements in the Neural Engine started with the M3 generation
   - Entry-level M4 16GB shows ~1.2x improvement, suggesting memory configuration or thermal constraints may be factors
   - M5 achieves 1.3x memory bandwidth improvement, falling between M2 and M3 Max performance levels
   - M5 Max and M5 Ultra reach 2.7x bandwidth over M1 Pro, while the M6 (Mac18,5, 32 GB) report reaches 2.8x; the M5 Ultra and M6 reports are single reported results rather than the existing multi-run dataset

3. **Latency Comparison Status**:
   - The historical M1–M5 latency values came from a `llama_lm_head_lut6` sample list while the surrounding table claimed the full `llama_lm_head`; they are intentionally not used for cross-chip factors or conclusions
   - The verified M5 Max and M6 full-model latency results are listed above without a relative factor; a controlled matching-model campaign is required for a valid cross-chip latency chart

### ANE Bandwidth vs System Memory Bandwidth

| Chip | ANE BW (GB/s) | System Mem BW (GB/s) | ANE Utilization |
|------|---------------:|--------------------:|----------------:|
| **M6 (32 GB)** | **152** | **170** | **89%** |
| **M5 Ultra (36/80/256)** | **150** | **1200** | **13%** |
| **M5 Max** | **148** | **614** | **24%** |
| M4 Pro | 126 | 273 | 46% |
| M3 Max | 120 | 400 | 30% |
| M4 Max | 119 | 546 | 22% |
| M5 | 70 | 154 | 46% |
| M4 | 64 | 120 | 53% |
| M3 | 63 | 100 | 63% |
| M2 Max | 62 | 400 | 16% |
| M2 Pro | 62 | 200 | 31% |
| M2 Ultra | 62 | 800 | 8% |
| M1 | 61 | 68 | 89% |
| M2 | 60 | 100 | 60% |
| M1 Pro | 55 | 200 | 27% |
| M1 Ultra | 55 | 800 | 7% |
| M1 Max | 55 | 400 | 14% |

The M6 (32 GB) report reaches 152.01 GB/s, or about 89% of its 170 GB/s unified-memory bandwidth. The M5 Ultra (36/80/256) contributor result reaches 150.33 GB/s, or about 13% of its 1.2 TB/s unified-memory bandwidth. Because the reported timing includes compute work, these utilization values are slightly conservative.

The ANE does not scale linearly with system memory bandwidth. Three clear performance tiers emerge:
- **Tier 1 (~120-152 GB/s)**: M6 (Mac18,5), M5 Ultra, M5 Max, M4 Pro, M3 Max, M4 Max — architectural ANE improvements
- **Tier 2 (~60-70 GB/s)**: M5, M4, M3, M2 series, M1 — base ANE performance
- **Tier 3 (~55 GB/s)**: M1 Pro/Max/Ultra — first-generation ANE with no scaling across variants

### Dual Model Benchmark Results (M5 Max)

| Configuration | Inference Time (ms) | Throughput (GB/s) | Model Size (MB) |
|--------------|--------------------:|------------------:|----------------:|
| llama_lm_head (solo) | 7.08 | 148.39 | 1050.67 |
| DeepHermes_lm_head (solo) | 7.07 | 148.57 | 1050.67 |
| llama_lm_head (parallel) | 14.19 | 74.06 | 1050.67 |
| DeepHermes_lm_head (parallel) | 14.21 | 73.94 | 1050.67 |
| Combined (parallel) | 14.21 | 147.62 | 2101.35 |

The M5 Max maintains nearly identical combined throughput (147.62 GB/s) when running two models in parallel compared to individual runs (~147 GB/s), demonstrating efficient bandwidth utilization across both ANE clusters.

### Dual Model Benchmark Results (M5 Ultra, 36C CPU / 80C GPU / 256 GB)

| Configuration | Inference Time (ms) | Estimated Throughput (GB/s) |
|--------------|--------------------:|----------------------------:|
| llama_lm_head (alone) | 6.98 | — |
| DeepHermes_lm_head (alone) | 6.98 | — |
| llama_lm_head (concurrent) | 14.00 | — |
| DeepHermes_lm_head (concurrent) | 12.67 | — |
| Combined (concurrent) | — | 150.07 |

The M5 Ultra contributor result reports 150.07 GB/s combined throughput when both models run concurrently. That is effectively the same as its 150.33 GB/s single-model `llama_lm_head` result, so the dual-model value is documented separately and is not used in the cross-chip single-model chart.

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
