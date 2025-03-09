# ANEMLL-Bench

## ⚠️ Attention: macOS 15.x is required! ⚠️

This alpha release requires macOS 15. We plan to update support for older OS versions in the next update.

## Overview
ANEMLL-Bench  (pronounced like "animal-bench") is a benchmarking tool specifically designed to measure and evaluate the performance of machine learning models on Apple's Neural Engine (ANE). It provides comprehensive metrics including inference time and memory bandwidth utilization (GB/s) to help researchers and developers optimize their models for Apple Silicon.

This alpha release requires macOS 15. We plan to update support for older OS versions in the next update. Currently, only Memory bandwidth (GB/s) is benchmarked in this release.

ANEMLL-Bench is part on ANEMLL Open Source Project [anemll.com](https://anemll.com)

## 📊 [View Benchmark Results](./Results.MD) 📊

[![Apple Silicon Performance Comparison](./reports/chip_comparison_llama_lm_head.png?v=20250309_v6)](./Results.MD)

Check out our latest [benchmark results](./Results.MD) comparing performance across different Apple Silicon chips (M1, M2, M3, M4 series).

<div align="center">
  <h2>📊 Help Us Build a Comprehensive Benchmark Database! 📊</h2>
  <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #007bff; margin: 20px 0; color: #333333;">
    <h3 style="color: #000000;">🚨 PLEASE SUBMIT YOUR BENCHMARK RESULTS FOR: 🚨</h3>
    <table align="center" style="color: #333333;">
      <tr>
        <td align="center"><strong style="color: #000000;">M1 Series</strong></td>
        <td align="center"><strong style="color: #000000;">M2 Series</strong></td>
        <td align="center"><strong style="color: #000000;">M3 Series</strong></td>
        <td align="center"><strong style="color: #000000;">M4 Series</strong></td>
      </tr>
      <tr>
        <td>
          ✓ M1<br>
          ✓ M1 PRO<br>
          ✓ M1 MAX ✅<br>
          ✓ M1 ULTRA ✅
        </td>
        <td>
          ✓ M2<br>
          ✓ M2 PRO<br>
          ✓ M2 MAX ✅<br>
          ✓ M2 ULTRA ✅
        </td>
        <td>
          ✓ M3<br>
          ✓ M3 PRO<br>
          ✓ M3 MAX ✅<br>
          ✓ M3 ULTRA
        </td>
        <td>
          ✓ M4 ✅<br>
          ✓ M4 PRO ✅<br>
          ✓ M4 MAX ✅
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

This will automatically download and prepare all the optimized models for your specific macOS version. The models are stored in `~/.cache/anemll-bench/`