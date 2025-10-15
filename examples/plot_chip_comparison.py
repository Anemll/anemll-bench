#!/usr/bin/env python3
"""
Example script demonstrating the use of ANEMLL-Bench visualization utilities
to create chip comparison charts.
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path to allow running this script directly
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from anemll_bench.utils import plot_chip_comparison, plot_benchmark_results


def main():
    parser = argparse.ArgumentParser(description='ANEMLL-Bench Chip Comparison Visualization')
    parser.add_argument('--save', action='store_true', help='Save the figure to a file')
    parser.add_argument('--output-dir', type=str, default='./reports', help='Directory to save the figure')
    parser.add_argument('--no-show', action='store_true', help='Do not display the figure')
    args = parser.parse_args()

    # Sample data from llama_lm_head model benchmarks
    chips = ['M1', 'M1 Pro', 'M1 Max', 'M1 Ultra', 'M2', 'M2 Max', 'M2 Ultra', 'M3', 'M3 Max', 'M4\n16GB MBP', 'M4 Pro\n24GB Mini', 'M4 Max']
    bandwidth = [60.87, 54.90, 54.62, 54.72, 60.45, 62.01, 61.68, 63.10, 120.22, 64.18, 126.36, 118.88]  # GB/s (llama_lm_head)
    inference = [7.52, 7.45, 7.61, 7.58, 8.67, 6.64, 6.70, 6.85, 3.98, 6.45, 3.85, 3.87]           # ms (llama_lm_head_lut6 for M3 base)
    bandwidth_factor = ['1.1x', '1.0x', '1.0x', '1.0x', '1.1x', '1.1x', '1.1x', '1.1x', '2.2x', '1.2x', '2.3x', '2.2x']
    inference_factor = ['1.0x', '1.0x', '1.0x', '1.0x', '0.9x', '1.1x', '1.1x', '1.1x', '1.9x', '1.2x', '2.0x', '2.0x']

    # Option 1: Use plot_chip_comparison directly
    print("Demonstrating direct use of plot_chip_comparison function...\n")
    
    save_path = None
    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, "chip_comparison_direct.png")
    
    plot_chip_comparison(
        chips=chips,
        bandwidth=bandwidth,
        inference=inference,
        bandwidth_factor=bandwidth_factor,
        inference_factor=inference_factor,
        title="ANEMLL-BENCH: Apple Neural Engine Performance Comparison (llama_lm_head)",
        save_path=save_path,
        show_plot=not args.no_show
    )
    
    # Option 2: Use plot_benchmark_results with a data dictionary
    print("Demonstrating use of plot_benchmark_results function with benchmark data...\n")
    
    # Create a benchmark data dictionary
    benchmark_data = {
        'chips': chips,
        'bandwidth': bandwidth,
        'inference': inference,
        'bandwidth_factor': bandwidth_factor,
        'inference_factor': inference_factor,
    }
    
    # Plot using the higher-level function
    plot_benchmark_results(
        benchmark_data=benchmark_data,
        model_name="llama_lm_head",
        save_dir=args.output_dir if args.save else None,
        show_plot=not args.no_show
    )
    
    if args.save:
        print(f"Figures saved to {args.output_dir}")


if __name__ == "__main__":
    main() 