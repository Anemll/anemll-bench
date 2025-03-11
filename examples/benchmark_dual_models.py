#!/usr/bin/env python
"""
Example script to benchmark two models simultaneously to measure bandwidth utilization.

This script demonstrates how to:
1. Load two different CoreML models
2. Benchmark them individually as a baseline
3. Benchmark them running simultaneously in separate threads
4. Compare the results to identify potential bandwidth improvements
"""

import os
import sys
import argparse
import time
import subprocess
import platform
from pathlib import Path

# Add parent directory to path to import anemll_bench
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anemll_bench import Benchmark
import coremltools as ct


def extract_input_shape_from_model(model):
    """Extract the required input shape from a CoreML model"""
    try:
        # Get model spec
        spec = model.get_spec().description.input
        if spec and len(spec) > 0:
            # Extract shape from first input
            input_tensor = spec[0]
            shape = [dim for dim in input_tensor.type.multiArrayType.shape]
            return shape
    except Exception as e:
        print(f"Error extracting input shape: {e}")
    
    # Return a default shape if we couldn't extract it
    return [1, 1, 4096]


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Benchmark two models simultaneously to measure bandwidth utilization")
    parser.add_argument("--runs", type=int, default=300, help="Number of benchmark runs per model (default: 300)")
    parser.add_argument("--backend", type=str, default="ANE", choices=["CPU", "GPU", "ANE", "ALL"], 
                       help="Backend to use for benchmarking (default: ANE)")
    parser.add_argument("--report", type=str, default=None, help="Generate HTML report with this filename")
    parser.add_argument("--no-browser", action="store_true", help="Don't open the report in a browser")
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = Benchmark()
    
    # Specify the exact models to use
    model1_name = "llama_lm_head"
    model2_name = "DeepHermes_lm_head"
    
    print(f"Using models: {model1_name} and {model2_name}")
    
    # Load the models
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "anemll-bench", "models")
    
    model1_path = os.path.join(cache_dir, f"{model1_name}.mlpackage")
    model2_path = os.path.join(cache_dir, f"{model2_name}.mlpackage")
    
    # Map string backend to CoreML compute units
    compute_units_map = {
        "CPU": ct.ComputeUnit.CPU_ONLY,
        "GPU": ct.ComputeUnit.CPU_AND_GPU,
        "ANE": ct.ComputeUnit.CPU_AND_NE,
        "ALL": ct.ComputeUnit.ALL
    }
    compute_unit = compute_units_map.get(args.backend, ct.ComputeUnit.CPU_AND_NE)
    
    # Load models directly using CoreML
    print(f"\nLoading models...")
    
    try:
        # Load both models
        model1 = ct.models.model.MLModel(model1_path, compute_units=compute_unit)
        model2 = ct.models.model.MLModel(model2_path, compute_units=compute_unit)
        
        # Extract input shapes directly from the models
        model1_input_shape = extract_input_shape_from_model(model1) 
        model2_input_shape = extract_input_shape_from_model(model2)
        
        print(f"Model 1 input shape: {model1_input_shape}")
        print(f"Model 2 input shape: {model2_input_shape}")
        
        # Run dual benchmark
        if model1 and model2:
            # Run dual benchmark
            dual_results = benchmark.benchmark_dual_models(
                model1=model1,
                model1_name=model1_name,
                model1_input_shape=model1_input_shape,
                model2=model2,
                model2_name=model2_name,
                model2_input_shape=model2_input_shape,
                backend=args.backend,
                num_runs=args.runs
            )
            
            # Get the results to calculate averages
            if isinstance(dual_results, dict) and 'parallel_results' in dual_results:
                parallel_results = dual_results['parallel_results']
                if len(parallel_results) >= 3:  # We should have two individual model results plus the combined result
                    # The third result should be the combined performance
                    combined_result = parallel_results[2]
                    print(f"Combined performance: {combined_result.inference_time_ms:.2f} ms, {combined_result.throughput_gb_s:.2f} GB/s")
            
            # Get the report directory
            reports_dir = os.path.join(os.path.expanduser("~"), ".cache", "anemll-bench", "reports")
            os.makedirs(reports_dir, exist_ok=True)
            
            # Generate report if requested
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_filename = args.report or f"dual_benchmark_report_{timestamp}.html"
            report_path = os.path.join(reports_dir, report_filename)
            
            benchmark.generate_report(output_path=report_path, include_charts=True, auto_open=False)  # Don't auto-open in coremltools
            
            print(f"\nReport saved to: {report_path}")
            
            # Open the report in the default web browser if not disabled
            if not args.no_browser:
                try:
                    print(f"Opening report in web browser...")
                    # Use system 'open' command for macOS, which works more reliably
                    if platform.system() == 'Darwin':  # macOS
                        subprocess.call(['open', report_path])
                    else:  # Try the webbrowser module for other platforms
                        import webbrowser
                        webbrowser.open(f"file://{report_path}")
                except Exception as e:
                    print(f"Error opening report in browser: {e}")
        else:
            print("Error loading models.")
            return 1
            
    except Exception as e:
        print(f"Error loading or benchmarking models: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 