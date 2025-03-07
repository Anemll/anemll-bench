#!/usr/bin/env python3
"""
Profile script for CoreML models using ANEMLL-Bench
Similar to profile_split.py but using the ANEMLL-Bench package
"""

import os
import sys
import argparse
import json
import time
import coremltools as ct
import numpy as np
import webbrowser
from pathlib import Path

# Add parent directory to path for development imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from anemll_bench import Benchmark
from anemll_bench.utils.system_info import get_system_info
from anemll_bench.models.coreml_adapter import (
    load_coreml_model, 
    profile_coreml_model, 
    prepare_inputs, 
    get_model_size,
    benchmark_coreml_model_file,
    get_model_metadata
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Profile CoreML models on Apple Neural Engine')
    
    # Model options
    parser.add_argument('--model', type=str, required=True,
                        help='Path to CoreML model (.mlmodel or .mlmodelc)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for profiling')
    parser.add_argument('--sequence-length', type=int, default=512,
                        help='Sequence length for text models')
    parser.add_argument('--hidden-size', type=int, default=4096,
                        help='Hidden size for text models')
    
    # Benchmark options
    parser.add_argument('--iterations', type=int, default=1000,
                        help='Number of iterations for profiling')
    parser.add_argument('--compute-units', type=str, default='CPU_AND_NE',
                        choices=['CPU_AND_NE', 'CPU_ONLY', 'ALL'],
                        help='Compute units to use for inference')
    
    # Output options
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save benchmark results (JSON)')
    parser.add_argument('--report', type=str, default='profile_report.html',
                        help='Path to save HTML report')
    parser.add_argument('--include-charts', action='store_true',
                        help='Include performance charts in HTML report (disabled by default)')
    parser.add_argument('--upload', action='store_true',
                        help='Upload report to sharing service')
    parser.add_argument('--upload-service', type=str, default='jsonbin',
                        choices=['gist', 'pastebin', 'jsonbin'],
                        help='Service to upload report to')
    
    # Extra options
    parser.add_argument('--compare-cpu', action='store_true',
                        help='Compare with CPU-only performance (disabled by default)')
    parser.add_argument('--tflops', type=float, default=None,
                        help='Specify the total number of trillion floating point operations (TFLOPs) per iteration (not TFLOPS rate)')
    
    args = parser.parse_args()
    
    return args


def print_model_info(model_path):
    """Print basic information about the model"""
    size_bytes = get_model_size(model_path)
    size_mb = size_bytes / (1024 * 1024)
    
    # Get weights-only size
    weights_bytes = get_model_size(model_path, weights_only=True)
    weights_mb = weights_bytes / (1024 * 1024)
    
    print(f"\n=== Model Information ===")
    print(f"Path: {model_path}")
    print(f"Total Size: {size_mb:.2f} MB ({size_bytes:,} bytes)")
    print(f"Weights Size: {weights_mb:.2f} MB ({weights_bytes:,} bytes)")
    print(f"Weights Percentage: {(weights_bytes/size_bytes)*100:.1f}% of total size")
    
    # Try to load the model to get more info
    try:
        model = load_coreml_model(model_path, compute_units="CPU_ONLY")  # Use CPU for quick loading
        
        # Get metadata for detailed model information
        metadata = get_model_metadata(model)
        if "type" in metadata:
            print(f"Model Type: {metadata['type']}")
        if "hidden_size" in metadata:
            print(f"Hidden Size: {metadata['hidden_size']}")
        if "vocab_size" in metadata:
            print(f"Vocabulary Size: {metadata['vocab_size']}")
            
        # Print input information
        print("\nInputs:")
        for input_info in metadata.get("inputs", []):
            name = input_info.get("name", "unknown")
            shape = input_info.get("shape", "unknown")
            data_type = input_info.get("data_type", "unknown")
            print(f"  - {name}: shape={shape}, type={data_type}")
        
        # Print output information
        print("\nOutputs:")
        for output_info in metadata.get("outputs", []):
            name = output_info.get("name", "unknown")
            shape = output_info.get("shape", "unknown")
            data_type = output_info.get("data_type", "unknown")
            print(f"  - {name}: shape={shape}, type={data_type}")
            
    except Exception as e:
        print(f"Error getting model details: {e}")
    
    print("===========================\n")


def main():
    """Main entry point"""
    args = parse_args()
    
    # Display system info
    system_info = get_system_info()
    print("\n=== System Information ===")
    print(f"Mac Model: {system_info.get('mac_model', 'Unknown')}")
    
    # Use the user-friendly macOS version if available, otherwise fall back to the old format
    if 'macos_version' in system_info:
        print(f"OS: {system_info['macos_version']}")
    else:
        print(f"OS: {system_info.get('os', {}).get('name', 'Unknown')} "
              f"{system_info.get('os', {}).get('release', '')}")
    
    print(f"CPU: {system_info.get('cpu', {}).get('brand', 'Unknown')}")
    print(f"RAM: {system_info.get('ram', {}).get('total_gb', 'Unknown')} GB")
    print(f"Apple Silicon: {'Yes' if system_info.get('apple_silicon', False) else 'No'}")
    print("===========================\n")
    
    # Print model info
    print_model_info(args.model)
    
    # Initialize benchmark
    benchmark = Benchmark()
    
    # Benchmark with the requested compute units
    print(f"Profiling with compute units: {args.compute_units}")
    print(f"Using specified hidden size: {args.hidden_size}")
    if args.tflops is not None:
        print(f"Using provided TFLOPS value: {args.tflops}")
    
    result = benchmark.benchmark_coreml_file(
        model_path=args.model,
        num_runs=args.iterations,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
        compute_units=args.compute_units,
        known_tflops=args.tflops
    )
    
    # Also benchmark with CPU only if requested
    if args.compare_cpu and args.compute_units != "CPU_ONLY":
        print("\nComparing with CPU-only performance...")
        cpu_result = benchmark.benchmark_coreml_file(
            model_path=args.model,
            num_runs=args.iterations,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            hidden_size=args.hidden_size,
            compute_units="CPU_ONLY",
            known_tflops=args.tflops
        )
        
        # Calculate speedup
        ane_time = result.inference_time_ms
        cpu_time = cpu_result.inference_time_ms
        ane_gbps = result.throughput_gb_s
        cpu_gbps = cpu_result.throughput_gb_s
        ane_tflops = result.tflops
        cpu_tflops = cpu_result.tflops
        
        speedup = cpu_time / ane_time if ane_time > 0 else 0
        gbps_ratio = ane_gbps / cpu_gbps if cpu_gbps > 0 else 0
        
        print("\n=== Performance Comparison ===")
        print(f"ANE Inference: {ane_time:.2f} ms")
        print(f"CPU Inference: {cpu_time:.2f} ms")
        print(f"Time Speedup: {speedup:.2f}x")
        print(f"ANE Throughput: {ane_gbps:.2f} GB/s")
        print(f"CPU Throughput: {cpu_gbps:.2f} GB/s")
        print(f"Throughput Ratio: {gbps_ratio:.2f}x")
        
        # Only print TFLOPS information if TFLOPS values are available
        if ane_tflops is not None and cpu_tflops is not None:
            tflops_ratio = ane_tflops / cpu_tflops if cpu_tflops > 0 else 0
            print(f"ANE TFLOPS: {ane_tflops:.4f}")
            print(f"CPU TFLOPS: {cpu_tflops:.4f}")
            print(f"TFLOPS Ratio: {tflops_ratio:.2f}x")
            
        print("=============================\n")
    
    # Generate report
    report_url = benchmark.generate_report(
        output_path=args.report,
        upload=args.upload,
        upload_service=args.upload_service,
        include_charts=args.include_charts
    )
    
    if report_url:
        print(f"Report uploaded to: {report_url}")
    
    # Open the HTML report in the default web browser
    report_path = os.path.abspath(args.report)
    print(f"Opening report: {report_path}")
    webbrowser.open(f"file://{report_path}", new=2)
    
    # Save JSON results if requested
    if args.output:
        # Create results dictionary
        results = {
            "model_path": args.model,
            "model_size_bytes": get_model_size(args.model),
            "model_size_mb": get_model_size(args.model) / (1024 * 1024),
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "hidden_size": args.hidden_size,
            "iterations": args.iterations,
            "compute_units": args.compute_units,
            "inference_time_ms": result.inference_time_ms,
            "throughput_gbps": result.throughput_gbps,
            "tflops": result.tflops,
            "system_info": system_info,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save to file
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {args.output}")
    
    print("\nProfile complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 