#!/usr/bin/env python3
"""
Basic example demonstrating how to use ANEMLL-Bench to benchmark
models on the Apple Neural Engine
"""

import os
import sys
import json
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoModel

# Add parent directory to path to import anemll_bench
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from anemll_bench import Benchmark
from anemll_bench.utils.system_info import get_system_info


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ANEMLL-Bench: Apple Neural Engine Benchmarking Tool')
    parser.add_argument('--model', type=str, default="microsoft/phi-2", 
                        help='Hugging Face model ID to benchmark')
    parser.add_argument('--sequence-length', type=int, default=128,
                        help='Sequence length for the input')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for the input')
    parser.add_argument('--runs', type=int, default=50,
                        help='Number of benchmark runs')
    parser.add_argument('--output', type=str, default="benchmark_report.html",
                        help='Path to save the benchmark report')
    parser.add_argument('--config', type=str, default=None,
                        help='Optional: Path to benchmark configuration JSON file')
    parser.add_argument('--upload', action='store_true',
                        help='Upload the report to a sharing service')
    parser.add_argument('--upload-service', type=str, default='jsonbin', choices=['gist', 'pastebin', 'jsonbin'],
                        help='Service to upload the report to')
    return parser.parse_args()


def main():
    """Run a basic benchmark example"""
    args = parse_args()
    
    # Print system information
    system_info = get_system_info()
    print("\n=== System Information ===")
    print(f"Mac Model: {system_info.get('mac_model', 'Unknown')}")
    print(f"OS: {system_info.get('os', {}).get('name', 'Unknown')} {system_info.get('os', {}).get('release', '')}")
    print(f"CPU: {system_info.get('cpu', {}).get('brand', 'Unknown')}")
    print(f"RAM: {system_info.get('ram', {}).get('total_gb', 'Unknown')} GB")
    print(f"Apple Silicon: {'Yes' if system_info.get('apple_silicon', False) else 'No'}")
    print(f"Python: {system_info.get('python_version', 'Unknown')}")
    print("===========================\n")
    
    # Initialize benchmark with optional config file
    benchmark = Benchmark(config_path=args.config)
    
    if args.config:
        print(f"Running benchmarks from config file: {args.config}")
        benchmark.run()
    else:
        print(f"Benchmarking model: {args.model}")
        try:
            # Create input shape
            # For text models: [batch_size, sequence_length]
            # For vision models or more complex shapes, this would need to be adjusted
            print(f"Loading model from Hugging Face: {args.model}")
            try:
                # Try to load as causal LM first
                model = AutoModelForCausalLM.from_pretrained(
                    args.model, 
                    torch_dtype=torch.float16,
                    device_map="cpu"  # Ensure it's loaded on CPU first
                )
                print("Model loaded as Causal LM")
            except Exception as e:
                print(f"Could not load as Causal LM, trying generic model: {e}")
                model = AutoModel.from_pretrained(
                    args.model, 
                    torch_dtype=torch.float16,
                    device_map="cpu"
                )
                print("Model loaded as generic model")
            
            # Get model's hidden size - different models have different attributes
            hidden_size = 768  # Default fallback
            if hasattr(model.config, 'hidden_size'):
                hidden_size = model.config.hidden_size
            elif hasattr(model.config, 'd_model'):
                hidden_size = model.config.d_model
            
            # Standard input shape for transformer models
            input_shape = [args.batch_size, args.sequence_length, hidden_size]
            print(f"Using input shape: {input_shape}")
            
            # Benchmark on CPU first
            print("\n=== Running CPU Benchmark ===")
            cpu_result = benchmark.benchmark_model(
                model=model, 
                model_name=args.model, 
                input_shape=input_shape,
                backend="CPU", 
                num_runs=args.runs
            )
            
            # Benchmark on Apple Neural Engine
            print("\n=== Running ANE Benchmark ===")
            ane_result = benchmark.benchmark_model(
                model=model, 
                model_name=args.model, 
                input_shape=input_shape,
                backend="ANE", 
                num_runs=args.runs
            )
            
            # Calculate speedup
            speedup = cpu_result.inference_time_ms / ane_result.inference_time_ms if ane_result.inference_time_ms > 0 else 0
            
            print("\n=== Benchmark Summary ===")
            print(f"Model: {args.model}")
            print(f"CPU Inference: {cpu_result.inference_time_ms:.2f} ms, {cpu_result.tflops:.4f} TFLOPS")
            print(f"ANE Inference: {ane_result.inference_time_ms:.2f} ms, {ane_result.tflops:.4f} TFLOPS")
            print(f"ANE Throughput: {ane_result.throughput_gbps:.2f} GB/s")
            print(f"Speedup: {speedup:.2f}x")
            
            # Generate report
            print(f"\nGenerating report to {args.output}...")
            report_url = benchmark.generate_report(
                output_path=args.output,
                upload=args.upload,
                upload_service=args.upload_service
            )
            
            if report_url:
                print(f"Report uploaded to: {report_url}")
                print("You can share this URL to let others view your benchmark results.")
            
        except Exception as e:
            print(f"Error benchmarking model: {e}")
    
    print("\nBenchmark complete! Check the reports directory for results.")


if __name__ == "__main__":
    main() 