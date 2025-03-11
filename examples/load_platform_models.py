#!/usr/bin/env python
"""
Example script demonstrating how to use platform-specific model loading functionality
"""

import logging
import sys
import os
import argparse

# Add parent directory to path to import anemll_bench
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anemll_bench.models import (
    list_available_platform_models,
    get_macos_version
)
from anemll_bench import Benchmark

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Benchmark platform-specific models")
    parser.add_argument("--check-online", action="store_true", help="Check online for model updates")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of benchmark runs (default: 10)")
    parser.add_argument("--model", type=str, help="Specific model to benchmark (default: all available)")
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Check macOS version
    macos_version = get_macos_version()
    if not macos_version:
        logger.error("This script is intended to run on macOS systems only.")
        return
    
    logger.info(f"Running on macOS version category: {macos_version}")
    
    # List available platform-specific models
    logger.info(f"Looking for models {'(including online check)' if args.check_online else '(local only)'}...")
    platform_models = list_available_platform_models(check_online=args.check_online)
    
    if not platform_models:
        logger.error("No platform-specific models available.")
        return
    
    # Create a benchmark instance
    benchmark = Benchmark()
    
    # Determine which models to benchmark
    models_to_benchmark = []
    if args.model:
        # Benchmark only the specified model
        for model_config in platform_models:
            if model_config.get("name") == args.model:
                models_to_benchmark.append(model_config)
                break
        
        if not models_to_benchmark:
            logger.error(f"Model '{args.model}' not found.")
            return
    else:
        # Benchmark all available models
        models_to_benchmark = platform_models
    
    # Try to benchmark each selected model
    for model_config in models_to_benchmark:
        model_name = model_config.get("name")
        logger.info(f"Loading and benchmarking model: {model_name}")
        
        try:
            # Using the simplified method with online check option
            result = benchmark.benchmark_platform_model(
                model_name=model_name,
                num_runs=args.num_runs,
                check_online=args.check_online
            )
            
            logger.info(f"Benchmark results for {model_name}:")
            logger.info(f"  - Inference time: {result.inference_time_ms:.2f} ms")
            if result.tflops is not None:
                logger.info(f"  - TFLOPs: {result.tflops:.2f}")
            else:
                logger.info(f"  - TFLOPs: Not available")
            logger.info(f"  - Throughput: {result.throughput_gb_s:.2f} GB/s")
            
        except Exception as e:
            logger.error(f"Error benchmarking model {model_name}: {e}")
    
    # Generate a benchmark report
    benchmark.generate_report(output_path="platform_models_benchmark.html")
    logger.info("Benchmark report generated: platform_models_benchmark.html")

if __name__ == "__main__":
    main() 