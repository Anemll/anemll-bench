#!/usr/bin/env python
"""
Benchmark all platform-specific models and generate a comprehensive report.
This script automates the entire process:
1. Syncs (downloads if needed) all models for the current macOS version
2. Benchmarks each available model
3. Generates a consolidated report with comparison charts
"""

import logging
import sys
import os
import argparse
import time
from datetime import datetime

# Add parent directory to path to import anemll_bench
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anemll_bench import Benchmark
from anemll_bench.models import get_macos_version

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Benchmark all platform-specific models")
    parser.add_argument("--no-sync", action="store_true", help="Skip model synchronization")
    parser.add_argument("--no-charts", action="store_true", help="Do not include charts in report")
    parser.add_argument("--force-redownload", action="store_true", help="Force re-download of models even if they exist")
    parser.add_argument("--use-local", action="store_true", help="Use local models if they exist, even if they might be corrupted")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the report in a browser")
    parser.add_argument("--runs", type=int, default=300, help="Number of benchmark runs per model (default: 300)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--sequence-length", type=int, default=1, help="Sequence length (default: 1)")
    parser.add_argument("--output", type=str, help="Custom output path for the report")
    parser.add_argument("--model", type=str, help="Specific model to benchmark (benchmarks all if not specified)")
    parser.add_argument("--local-model-path", type=str, help="Path to a local model file to use instead of downloading")
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Check if running on macOS
    macos_version = get_macos_version()
    if not macos_version:
        logger.error("This script is intended to run on macOS systems only.")
        return 1
    
    logger.info(f"Running on macOS version category: {macos_version}")
    
    # Define report output path if not specified
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"benchmark_report_{macos_version}_{timestamp}.html"
    
    # Create benchmark instance
    benchmark = Benchmark()
    
    # If a local model path is provided, try to use it directly
    if args.local_model_path:
        if not os.path.exists(args.local_model_path):
            logger.error(f"Local model path does not exist: {args.local_model_path}")
            return 1
        
        logger.info(f"Using local model: {args.local_model_path}")
        try:
            import coremltools as ct
            model = ct.models.MLModel(args.local_model_path)
            model_name = os.path.basename(args.local_model_path)
            
            # Run the benchmark directly
            result = benchmark.benchmark_model(
                model=model,
                model_name=model_name,
                input_shape=[args.batch_size, args.sequence_length, 4096],  # Assuming 4096 is the hidden size
                backend="ANE",
                num_runs=args.runs
            )
            
            logger.info(f"Benchmark result for {model_name}:")
            logger.info(f"  - Inference time: {result.inference_time_ms:.2f} ms")
            if result.tflops is not None:
                logger.info(f"  - TFLOPs: {result.tflops:.2f}")
            logger.info(f"  - Throughput: {result.throughput_gb_s:.2f} GB/s")
            
            # Generate a report
            benchmark.generate_report(
                output_path=args.output,
                include_charts=not args.no_charts,
                auto_open=not args.no_browser
            )
            
            logger.info(f"Report generated: {args.output}")
            return 0
        except Exception as e:
            logger.error(f"Error loading or benchmarking local model: {e}")
            return 1
    
    # Start timing
    start_time = time.time()
    logger.info("Starting platform model benchmarking process...")
    
    try:
        # If a specific model is requested
        if args.model:
            logger.info(f"Benchmarking specific model: {args.model}")
            result = benchmark.benchmark_platform_model(
                model_name=args.model,
                num_runs=args.runs,
                batch_size=args.batch_size,
                sequence_length=args.sequence_length,
                check_online=not args.no_sync,
                force_redownload=args.force_redownload,
                use_local_if_exists=args.use_local
            )
            
            logger.info(f"Benchmark result for {args.model}:")
            logger.info(f"  - Inference time: {result.inference_time_ms:.2f} ms")
            if result.tflops is not None:
                logger.info(f"  - TFLOPs: {result.tflops:.2f}")
            logger.info(f"  - Throughput: {result.throughput_gb_s:.2f} GB/s")
            
            # Generate a report for this single model
            benchmark.generate_report(
                output_path=args.output, 
                include_charts=not args.no_charts,
                auto_open=not args.no_browser
            )
            
        else:
            # Run the benchmarks for all models
            results = benchmark.benchmark_all_platform_models(
                num_runs=args.runs,
                batch_size=args.batch_size,
                sequence_length=args.sequence_length,
                sync_first=not args.no_sync,
                include_charts=not args.no_charts,
                output_path=args.output,
                force_redownload=args.force_redownload,
                auto_open=not args.no_browser,
                use_local_if_exists=args.use_local
            )
            
            # No need to manually open the browser here, as it's handled by benchmark_all_platform_models
        
        # End timing
        end_time = time.time()
        duration = end_time - start_time
        
        # Log results
        logger.info(f"Benchmark complete in {duration:.1f} seconds")
        logger.info(f"Report generated: {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during benchmarking: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 