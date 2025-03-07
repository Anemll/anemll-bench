#!/usr/bin/env python3
# Simple script to benchmark LM head models with sequence_length=1
import os
import sys
import time
import logging
import argparse

from anemll_bench import Benchmark
from anemll_bench.models.model_loader import get_macos_version, sync_platform_models, get_platform_specific_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Benchmark LM head models with sequence_length=1")
    parser.add_argument("--use-local", action="store_true", help="Use local models if they exist")
    parser.add_argument("--num-runs", type=int, default=50, help="Number of benchmark runs")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser with report")
    parser.add_argument("--model", type=str, help="Specific model to benchmark")
    args = parser.parse_args()
    
    logger.info(f"Running on macOS version category: {get_macos_version()}")
    
    # Initialize the benchmark tool
    benchmark = Benchmark()
    
    # Synchronize models first
    logger.info("Starting platform model synchronization...")
    sync_platform_models()
    
    # Get all available models
    all_models = get_platform_specific_models()
    
    # Filter for LM head models only
    lm_head_models = [model for model in all_models if "lm_head" in model.get("name", "").lower()]
    
    # If a specific model is specified, filter for just that model
    if args.model:
        lm_head_models = [model for model in lm_head_models if args.model.lower() in model.get("name", "").lower()]
    
    if not lm_head_models:
        logger.error("No LM head models found!")
        return 1
    
    logger.info(f"Found {len(lm_head_models)} LM head models to benchmark")
    
    results = []
    
    # Benchmark each model with sequence_length=1
    for model_config in lm_head_models:
        model_name = model_config.get("name")
        if not model_name:
            continue
            
        logger.info(f"Benchmarking model: {model_name}")
        
        try:
            # Explicitly use sequence_length=1 for LM head models
            result = benchmark.benchmark_platform_model(
                model_name=model_name,
                num_runs=args.num_runs,
                batch_size=1,
                sequence_length=1,  # Explicitly force sequence_length to 1
                check_online=False,
                force_redownload=False,
                use_local_if_exists=args.use_local
            )
            
            results.append(result)
            
            logger.info(f"Benchmark successful for {model_name}")
            logger.info(f"  - Latency: {result.latency_ms:.2f} ms")
            logger.info(f"  - TFLOPs: {result.tflops:.2f}")
            logger.info(f"  - Throughput: {result.throughput_gb_s:.2f} GB/s")
            
        except Exception as e:
            logger.error(f"Error benchmarking {model_name}: {e}")
    
    # Generate a report
    if results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = f"lm_head_benchmark_report_{timestamp}.html"
        
        logger.info(f"Generating report: {report_path}")
        benchmark.generate_report(
            output_path=report_path,
            include_charts=True,
            auto_open=not args.no_browser
        )
        
        logger.info(f"Report generated: {report_path}")
    else:
        logger.warning("No benchmark results were generated")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 