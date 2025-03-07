#!/usr/bin/env python3
"""
Simple script to benchmark local LM head models with sequence_length=1
without performing any synchronization or downloads.
"""

import os
import sys
import logging
import argparse
from typing import List, Dict

from anemll_bench import Benchmark
from anemll_bench.models.model_loader import read_meta_file, get_macos_version

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_local_models() -> List[Dict]:
    """Get list of models available locally without any synchronization."""
    meta_data = read_meta_file()
    if not meta_data or 'model_info' not in meta_data:
        logger.error("No model info found in meta file")
        return []
    
    macos_version = get_macos_version()
    if not macos_version or macos_version not in meta_data['model_info']:
        logger.error(f"No models found for {macos_version}")
        return []
    
    return meta_data['model_info'][macos_version]

def main():
    parser = argparse.ArgumentParser(description="Benchmark local LM head models with sequence_length=1")
    parser.add_argument("--model", type=str, help="Specific model to benchmark")
    parser.add_argument("--num-runs", type=int, default=50, help="Number of benchmark runs")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser with report")
    args = parser.parse_args()
    
    logger.info(f"Running on macOS version category: {get_macos_version()}")
    
    # Initialize the benchmark tool
    benchmark = Benchmark()
    
    # Get all available models without syncing
    logger.info("Getting local models without synchronization...")
    all_models = get_local_models()
    logger.info(f"Found {len(all_models)} models locally")
    
    # Filter for LM head models only
    lm_head_models = [model for model in all_models if "lm_head" in model.get("name", "").lower()]
    
    # If a specific model is specified, filter for just that model
    if args.model:
        lm_head_models = [model for model in lm_head_models if args.model.lower() in model.get("name", "").lower()]
    
    if not lm_head_models:
        logger.error("No LM head models found!")
        return 1
    
    logger.info(f"Found {len(lm_head_models)} LM head models to benchmark")
    for model in lm_head_models:
        logger.info(f"  - {model.get('name')}")
    
    results = []
    
    # Benchmark each model with sequence_length=1
    for model_config in lm_head_models:
        model_name = model_config.get("name")
        if not model_name:
            continue
            
        logger.info(f"Benchmarking model: {model_name}")
        
        try:
            # Explicitly use sequence_length=1 for LM head models
            # Set use_local_if_exists=True to avoid re-downloads
            # Set check_online=False to avoid checking for updates
            result = benchmark.benchmark_platform_model(
                model_name=model_name,
                num_runs=args.num_runs,
                batch_size=1,
                sequence_length=1,  # Explicitly force sequence_length to 1
                check_online=False,  # Don't check online
                force_redownload=False,  # Don't force redownload
                use_local_if_exists=True  # Use local models even if they might be corrupted
            )
            
            results.append(result)
            
            logger.info(f"Benchmark successful for {model_name}")
            if hasattr(result, 'latency_ms'):
                logger.info(f"  - Latency: {result.latency_ms:.2f} ms")
                logger.info(f"  - TFLOPs: {result.tflops:.2f}")
                logger.info(f"  - Throughput: {result.throughput_gb_s:.2f} GB/s")
            
        except Exception as e:
            logger.error(f"Error benchmarking {model_name}: {e}")
    
    # Generate a report
    if results:
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = f"lm_head_local_benchmark_{timestamp}.html"
        
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