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
import webbrowser
import subprocess
from pathlib import Path
from typing import Optional
from datetime import datetime

# Add parent directory to path to import anemll_bench
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anemll_bench import Benchmark
from anemll_bench.models import get_macos_version, list_available_platform_models, clear_cache

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
    parser.add_argument("--sequence-length", type=int, default=None, help="Sequence length (default: 1)")
    parser.add_argument("--output", type=str, default=None, help="Custom output path for the report")
    parser.add_argument("--model", type=str, default=None, help="Specific model to benchmark (benchmarks all if not specified)")
    parser.add_argument("--local-model-path", type=str, default=None, help="Path to a local model file to use instead of downloading")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging (DEBUG level)")
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Check if running on macOS
    macos_version = get_macos_version()
    if not macos_version:
        logger.error("This script is intended to run on macOS systems only.")
        return 1
    
    logger.info(f"Running on macOS version category: {macos_version}")
    
    # Set output path
    if args.output is None:
        # Use the macOS version in the report name
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_report_{macos_version}_{timestamp}.html"
        # Report will be saved in the cache directory
        cache_dir = os.path.expanduser("~/.cache/anemll-bench/reports")
        os.makedirs(cache_dir, exist_ok=True)
        args.output = os.path.join(cache_dir, filename)
    else:
        # If a custom path is provided, check if it's absolute
        if not os.path.isabs(args.output):
            # If it's relative, save it in the cache directory
            cache_dir = os.path.expanduser("~/.cache/anemll-bench/reports")
            os.makedirs(cache_dir, exist_ok=True)
            args.output = os.path.join(cache_dir, args.output)

    # Debug output path
    print(f"DEBUG: Output path: {args.output}")
    print(f"DEBUG: Absolute output path: {os.path.abspath(args.output)}")
    
    logger.info("Starting platform model benchmarking process...")

    # Initialize benchmark
    benchmark = Benchmark()
    
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
        
        # Generate report for single model
        benchmark.generate_report(args.output)
    else:
        # Run benchmarks for all models
        benchmark.benchmark_all_platform_models(
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

    # Log completion
    logger.info(f"Report generated: {args.output}")

    return 0

if __name__ == "__main__":
    sys.exit(main()) 