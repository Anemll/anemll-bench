#!/usr/bin/env python
"""
Utility script to synchronize all platform-specific models
"""

import logging
import sys
import os
import argparse

# Add parent directory to path to import anemll_bench
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anemll_bench.models import sync_platform_models, get_macos_version

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Synchronize ANEMLL-Bench platform models")
    parser.add_argument("--force", action="store_true", help="Force update of meta.yalm")
    parser.add_argument("--update", action="store_true", 
                        help="Update meta.yalm file and download any missing or new models (recommended)")
    parser.add_argument("--parallel", action="store_true", 
                        help="Download models in parallel for faster synchronization")
    parser.add_argument("--workers", type=int, default=4, 
                        help="Number of parallel download workers (default: 4)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode (less output)")
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Check if running on macOS
    macos_version = get_macos_version()
    if not macos_version:
        logger.error("This script is intended to run on macOS systems only.")
        return 1
    
    logger.info(f"Running on macOS version category: {macos_version}")
    
    # Use force_update if either --force or --update flag is specified
    force_update = args.force or args.update
    
    # Synchronize all platform models
    try:
        results = sync_platform_models(
            force_update=force_update,
            parallel=args.parallel,
            max_workers=args.workers
        )
        
        # Print summary
        print(f"\nSynchronization Summary:")
        print(f"  - Meta file updated: {'Yes' if results['meta_updated'] else 'No'}")
        print(f"  - Models checked: {results['models_checked']}")
        print(f"  - Models downloaded: {results['models_downloaded']}")
        print(f"  - Models skipped (already exist): {results['models_skipped']}")
        print(f"  - Models failed: {results['models_failed']}")
        
        # Print details for downloaded models
        if results['models_downloaded'] > 0:
            print("\nDownloaded Models:")
            for model in results['models']:
                if model['action'] == 'downloaded':
                    print(f"  - {model['name']} ({model['type']})")
        
        # Print details for failed models
        if results['models_failed'] > 0:
            print("\nFailed Models:")
            for model in results['models']:
                if model['action'] in ['failed', 'error']:
                    print(f"  - {model['name']} ({model['type']})")
                    if 'error' in model:
                        print(f"    Error: {model['error']}")
        
        # Return success if we didn't have any failures
        return 0 if results['models_failed'] == 0 else 1
    
    except Exception as e:
        logger.error(f"Error synchronizing models: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 