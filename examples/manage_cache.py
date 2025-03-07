#!/usr/bin/env python
"""
Utility script for managing the ANEMLL-Bench model cache
"""

import logging
import sys
import os
import argparse
import json

# Add parent directory to path to import anemll_bench
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anemll_bench.models import (
    get_cache_info,
    clear_cache,
    check_and_update_platform_models,
    sync_platform_models,
    CACHE_DIR
)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Manage ANEMLL-Bench model cache")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Display cache information")
    info_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    
    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear the cache")
    clear_parser.add_argument("--all", action="store_true", help="Clear all cache including meta file")
    clear_parser.add_argument("--model", type=str, help="Clear only the specified model")
    
    # Update command
    update_parser = subparsers.add_parser("update", help="Update model definitions from online source")
    
    # Sync command
    sync_parser = subparsers.add_parser("sync", help="Synchronize all platform models (download if not present)")
    sync_parser.add_argument("--force", action="store_true", help="Force update of meta.yalm before syncing")
    sync_parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Execute the requested command
    if args.command == "info":
        display_cache_info(json_output=args.json)
    elif args.command == "clear":
        clear_model_cache(include_meta=args.all, model_name=args.model)
    elif args.command == "update":
        update_model_definitions()
    elif args.command == "sync":
        sync_all_models(force_update=args.force, json_output=args.json)
    else:
        parser.print_help()

def display_cache_info(json_output=False):
    """Display information about the cache"""
    cache_info = get_cache_info()
    
    if json_output:
        # Output in JSON format
        print(json.dumps(cache_info, indent=2))
    else:
        # Output in human-readable format
        print(f"\nANEMLL-Bench Cache Information")
        print(f"==============================")
        print(f"Cache Directory: {cache_info['cache_dir']}")
        print(f"Models Directory: {cache_info['models_dir']}")
        print(f"Meta File: {cache_info['meta_file']} (Exists: {cache_info['meta_file_exists']})")
        print(f"Total Cache Size: {cache_info['total_size_mb']:.2f} MB")
        
        if cache_info['models']:
            print(f"\nCached Models:")
            print(f"-------------")
            for model in cache_info['models']:
                print(f"  - {model['name']} ({model['type']})")
                print(f"    Path: {model['path']}")
                print(f"    Size: {model['size_mb']:.2f} MB")
                print()
        else:
            print("\nNo models in cache")

def clear_model_cache(include_meta=False, model_name=None):
    """Clear the model cache"""
    logger = logging.getLogger(__name__)
    
    if model_name:
        logger.info(f"Clearing model: {model_name}")
    elif include_meta:
        logger.info("Clearing entire cache including meta file")
    else:
        logger.info("Clearing model cache")
    
    success = clear_cache(include_meta=include_meta, model_name=model_name)
    
    if success:
        logger.info("Cache cleared successfully")
    else:
        logger.error("Failed to clear cache")

def update_model_definitions():
    """Update model definitions from online source"""
    logger = logging.getLogger(__name__)
    
    logger.info("Checking for updated model definitions...")
    models = check_and_update_platform_models()
    
    if models:
        logger.info(f"Updated model definitions: {len(models)} models available")
        for model in models:
            name = model.get("name", "unknown")
            model_type = model.get("type", "unknown")
            logger.info(f"  - {name} ({model_type})")
    else:
        logger.warning("No model definitions found online")

def sync_all_models(force_update=False, json_output=False):
    """Synchronize all platform models"""
    logger = logging.getLogger(__name__)
    
    logger.info("Synchronizing all platform models...")
    results = sync_platform_models(force_update=force_update)
    
    if json_output:
        # Output in JSON format
        print(json.dumps(results, indent=2))
    else:
        # Output in human-readable format
        print(f"\nPlatform Model Synchronization Results")
        print(f"=====================================")
        print(f"Meta file updated: {results['meta_updated']}")
        print(f"Models checked: {results['models_checked']}")
        print(f"Models downloaded: {results['models_downloaded']}")
        print(f"Models skipped (already exist): {results['models_skipped']}")
        print(f"Models failed: {results['models_failed']}")
        
        if results['models']:
            print(f"\nModel Details:")
            print(f"-------------")
            for model in results['models']:
                print(f"  - {model['name']} ({model['type']})")
                print(f"    Path: {model['path']}")
                print(f"    Action: {model['action']}")
                if 'error' in model:
                    print(f"    Error: {model['error']}")
                print()

if __name__ == "__main__":
    main() 