#!/usr/bin/env python
"""
Example script demonstrating how to check for updated model definitions from Hugging Face
"""

import logging
import sys
import os

# Add parent directory to path to import anemll_bench
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anemll_bench.models import (
    check_and_update_platform_models,
    list_available_platform_models,
    get_macos_version,
    download_meta_file
)

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Check macOS version
    macos_version = get_macos_version()
    if not macos_version:
        logger.error("This script is intended to run on macOS systems only.")
        return
    
    logger.info(f"Running on macOS version category: {macos_version}")
    
    # Get local model definitions
    logger.info("Local model definitions:")
    local_models = list_available_platform_models()
    
    # Check for updates from Hugging Face
    logger.info("\nChecking for updated model definitions on Hugging Face...")
    online_models = check_and_update_platform_models()
    
    # Compare local and online models
    if not local_models and not online_models:
        logger.warning("No models found locally or online.")
        return
    
    # Show differences if any
    local_model_names = set(model.get("name") for model in local_models)
    online_model_names = set(model.get("name") for model in online_models)
    
    new_models = online_model_names - local_model_names
    removed_models = local_model_names - online_model_names
    common_models = local_model_names.intersection(online_model_names)
    
    if new_models:
        logger.info(f"\nNew models available online: {', '.join(new_models)}")
    
    if removed_models:
        logger.info(f"\nModels no longer available online: {', '.join(removed_models)}")
    
    if common_models:
        logger.info(f"\nModels available both locally and online: {', '.join(common_models)}")
    
    # Force update the meta file
    if online_models:
        logger.info("\nUpdating local meta.yalm file...")
        download_meta_file(force_update=True)
        logger.info("Local meta.yalm file updated with latest model definitions.")

if __name__ == "__main__":
    main() 