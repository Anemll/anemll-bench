#!/usr/bin/env python
"""
Test script for manually loading and running CoreML models
"""

import os
import sys
import logging
import numpy as np
import argparse

try:
    import coremltools as ct
except ImportError:
    print("CoreML Tools not installed. Please install with: pip install coremltools")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def list_models_in_cache():
    """List all models in the cache directory"""
    cache_dir = os.path.expanduser("~/.cache/anemll-bench/models")
    
    if not os.path.exists(cache_dir):
        logger.error(f"Cache directory not found: {cache_dir}")
        return []
    
    models = os.listdir(cache_dir)
    return [os.path.join(cache_dir, model) for model in models]

def load_model_with_various_methods(model_path):
    """Try loading a model with different approaches"""
    logger.info(f"Testing model loading for: {model_path}")
    
    # Print model directory contents for mlmodelc
    if os.path.isdir(model_path):
        logger.info(f"Model directory contents:")
        for root, dirs, files in os.walk(model_path):
            rel_path = os.path.relpath(root, model_path)
            prefix = "└── " if rel_path == "." else f"├── {rel_path}/"
            for file in files:
                logger.info(f"{prefix}{file}")
    
    methods_tried = 0
    errors = []
    
    # Method 1: Standard loading
    try:
        logger.info("Method 1: Trying standard loading")
        model = ct.models.MLModel(model_path)
        logger.info("✅ SUCCESS: Standard loading worked")
        
        # Get model spec information
        logger.info(f"Model inputs: {model.get_spec().description.input}")
        logger.info(f"Model outputs: {model.get_spec().description.output}")
        
        return model
    except Exception as e:
        methods_tried += 1
        errors.append(f"Method 1 failed: {str(e)}")
        logger.warning(f"Method 1 failed: {str(e)}")
    
    # Method 2: Loading with skip_model_load=True
    try:
        logger.info("Method 2: Trying with skip_model_load=True")
        model = ct.models.MLModel(model_path, skip_model_load=True)
        logger.info("✅ SUCCESS: Loading with skip_model_load worked")
        
        # Get model spec information
        try:
            logger.info(f"Model inputs: {model.get_spec().description.input}")
            logger.info(f"Model outputs: {model.get_spec().description.output}")
        except:
            logger.warning("Couldn't get model spec information")
        
        return model
    except Exception as e:
        methods_tried += 1
        errors.append(f"Method 2 failed: {str(e)}")
        logger.warning(f"Method 2 failed: {str(e)}")
    
    # Method 3: Create a dummy spec and try to run prediction
    if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "model.mil")):
        try:
            logger.info("Method 3: Creating a custom model")
            
            # Create a dummy model
            from coremltools.proto import Model_pb2
            
            # Create a minimal spec
            spec = Model_pb2.Model()
            spec.specificationVersion = 5
            
            # Add input description
            input_desc = spec.description.input.add()
            input_desc.name = "input_ids"
            input_desc.type.multiArrayType.shape.append(1)  # Batch
            input_desc.type.multiArrayType.shape.append(512)  # Sequence length
            input_desc.type.multiArrayType.dataType = Model_pb2.ArrayFeatureType.FLOAT32
            
            # Create dummy model
            dummy_model = ct.models.MLModel(spec)
            dummy_model.path = model_path
            
            logger.info("✅ SUCCESS: Created dummy model")
            return dummy_model
        except Exception as e:
            methods_tried += 1
            errors.append(f"Method 3 failed: {str(e)}")
            logger.warning(f"Method 3 failed: {str(e)}")
    
    # If all methods failed
    logger.error(f"All {methods_tried} loading methods failed.")
    for error in errors:
        logger.error(f"  - {error}")
    
    return None

def run_model_inference(model, model_path):
    """Try to run inference on the model"""
    if model is None:
        logger.error("No model to run inference on")
        return
    
    logger.info(f"Testing inference on model: {model_path}")
    
    try:
        # Create a dummy input
        dummy_input = {"input_ids": np.random.rand(1, 512).astype(np.float32)}
        
        # Try to run inference
        logger.info("Running inference with dummy input...")
        result = model.predict(dummy_input)
        
        logger.info("✅ SUCCESS: Model inference successful!")
        logger.info(f"Prediction result keys: {list(result.keys())}")
        
        # Print first few values of first output
        first_output_key = list(result.keys())[0]
        first_output = result[first_output_key]
        logger.info(f"First output ({first_output_key}) shape: {first_output.shape}")
        logger.info(f"First few values: {first_output.flatten()[:5]}")
        
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Test CoreML model loading")
    parser.add_argument("--model", type=str, help="Path to the model to test")
    parser.add_argument("--list-models", action="store_true", help="List all models in cache")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference test")
    args = parser.parse_args()
    
    if args.list_models:
        models = list_models_in_cache()
        logger.info(f"Found {len(models)} models in cache:")
        for model in models:
            logger.info(f" - {model}")
        return
    
    # Test loading one specific model or all models in cache
    if args.model:
        model_path = os.path.expanduser(args.model)
        model = load_model_with_various_methods(model_path)
        
        if model and not args.skip_inference:
            run_model_inference(model, model_path)
    else:
        # Test all models in cache
        models = list_models_in_cache()
        logger.info(f"Testing {len(models)} models in cache")
        
        for model_path in models:
            model = load_model_with_various_methods(model_path)
            
            if model and not args.skip_inference:
                run_model_inference(model, model_path)
            
            logger.info("-" * 80)

if __name__ == "__main__":
    main() 