"""
Utilities for loading and preparing models for benchmarking
"""

import os
import torch
from transformers import AutoModel, AutoModelForCausalLM
import coremltools as ct
from typing import Dict, Any, Tuple, Optional


def download_from_hf(model_id: str, use_cache: bool = True) -> torch.nn.Module:
    """
    Download a model from Hugging Face model hub
    
    Args:
        model_id: Hugging Face model identifier
        use_cache: Whether to use cached models
        
    Returns:
        PyTorch model
    """
    try:
        # Try loading as a causaslLM model first (for most LLMs)
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            cache_dir="./model_cache" if use_cache else None
        )
    except:
        # Fallback to generic model loading
        return AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            cache_dir="./model_cache" if use_cache else None
        )


def convert_to_coreml(model: torch.nn.Module, 
                     input_shape: Tuple[int, ...],
                     model_name: str,
                     compute_units: str = "ALL") -> str:
    """
    Convert PyTorch model to CoreML format for Apple Neural Engine
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        model_name: Name for the saved model
        compute_units: CoreML compute units (ALL, CPU_AND_GPU, CPU_ONLY, etc.)
        
    Returns:
        Path to the saved CoreML model
    """
    # Prepare example input
    example_input = torch.randn(input_shape)
    
    # Create traced model
    traced_model = torch.jit.trace(model, example_input)
    
    # Convert to CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="input", shape=input_shape)
        ],
        compute_units=ct.ComputeUnit[compute_units]
    )
    
    # Save the model
    model_dir = os.path.join("model_cache", "coreml")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.mlmodel")
    mlmodel.save(model_path)
    
    return model_path


def load_model(config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a model based on configuration
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Tuple of (model, model_info)
    """
    model_type = config.get("type", "pytorch")
    model_id = config.get("id")
    model_path = config.get("path")
    input_shape = config.get("input_shape")
    
    model_info = {
        "name": config.get("name", model_id or os.path.basename(model_path)),
        "type": model_type,
        "input_shape": input_shape
    }
    
    if model_type == "pytorch":
        if model_id:
            model = download_from_hf(model_id)
        elif model_path and os.path.exists(model_path):
            model = torch.load(model_path)
        else:
            raise ValueError("Either model_id or model_path must be provided")
            
    elif model_type == "coreml":
        if model_path and os.path.exists(model_path):
            model = ct.models.MLModel(model_path)
        elif model_id:
            # Download from HF and convert
            pt_model = download_from_hf(model_id)
            model_path = convert_to_coreml(
                pt_model, 
                tuple(input_shape), 
                os.path.basename(model_id).replace("/", "_")
            )
            model = ct.models.MLModel(model_path)
        else:
            raise ValueError("Either model_id or model_path must be provided")
            
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
        
    return model, model_info 