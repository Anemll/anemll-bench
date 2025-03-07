"""
Utilities for loading and profiling CoreML models directly
"""

import os
import time
import numpy as np
import coremltools as ct
from typing import Dict, Any, List, Optional, Tuple, Union


def load_coreml_model(model_path: str, compute_units: str = "CPU_AND_NE") -> Any:
    """
    Load a CoreML model with specified compute units
    
    Args:
        model_path: Path to the CoreML model (.mlmodel or .mlmodelc)
        compute_units: Compute units to use (CPU_AND_NE, CPU_ONLY, ALL)
        
    Returns:
        Loaded CoreML model
    """
    print(f"Loading model: {model_path}")
    
    # Map string compute units to CoreML compute units
    compute_units_map = {
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "ALL": ct.ComputeUnit.ALL
    }
    
    cu = compute_units_map.get(compute_units, ct.ComputeUnit.CPU_AND_NE)
    
    try:
        if model_path.endswith('.mlmodelc'):
            # Load compiled model - note the parameter is called compute_units (plural)
            model = ct.models.CompiledMLModel(model_path, compute_units=cu)
        else:
            # Load uncompiled model - note the parameter is called compute_units (plural)
            model = ct.models.MLModel(model_path, compute_units=cu)
        
        print(f"Successfully loaded model: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def get_model_size(model_path: str, weights_only: bool = False) -> int:
    """
    Get the size of a CoreML model in bytes
    
    Args:
        model_path: Path to the CoreML model
        weights_only: If True, returns only the size of the model weights,
                     otherwise returns the total size of all model files
        
    Returns:
        Size of the model in bytes
    """
    if weights_only:
        # Look for the specific weights file based on the model format
        if model_path.endswith('.mlpackage'):
            weights_path = os.path.join(model_path, 'Data', 'com.apple.CoreML', 'weights', 'weight.bin')
            if os.path.exists(weights_path):
                return os.path.getsize(weights_path)
        elif model_path.endswith('.mlmodelc'):
            weights_path = os.path.join(model_path, 'weights', 'weight.bin')
            if os.path.exists(weights_path):
                return os.path.getsize(weights_path)
        
        # If we couldn't find the specific weights file, look for any weight file
        for dirpath, _, filenames in os.walk(model_path):
            for filename in filenames:
                if filename == 'weight.bin' or filename.endswith('.weights'):
                    filepath = os.path.join(dirpath, filename)
                    return os.path.getsize(filepath)
        
        # If no weights file is found, fall back to total size
        print(f"Warning: Could not find weights file in {model_path}, returning total model size.")
    
    # Calculate total model size (or fallback for weights_only)
    if os.path.isfile(model_path):
        return os.path.getsize(model_path)
    
    # If it's a directory (like .mlmodelc), sum up sizes
    total_size = 0
    for dirpath, _, filenames in os.walk(model_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    
    return total_size


def prepare_inputs(model: Any, batch_size: int = 1, sequence_length: int = 512, 
                  hidden_size: int = 2048) -> Dict[str, np.ndarray]:
    """
    Prepare inputs for the CoreML model
    
    Args:
        model: CoreML model
        batch_size: Batch size
        sequence_length: Sequence length for text models
        hidden_size: Hidden size for text models
        
    Returns:
        Dictionary of input tensors
    """
    # Get the input shapes from model
    input_specs = model.get_spec().description.input
    
    inputs = {}
    for input_spec in input_specs:
        name = input_spec.name
        
        # Handle different CoreML shape formats
        shape = []
        try:
            if hasattr(input_spec.type.multiArrayType, "shape"):
                # Get shape - handle both object with size attribute and direct int
                shape_elements = input_spec.type.multiArrayType.shape
                for dim in shape_elements:
                    if hasattr(dim, 'size'):
                        shape.append(dim.size)
                    else:
                        # If it's a direct integer
                        shape.append(dim)
            else:
                # Fallback for other formats
                print(f"Warning: Could not determine shape for input {name}, using default")
                shape = [batch_size, sequence_length, hidden_size]
        except Exception as e:
            print(f"Error getting shape for {name}: {e}")
            # Default fallback shape
            shape = [batch_size, sequence_length, hidden_size]
            
        # Adjust shape for batch dimension if needed
        if len(shape) >= 3 and shape[0] == 1:
            shape[0] = batch_size
        
        # For text models, adjust sequence length if needed
        if len(shape) >= 3 and shape[1] > 1:
            shape[1] = sequence_length
        
        # Create random input based on the shape
        try:
            data_type = input_spec.type.multiArrayType.dataType
            if data_type == ct.proto.FeatureType.ArrayFeatureType.ArrayDataType.FLOAT32:
                inputs[name] = np.random.rand(*shape).astype(np.float32)
            elif data_type == ct.proto.FeatureType.ArrayFeatureType.ArrayDataType.INT32:
                inputs[name] = np.random.randint(0, 100, size=shape).astype(np.int32)
            elif data_type == ct.proto.FeatureType.ArrayFeatureType.ArrayDataType.FLOAT16:
                inputs[name] = np.random.rand(*shape).astype(np.float16)
            else:
                # Default to float32
                inputs[name] = np.random.rand(*shape).astype(np.float32)
        except Exception as e:
            print(f"Error creating input for {name}: {e}, using default float32")
            inputs[name] = np.random.rand(*shape).astype(np.float32)
    
    return inputs


def profile_coreml_model(model: Any, num_iterations: int = 100, 
                       inputs: Optional[Dict[str, np.ndarray]] = None,
                       batch_size: int = 1, sequence_length: int = 512, 
                       hidden_size: Optional[int] = None, known_tflops: Optional[float] = None) -> Dict[str, Any]:
    """
    Profile a CoreML model's performance
    
    Args:
        model: CoreML model to profile
        num_iterations: Number of iterations for profiling
        inputs: Optional pre-generated inputs
        batch_size: Batch size (used if inputs not provided)
        sequence_length: Sequence length (used if inputs not provided)
        hidden_size: Hidden size (used if inputs not provided)
        known_tflops: Total trillion floating point operations per iteration (not TFLOPS rate)
        
    Returns:
        Dictionary containing profiling results
    """
    # Get model metadata to understand its architecture
    metadata = get_model_metadata(model)
    
    # Update hidden_size if we can determine it from metadata
    if hidden_size is None and "hidden_size" in metadata:
        hidden_size = metadata["hidden_size"]
        print(f"Using detected hidden_size: {hidden_size}")
    
    # If inputs not provided, generate them based on model type
    if inputs is None:
        # Check if this is a language model
        is_lang_model = any(name for name in model.get_spec().description.input 
                           if name.endswith('_ids') or name == 'input_ids')
        
        # For language models, use specialized input preparation
        if is_lang_model and "hidden_size" in metadata:
            inputs = prepare_lm_inputs(model, batch_size, sequence_length, hidden_size)
        else:
            inputs = prepare_inputs(model, batch_size, sequence_length, hidden_size)
    
    # Print a summary of the inputs we're using
    input_shapes = {name: arr.shape for name, arr in inputs.items()}
    print(f"Using inputs: {input_shapes}")
    
    # Warm up - Initial runs can be slower due to compilation and memory allocation
    print("Warming up model...")
    for _ in range(5):
        try:
            model.predict(inputs)
        except Exception as e:
            print(f"Warning: Error during warmup: {e}")
            break
    
    # Perform the actual profiling
    print(f"Running {num_iterations} iterations for profiling...")
    times = []
    for _ in range(num_iterations):
        start_time = time.time()
        try:
            prediction = model.predict(inputs)
            times.append((time.time() - start_time) * 1000)  # Convert to ms
        except Exception as e:
            print(f"Error during prediction: {e}")
            if not times:  # If no successful runs, can't continue
                raise Exception(f"Failed to run model: {e}")
    
    # Calculate statistics
    avg_time = np.mean(times)
    median_time = np.median(times)
    min_time = np.min(times)
    p99_time = np.percentile(times, 99)
    
    # Calculate memory footprint
    input_size_bytes = sum(inp.nbytes for inp in inputs.values())
    
    # Try to get output size
    try:
        prediction = model.predict(inputs)
        output_size_bytes = sum(out.nbytes for out in prediction.values())
    except:
        # Fallback if we can't measure output
        output_size_bytes = input_size_bytes  # Assume similar size
    
    # Get model size in bytes
    model_size_bytes = 0
    weights_size_bytes = 0
    model_path = None
    
    # Try to get model path from spec or attribute
    try:
        if hasattr(model, 'get_spec') and hasattr(model.get_spec(), 'description'):
            source = model.get_spec().description.metadata.source
            if source:
                model_path = source
        elif hasattr(model, 'origin_path'):
            model_path = model.origin_path
    except:
        model_path = None
    
    # Get sizes from model path if available
    if model_path:
        try:
            model_size_bytes = get_model_size(model_path)
            weights_size_bytes = get_model_size(model_path, weights_only=True)
        except Exception as e:
            print(f"Warning: Unable to get model size: {e}")
            # For LM head models, estimate based on hidden size
            if hidden_size and "vocab_size" in metadata:
                vocab_size = metadata["vocab_size"]
                # Rough estimation: hidden_size * vocab_size * 2 bytes (float16)
                model_size_bytes = hidden_size * vocab_size * 2
                weights_size_bytes = model_size_bytes  # Assume weights are dominant
            else:
                # Very rough fallback estimation
                model_size_bytes = 100 * 1024 * 1024  # Assume 100MB
                weights_size_bytes = 80 * 1024 * 1024  # Assume 80% are weights
    else:
        # If we can't get the model path, use metadata to estimate
        if "model_size_bytes" in metadata:
            model_size_bytes = metadata["model_size_bytes"]
            weights_size_bytes = metadata.get("weights_size_bytes", int(model_size_bytes * 0.8))
        elif hidden_size and "vocab_size" in metadata:
            vocab_size = metadata["vocab_size"]
            # Rough estimation: hidden_size * vocab_size * 2 bytes (float16)
            model_size_bytes = hidden_size * vocab_size * 2
            weights_size_bytes = model_size_bytes  # For LMs, weights are dominant
        else:
            # Very rough fallback estimation
            model_size_bytes = 100 * 1024 * 1024  # Assume 100MB
            weights_size_bytes = 80 * 1024 * 1024  # Assume 80% are weights
    
    # Calculate I/O size
    io_bytes = input_size_bytes + output_size_bytes
    
    # Total data size is the model weights plus I/O
    # Use weights size rather than total model size for throughput calculation
    total_data_bytes = weights_size_bytes + io_bytes
    
    # Calculate throughput based on the weights and I/O size
    # We only consider the actual weights and I/O that need to be read during inference
    avg_inference_time_s = avg_time / 1000.0  # Convert ms to seconds
    
    throughput_gb_s = 0
    if avg_inference_time_s > 0:
        throughput_gb_s = (total_data_bytes / 1e9) / avg_inference_time_s  # Gigabytes per second
    
    # Calculate TFLOPS only if known_tflops is provided
    if known_tflops is not None:
        # The provided value represents total TFLOPs per iteration
        # We need to calculate TFLOPS (TFLOPs per second) by dividing by the actual time
        print(f"Using provided TFLOPs per iteration: {known_tflops}")
        tflops = known_tflops / avg_inference_time_s  # Convert to TFLOPS by dividing by seconds
    else:
        # Don't calculate TFLOPS when not explicitly requested
        tflops = None
    
    # Create results dictionary
    results = {
        "avg_inference_time_ms": avg_time,
        "median_inference_time_ms": median_time,
        "min_inference_time_ms": min_time,
        "p99_inference_time_ms": p99_time,
        "avg_inference_time_s": avg_inference_time_s,
        "input_bytes": input_size_bytes,
        "output_bytes": output_size_bytes,
        "io_bytes": io_bytes,
        "io_mb": io_bytes / (1024 * 1024),
        "model_size_bytes": model_size_bytes,
        "model_size_mb": model_size_bytes / (1024 * 1024),
        "weights_size_bytes": weights_size_bytes,
        "weights_size_mb": weights_size_bytes / (1024 * 1024),
        "total_data_bytes": total_data_bytes,
        "total_data_mb": total_data_bytes / (1024 * 1024),
        "throughput_gb_s": throughput_gb_s,
        "tflops": tflops,
        "num_iterations": num_iterations,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "hidden_size": hidden_size,
        "model_path": model_path,
        "model_type": metadata.get("type", "unknown"),
    }
    
    return results


def benchmark_coreml_model_file(model_path: str, num_iterations: int = 100, 
                              batch_size: int = 1, sequence_length: int = 512,
                              hidden_size: Optional[int] = None, compute_units: str = "CPU_AND_NE",
                              known_tflops: Optional[float] = None) -> Dict[str, Any]:
    """
    Benchmark a CoreML model file directly
    
    Args:
        model_path: Path to the CoreML model
        num_iterations: Number of iterations for benchmarking
        batch_size: Batch size
        sequence_length: Sequence length
        hidden_size: Hidden size (optional, will be detected if possible)
        compute_units: Compute units to use
        known_tflops: Total trillion floating point operations per iteration (not TFLOPS rate)
        
    Returns:
        Dictionary containing benchmarking results
    """
    # Load the model
    model = load_coreml_model(model_path, compute_units)
    
    # Get metadata to help determine proper inputs
    metadata = get_model_metadata(model)
    
    # Check if this is an LM head model based on name or structure
    is_lm_head = False
    if "llama_lm_head" in model_path.lower() or "lm_head" in model_path.lower():
        is_lm_head = True
        print("Detected LM head model based on filename")
    
    if any(out.get("name", "").startswith("logits") for out in metadata.get("outputs", [])):
        is_lm_head = True
        print("Detected LM head model based on output names")
    
    # Try to determine hidden size if not provided
    if hidden_size is None:
        # If we can extract it from metadata
        if "hidden_size" in metadata:
            hidden_size = metadata["hidden_size"]
            print(f"Using detected hidden size: {hidden_size}")
        # Special case for llama_lm_head models 
        elif is_lm_head:
            # Try to infer from model name or fixed defaults
            if "llama2" in model_path.lower() or "llama-2" in model_path.lower():
                hidden_size = 4096  # Llama-2 7B/13B
                print(f"Using Llama-2 standard hidden size: {hidden_size}")
            elif "llama3" in model_path.lower() or "llama-3" in model_path.lower():
                hidden_size = 6656  # Llama-3 8B
                print(f"Using Llama-3 standard hidden size: {hidden_size}")
            else:
                # Detect from outputs if possible
                for out_info in metadata.get("outputs", []):
                    if out_info.get("name", "").startswith("logits") and out_info.get("shape"):
                        # For LM head, vocab size is in last dimension
                        vocab_size = out_info["shape"][-1]
                        # Rough heuristic: most Llama variants use these vocab sizes
                        if vocab_size > 32000:
                            hidden_size = 4096  # Llama-2
                        else:
                            hidden_size = 4096  # Default fallback
                        print(f"Inferred hidden size from vocab size: {hidden_size}")
                        break
                else:
                    hidden_size = 4096  # Default for most LLaMA variants
                    print(f"Using default LLaMA hidden size: {hidden_size}")
    
    # Extract exact input shape from model metadata if available
    exact_input_shapes = {}
    for input_info in metadata.get("inputs", []):
        name = input_info.get("name")
        shape = input_info.get("shape")
        if name and shape and len(shape) > 0:
            exact_input_shapes[name] = shape
            print(f"Found exact input shape for {name}: {shape}")
    
    # Prepare inputs
    try:
        if is_lm_head:
            print("Using LM head specialized input preparation")
            # For LM head models, check if we have exact shape for hidden_states
            if "hidden_states" in exact_input_shapes:
                # Use exact shape from model metadata
                exact_shape = exact_input_shapes["hidden_states"]
                print(f"Using exact shape from model: {exact_shape}")
                inputs = {"hidden_states": np.random.rand(*exact_shape).astype(np.float16)}
            else:
                # Use default shape but respect sequence length if specified in arguments
                inputs = {"hidden_states": np.random.rand(batch_size, 1, hidden_size or 4096).astype(np.float16)}
        elif any(info.get("name") == "input_ids" for info in metadata.get("inputs", [])) or \
           any(info.get("name") == "hidden_states" for info in metadata.get("inputs", [])):
            # Use specialized LM inputs
            inputs = prepare_lm_inputs(model, batch_size, sequence_length, hidden_size)
        else:
            # Use standard inputs
            inputs = prepare_inputs(model, batch_size, sequence_length, hidden_size or 2048)
            
        # Verify inputs are non-empty
        if not inputs:
            raise ValueError("Failed to generate inputs - empty dictionary returned")
            
    except Exception as e:
        print(f"Error preparing regular inputs: {e}")
        print("Falling back to hardcoded inputs based on model type")
        
        # Fallback: use hardcoded inputs based on model structure
        if is_lm_head:
            # For LM head models, check if we have exact shape info
            if "hidden_states" in exact_input_shapes:
                shape = exact_input_shapes["hidden_states"]
                inputs = {"hidden_states": np.random.rand(*shape).astype(np.float16)}
            else:
                # Fallback to default shape with sequence length 1
                inputs = {"hidden_states": np.random.rand(batch_size, 1, hidden_size or 4096).astype(np.float16)}
        else:
            # Generic fallback for any model
            inputs = {}
            # Try to extract input names from metadata
            input_names = [info.get("name") for info in metadata.get("inputs", []) if info.get("name")]
            
            if input_names:
                # Create a random tensor for each input
                for name in input_names:
                    if name in exact_input_shapes:
                        # Use exact shape
                        inputs[name] = np.random.rand(*exact_input_shapes[name]).astype(np.float16)
                    elif name.lower() == "hidden_states":
                        inputs[name] = np.random.rand(batch_size, 1, hidden_size or 4096).astype(np.float16)
                    elif name.lower() == "input_ids":
                        inputs[name] = np.random.randint(0, 1000, size=(batch_size, sequence_length)).astype(np.int32)
                    else:
                        # Generic input - try float16 first as it's common
                        inputs[name] = np.random.rand(batch_size, sequence_length, hidden_size or 2048).astype(np.float16)
            else:
                # Ultimate fallback - just try a standard input name
                inputs = {"input": np.random.rand(batch_size, sequence_length, hidden_size or 2048).astype(np.float16)}
    
    # Print the inputs we're using
    print(f"Using inputs: {', '.join([f'{k}:{v.shape}' for k, v in inputs.items()])}")
    
    # Profile the model
    results = profile_coreml_model(model, num_iterations, inputs, known_tflops=known_tflops)
    
    # Add model info
    results["model_path"] = model_path
    
    # Get the actual model size from the file
    actual_model_size_bytes = get_model_size(model_path)
    weights_size_bytes = get_model_size(model_path, weights_only=True)
    results["model_size_bytes"] = actual_model_size_bytes
    results["model_size_mb"] = actual_model_size_bytes / (1024 * 1024)
    results["weights_size_bytes"] = weights_size_bytes
    results["weights_size_mb"] = weights_size_bytes / (1024 * 1024)
    
    # Recalculate total data size with actual weights size and I/O
    results["total_data_bytes"] = weights_size_bytes + results["io_bytes"]
    results["total_data_mb"] = results["total_data_bytes"] / (1024 * 1024)
    
    # Update throughput based on weights and I/O size
    if results["avg_inference_time_s"] > 0:
        results["throughput_gb_s"] = (results["total_data_bytes"] / 1e9) / results["avg_inference_time_s"]
    
    results["compute_units"] = compute_units
    
    # Print results
    print(f"Average inference time: {results['avg_inference_time_ms']:.2f} ms")
    print(f"Throughput: {results['throughput_gb_s']:.2f} GB/s (based on weights + I/O)")
    if results['tflops'] is not None:
        print(f"TFLOPS: {results['tflops']:.4f}")
    print(f"Model size: {results['model_size_mb']:.2f} MB")
    print(f"Weights size: {results['weights_size_mb']:.2f} MB ({(results['weights_size_bytes']/results['model_size_bytes'])*100:.1f}% of total)")
    print(f"Input/Output size: {results['io_mb']:.2f} MB")
    print(f"Total data size (weights + I/O): {results['total_data_mb']:.2f} MB")
    
    return results


def get_model_metadata(model: Any) -> Dict[str, Any]:
    """
    Extract metadata from a CoreML model, including architecture details
    
    Args:
        model: CoreML model object
        
    Returns:
        Dictionary with model metadata
    """
    metadata = {}
    
    try:
        # Get the model spec
        spec = model.get_spec()
        
        # Get basic model info
        metadata["type"] = spec.WhichOneof("Type")
        
        # Try to extract model-specific information
        if hasattr(spec, "description") and hasattr(spec.description, "metadata"):
            if hasattr(spec.description.metadata, "userDefined"):
                for key, value in spec.description.metadata.userDefined.items():
                    metadata[key] = value
        
        # Check for mlprogram, which is used for newer transformer models
        if metadata.get("type") == "mlProgram":
            # Try to extract architecture details like hidden size
            # This is model-specific and may require parsing the model structure
            pass
            
        # Get input and output details
        metadata["inputs"] = []
        for input_spec in spec.description.input:
            # Get shape - safely handle different formats
            shape = None
            if hasattr(input_spec.type, "multiArrayType") and hasattr(input_spec.type.multiArrayType, "shape"):
                try:
                    shape_elements = input_spec.type.multiArrayType.shape
                    shape = []
                    # Handle both size attribute and direct int
                    for dim in shape_elements:
                        if hasattr(dim, 'size'):
                            shape.append(dim.size)
                        else:
                            # If it's a direct integer
                            shape.append(dim)
                except Exception as e:
                    print(f"Error extracting shape for {input_spec.name}: {e}")
            
            input_info = {
                "name": input_spec.name,
                "shape": shape,
                "data_type": input_spec.type.multiArrayType.dataType if hasattr(input_spec.type, "multiArrayType") else None
            }
            metadata["inputs"].append(input_info)
            
        metadata["outputs"] = []
        for output_spec in spec.description.output:
            # Get shape - safely handle different formats
            shape = None
            if hasattr(output_spec.type, "multiArrayType") and hasattr(output_spec.type.multiArrayType, "shape"):
                try:
                    shape_elements = output_spec.type.multiArrayType.shape
                    shape = []
                    # Handle both size attribute and direct int
                    for dim in shape_elements:
                        if hasattr(dim, 'size'):
                            shape.append(dim.size)
                        else:
                            # If it's a direct integer
                            shape.append(dim)
                except Exception as e:
                    print(f"Error extracting shape for {output_spec.name}: {e}")
                        
            output_info = {
                "name": output_spec.name,
                "shape": shape,
                "data_type": output_spec.type.multiArrayType.dataType if hasattr(output_spec.type, "multiArrayType") else None
            }
            metadata["outputs"].append(output_info)
            
        # Try to infer hidden size from input or output shapes for LM models
        # For many transformer models, output last dimension is vocabulary size
        # and hidden dimension can be inferred from intermediate layers
        for output_info in metadata["outputs"]:
            if output_info["name"].startswith("logits") and output_info["shape"] and len(output_info["shape"]) >= 2:
                metadata["vocab_size"] = output_info["shape"][-1]
                break
                
        for input_info in metadata["inputs"]:
            if input_info["name"] == "hidden_states" and input_info["shape"] and len(input_info["shape"]) >= 3:
                metadata["hidden_size"] = input_info["shape"][-1]
                break
        
    except Exception as e:
        print(f"Warning: Error extracting model metadata: {e}")
    
    return metadata


def prepare_lm_inputs(model: Any, batch_size: int = 1, sequence_length: int = 512, 
                     hidden_size: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Prepare inputs specifically for language models (LM head models)
    
    Args:
        model: CoreML model
        batch_size: Batch size
        sequence_length: Sequence length
        hidden_size: Hidden size (if known, otherwise inferred)
        
    Returns:
        Dictionary of input tensors
    """
    # Get model metadata to understand the input requirements
    metadata = get_model_metadata(model)
    
    # If hidden_size is not provided, try to get from metadata
    if hidden_size is None and "hidden_size" in metadata:
        hidden_size = metadata["hidden_size"]
        print(f"Using detected hidden_size: {hidden_size}")
    elif hidden_size is None:
        # Default fallbacks based on common architectures
        if "vocab_size" in metadata:
            vocab_size = metadata["vocab_size"]
            if vocab_size > 30000:  # GPT-2 and larger models
                hidden_size = 2048
            elif vocab_size > 20000:  # Mid-sized models
                hidden_size = 1024
            else:  # Smaller models
                hidden_size = 768
        else:
            # Default fallback
            hidden_size = 2048
            print(f"Warning: Could not determine hidden size. Using default: {hidden_size}")
    
    inputs = {}
    
    # Extract exact input shapes from metadata
    exact_input_shapes = {}
    for input_info in metadata.get("inputs", []):
        name = input_info.get("name")
        shape = input_info.get("shape")
        if name and shape and len(shape) > 0:
            exact_input_shapes[name] = shape
            print(f"Found exact input shape for {name}: {shape}")
    
    # Check for specific known input patterns
    has_input_ids = False
    has_hidden_states = False
    
    # Check which inputs the model expects
    for input_info in metadata.get("inputs", []):
        if input_info["name"] == "input_ids":
            has_input_ids = True
        elif input_info["name"] == "hidden_states":
            has_hidden_states = True
    
    # Generate appropriate inputs based on detected pattern
    if has_input_ids:
        # For models that take token IDs directly (like full LLMs)
        if "input_ids" in exact_input_shapes:
            # Use exact shape from metadata
            shape = exact_input_shapes["input_ids"]
            inputs["input_ids"] = np.random.randint(0, 1000, size=shape).astype(np.int32)
        else:
            # Use default shape
            inputs["input_ids"] = np.random.randint(0, 1000, size=(batch_size, sequence_length)).astype(np.int32)
        
        # Some models may also expect attention_mask
        if any(info["name"] == "attention_mask" for info in metadata.get("inputs", [])):
            if "attention_mask" in exact_input_shapes:
                shape = exact_input_shapes["attention_mask"]
                inputs["attention_mask"] = np.ones(shape, dtype=np.int32)
            else:
                inputs["attention_mask"] = np.ones((batch_size, sequence_length), dtype=np.int32)
            
    if has_hidden_states:
        # For models that take hidden states (like standalone LM heads)
        if "hidden_states" in exact_input_shapes:
            # Use exact shape from metadata
            shape = exact_input_shapes["hidden_states"]
            inputs["hidden_states"] = np.random.rand(*shape).astype(np.float16)
            print(f"Using exact input shape for hidden_states: {shape}")
        else:
            # Check if this is likely an LM head model, which often expects sequence length of 1
            is_lm_head = any(out.get("name", "").startswith("logits") for out in metadata.get("outputs", []))
            if is_lm_head:
                # LM head models often expect a single token embedding
                inputs["hidden_states"] = np.random.rand(batch_size, 1, hidden_size).astype(np.float16)
                print(f"Using sequence length of 1 for LM head model")
            else:
                # Use standard sequence length
                inputs["hidden_states"] = np.random.rand(batch_size, sequence_length, hidden_size).astype(np.float16)
    
    # If we couldn't determine inputs using patterns, fall back to default logic
    if not inputs:
        return prepare_inputs(model, batch_size, sequence_length, hidden_size)
    
    return inputs


def estimate_flops_for_model(model: Any, inputs: Dict[str, np.ndarray]) -> float:
    """
    Estimate FLOPS for a specific model based on its architecture
    
    Args:
        model: CoreML model
        inputs: Input tensors
        
    Returns:
        Estimated FLOPS 
    """
    metadata = get_model_metadata(model)
    
    # Get some basic parameters
    vocab_size = metadata.get("vocab_size", 0)
    hidden_size = metadata.get("hidden_size", 0)
    
    # If we have hidden_states input, we can estimate from that
    if "hidden_states" in inputs:
        hidden_states = inputs["hidden_states"]
        batch_size, seq_len, d_model = hidden_states.shape
        
        # For a typical LM head, the computation is mainly the final projection
        # from hidden_size to vocab_size, which is approximately:
        # hidden_size * vocab_size * seq_len operations
        if vocab_size and hidden_size:
            return 2 * hidden_size * vocab_size * seq_len * batch_size
    
    # If we have input_ids, this is likely a full transformer model
    if "input_ids" in inputs:
        input_ids = inputs["input_ids"]
        batch_size, seq_len = input_ids.shape
        
        # Very rough estimation for a transformer model
        # This is highly model-dependent and should be refined
        if vocab_size and hidden_size:
            # Assume 24 layers for large models, 12 for mid-size, 6 for small
            num_layers = 24 if hidden_size >= 1024 else (12 if hidden_size >= 768 else 6)
            # Each attention layer: 4 * hidden_size^2 * seq_len
            # Each FFN layer: 8 * hidden_size^2 * seq_len
            # Total per layer: ~12 * hidden_size^2 * seq_len
            return 2 * 12 * num_layers * hidden_size**2 * seq_len * batch_size
    
    # Default fallback using model size-based estimation
    try:
        spec = model.get_spec()
        if hasattr(spec, "description") and hasattr(spec.description, "metadata"):
            if hasattr(spec.description.metadata, "userDefined") and "params" in spec.description.metadata.userDefined:
                params = float(spec.description.metadata.userDefined["params"])
                # Assume each parameter is used ~2 times
                return 2 * params
    except:
        pass
    
    # Ultimate fallback: very rough estimation based on input size
    total_input_elements = sum(np.prod(arr.shape) for arr in inputs.values())
    return total_input_elements * 1000  # Very rough multiplier 