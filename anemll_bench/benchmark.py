"""
Benchmarking utilities for ANEMLL
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import platform
import logging
import importlib
from datetime import datetime
from tqdm import tqdm
import psutil
from dataclasses import dataclass, field

try:
    import coremltools as ct
except ImportError:
    print("CoreML Tools not found. Some functionality may be limited.")
    
try:
    import torch
except ImportError:
    print("PyTorch not found. Some functionality may be limited.")

# Internal imports
from anemll_bench.utils.system_info import get_system_info
from anemll_bench.models.model_syncer import ModelSyncer
from anemll_bench.models.model_loader import load_model, convert_to_coreml
from anemll_bench.models.coreml_adapter import profile_coreml_model, benchmark_coreml_model_file
from anemll_bench.reports.report_generator import generate_report
from anemll_bench.reports.report_uploader import ReportUploader
from anemll_bench.models.benchmark_result import BenchmarkResult


class Benchmark:
    """Benchmark class for measuring Apple Neural Engine performance"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the benchmarking tool.
        
        Args:
            config_path: Path to a configuration file
        """
        self.results = []
        self.config_path = config_path
        self.config = {}
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                
        self.system_info = get_system_info()
    
    def _estimate_flops(self, model, input_shape: List[int]) -> float:
        """Estimate the number of FLOPS for a model with given input shape"""
        # This is a simplified estimation and should be refined for each model architecture
        if hasattr(model, 'parameters'):
            # For PyTorch models
            params = sum(p.numel() for p in model.parameters())
            # Rough estimate: each parameter is used twice on average per inference
            return params * 2
        elif hasattr(model, 'compute_flops'):
            # If the model has a method to compute FLOPS
            return model.compute_flops(input_shape)
        else:
            # Very rough estimate based on input size
            input_size = np.prod(input_shape)
            return input_size * 1000  # Arbitrary multiplier
    
    def _calculate_tflops(self, flops: float, time_ms: float) -> float:
        """Calculate TFLOPS (Tera Floating Point Operations Per Second)"""
        # Convert ms to seconds and FLOPS to TFLOPS
        return (flops / (time_ms / 1000)) / 1e12
    
    def _calculate_throughput(self, model_size_bytes: float, time_ms: float) -> float:
        """
        Calculate throughput in GB/s based on model size and inference time.
        
        For neural network inference, the throughput primarily measures how quickly
        the model can access its weights during forward propagation.
        
        Args:
            model_size_bytes: Size of the model weights in bytes
            time_ms: Inference time in milliseconds
            
        Returns:
            Throughput in GB/s
        """
        # Convert ms to seconds
        time_s = time_ms / 1000
        
        # Convert bytes to GB
        model_size_gb = model_size_bytes / 1e9
        
        # Calculate GB/s: how much model weight data is processed per second
        return model_size_gb / time_s
    
    def _get_model_size_bytes(self, model) -> float:
        """
        Get the total size of a CoreML model's weights and parameters in bytes.
        Works with both .mlmodelc and .mlpackage formats.
        
        Args:
            model: A CoreML model instance
            
        Returns:
            Size in bytes
        """
        import os
        import re
        import traceback
        import glob
        
        # Print model type for debugging
        print(f"Model type: {type(model)}")
        
        # Try to get the model path
        model_path = None
        
        # Method 1: Try to get path directly from model attributes
        if hasattr(model, 'path'):
            model_path = model.path
            print(f"Found path attribute: {model_path}")
        elif hasattr(model, '_spec') and hasattr(model._spec, 'path'):
            model_path = model._spec.path
            print(f"Found _spec.path attribute: {model_path}")
        elif hasattr(model, '_internal_model') and hasattr(model._internal_model, 'path'):
            model_path = model._internal_model.path
            print(f"Found _internal_model.path attribute: {model_path}")
        
        # Method 2: Try to find the path in the string representation of the model
        if not model_path:
            model_str = str(model)
            print(f"Model string representation: {model_str[:200]}...")  # Print first 200 chars
            
            # Look for path patterns in the string
            path_patterns = [
                r"path='(.*?)'",
                r"path=\"(.*?)\"",
                r"filename='(.*?)'",
                r"filename=\"(.*?)\"",
                r"mlmodelc_path='(.*?)'",
                r"mlmodelc_path=\"(.*?)\"",
                r"mlpackage_path='(.*?)'",
                r"mlpackage_path=\"(.*?)\""
            ]
            
            for pattern in path_patterns:
                matches = re.findall(pattern, model_str)
                if matches and os.path.exists(matches[0]):
                    model_path = matches[0]
                    print(f"Found path in string representation: {model_path}")
                    break
        
        # Method 3: Check the loaded model's cache directory
        if not model_path:
            # Look in the model cache directory for models matching the pattern
            potential_paths = []
            for format_type in ['.mlmodelc', '.mlpackage']:
                potential_paths.extend(glob.glob(os.path.join(os.environ.get('HOME', ''), '.cache/anemll-bench/models', f'*{format_type}')))
            
            # Try to match the path to the model
            if potential_paths:
                for path in potential_paths:
                    print(f"Checking potential path: {path}")
                    # We'll choose the most recent one as a fallback
                    if not model_path or os.path.getmtime(path) > os.path.getmtime(model_path):
                        model_path = path
                
                print(f"Using most recent model path from cache: {model_path}")
        
        # Method 4: Direct approach for known model formats
        if not model_path:
            # Get model name from the object if possible
            model_name = None
            if hasattr(model, 'name'):
                model_name = model.name
            elif hasattr(model, '_spec') and hasattr(model._spec, 'description') and hasattr(model._spec.description, 'metadata') and hasattr(model._spec.description.metadata, 'shortDescription'):
                model_name = model._spec.description.metadata.shortDescription
            
            # Try standard paths in the cache directory
            if model_name:
                for format_type in ['.mlpackage', '.mlmodelc']:
                    test_path = os.path.join(os.environ.get('HOME', ''), '.cache/anemll-bench/models', f"{model_name}{format_type}")
                    if os.path.exists(test_path):
                        model_path = test_path
                        print(f"Found model at standard path: {model_path}")
                        break
            
            # Hardcoded approach for the specific models we know exist
            llama_lm_head_path = os.path.join(os.environ.get('HOME', ''), '.cache/anemll-bench/models/llama_lm_head.mlpackage')
            if not model_path and os.path.exists(llama_lm_head_path):
                model_path = llama_lm_head_path
                print(f"Using hardcoded path for llama_lm_head: {model_path}")
                
            llama_lm_head_lut6_path = os.path.join(os.environ.get('HOME', ''), '.cache/anemll-bench/models/llama_lm_head_lut6.mlpackage')
            if not model_path and os.path.exists(llama_lm_head_lut6_path):
                model_path = llama_lm_head_lut6_path
                print(f"Using hardcoded path for llama_lm_head_lut6: {model_path}")
                
            llama_lm_head_mlmodelc_path = os.path.join(os.environ.get('HOME', ''), '.cache/anemll-bench/models/llama_lm_head.mlmodelc')
            if not model_path and os.path.exists(llama_lm_head_mlmodelc_path):
                model_path = llama_lm_head_mlmodelc_path
                print(f"Using hardcoded path for llama_lm_head (mlmodelc): {model_path}")
        
        # Final check if we have a valid path
        if not model_path or not os.path.exists(model_path):
            print(f"Warning: Could not determine model path, using estimation from input shape")
            
            # Try to print some useful debugging info
            print("Available model attributes:")
            for attr in dir(model):
                if not attr.startswith('__'):
                    try:
                        value = getattr(model, attr)
                        print(f"  - {attr}: {str(value)[:100]}")
                    except:
                        print(f"  - {attr}: <error accessing attribute>")
                        
            return 0
            
        # For .mlmodelc format
        if model_path.endswith('.mlmodelc'):
            weights_dir = os.path.join(model_path, 'weights')
            if os.path.exists(weights_dir):
                total_size = 0
                for root, _, files in os.walk(weights_dir):
                    for file in files:
                        if file.endswith('.bin'):
                            file_path = os.path.join(root, file)
                            size = os.path.getsize(file_path)
                            print(f"Found weight file: {file_path}, size: {size / 1e6:.2f} MB")
                            total_size += size
                return total_size
                
        # For .mlpackage format
        elif model_path.endswith('.mlpackage'):
            weights_dir = os.path.join(model_path, 'Data', 'com.apple.CoreML', 'weights')
            if os.path.exists(weights_dir):
                total_size = 0
                for root, _, files in os.walk(weights_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        size = os.path.getsize(file_path)
                        print(f"Found weight file: {file_path}, size: {size / 1e6:.2f} MB")
                        total_size += size
                return total_size
                
        # Fallback: calculate directory size
        total_size = 0
        for root, _, files in os.walk(model_path):
            for file in files:
                file_path = os.path.join(root, file)
                size = os.path.getsize(file_path)
                total_size += size
                
        print(f"Total model size: {total_size / 1e6:.2f} MB")
        return total_size
    
    def benchmark_model(self, model, model_name: str, input_shape: List[int], 
                       backend: str = 'ANE', num_runs: int = 300, input_name: str = None):
        """
        Benchmark a model.
        
        Args:
            model: The model to benchmark
            model_name: Name of the model
            input_shape: Shape of the input tensor
            backend: Backend to use ('CPU', 'GPU', 'ANE')
            num_runs: Number of runs to average (default: 300)
            input_name: Name of the input tensor (if None, will use default based on model type)
            
        Returns:
            BenchmarkResult with benchmark metrics
        """
        try:
            # Import appropriate backends
            import coremltools as ct
            
            # Extract batch size, sequence length, hidden size
            batch_size = input_shape[0] if len(input_shape) > 0 else 1
            sequence_length = input_shape[1] if len(input_shape) > 1 else 1
            hidden_size = input_shape[2] if len(input_shape) > 2 else 1
            
            # Create a random input tensor with the right shape
            import numpy as np
            input_data = np.random.rand(*input_shape).astype(np.float32)
            
            # Determine input name if not provided
            if input_name is None:
                # Try to get input name from model spec
                try:
                    spec = model.get_spec().description.input
                    if spec and len(spec) > 0:
                        input_name = spec[0].name
                    else:
                        # Default fallback input names
                        input_name = "input_ids"  # Common for transformer models
                except:
                    # Check if this is likely an LM head model based on name
                    if "lm_head" in model_name.lower():
                        input_name = "hidden_states"  # Common for LM head models
                    else:
                        input_name = "input_ids"  # Common default
            
            print(f"Using input name: {input_name}")
            
            # Create input dictionary
            inputs = {input_name: input_data}
            
            # Map backend to compute units
            compute_units_map = {
                "CPU": ct.ComputeUnit.CPU_ONLY,
                "GPU": ct.ComputeUnit.CPU_AND_GPU,
                "ANE": ct.ComputeUnit.CPU_AND_NE,
                "ALL": ct.ComputeUnit.ALL
            }
            compute_unit = compute_units_map.get(backend, ct.ComputeUnit.CPU_AND_NE)
            
            # Set compute unit if the model supports it
            if hasattr(model, 'compute_unit'):
                model.compute_unit = compute_unit
            
            # Warm up
            print(f"Warming up model...")
            for _ in range(5):
                _ = model.predict(inputs)
            
            # Run benchmark
            print(f"Running {num_runs} iterations for benchmarking...")
            start_time = time.time()
            
            # Run all iterations without progress updates
            for _ in range(num_runs):
                _ = model.predict(inputs)
                
            end_time = time.time()
            
            # Log completion
            elapsed_time = end_time - start_time
            print(f"Completed {num_runs} iterations in {elapsed_time:.2f} seconds ({elapsed_time/num_runs:.4f}s per iteration)")
            
            # Calculate metrics
            inference_time_ms = (elapsed_time * 1000) / num_runs
            
            # Calculate FLOPs
            flops = self._estimate_flops(model, input_shape)
            
            # Calculate throughput
            model_size_bytes = self._get_model_size_bytes(model)
            
            # Log the calculated sizes for debugging
            print(f"Model weights size: {model_size_bytes / 1e6:.2f} MB")
            
            # Calculate TFLOPs
            # Not calculating TFLOPS for now as the calculation is not accurate
            tflops = None  # Set to None to indicate it's not being calculated
            
            # Calculate throughput in GB/s
            throughput_gb_s = self._calculate_throughput(model_size_bytes, inference_time_ms)
            
            # Create benchmark result
            from .models.benchmark_result import BenchmarkResult
            
            result = BenchmarkResult(
                model_name=model_name,
                backend=backend,
                inference_time_ms=inference_time_ms,
                tflops=tflops,  # Set to None as we're not calculating it
                throughput_gb_s=throughput_gb_s,
                input_shape=input_shape,
                params_count=0,  # We don't calculate params count here
                memory_used_mb=0.0,  # Memory usage tracking not implemented
                system_info=self.system_info,
                model_size_mb=self._get_model_size_bytes(model) / 1e6  # Convert bytes to MB
            )
            
            # Print results - omit TFLOPs since they're not calculated correctly
            print(f"Benchmark results for {model_name} on {backend}:")
            print(f"  - Inference time: {inference_time_ms:.2f} ms")
            # Only print TFLOPs if provided externally (currently set to None)
            if tflops is not None:
                print(f"  - TFLOPs: {tflops:.2f}")
            print(f"  - Throughput: {throughput_gb_s:.2f} GB/s")
            
            # Add to results
            self._add_result_to_history(result)
            
            return result
            
        except Exception as e:
            print(f"Error benchmarking model {model_name}: {e}")
            raise
    
    def benchmark_coreml_file(self, model_path: str, model_name: Optional[str] = None, 
                             num_runs: int = 100, batch_size: int = 1, 
                             sequence_length: int = 512, hidden_size: int = 4096,
                             compute_units: str = "CPU_AND_NE", known_tflops: Optional[float] = None) -> BenchmarkResult:
        """
        Benchmark a CoreML model file directly
        
        Args:
            model_path: Path to the CoreML model
            model_name: Name for the model (defaults to filename if not provided)
            num_runs: Number of benchmark iterations
            batch_size: Batch size
            sequence_length: Sequence length
            hidden_size: Hidden size
            compute_units: Compute units to use (CPU_AND_NE, CPU_ONLY, ALL)
            known_tflops: Total trillion floating point operations per iteration (not TFLOPS rate)
            
        Returns:
            BenchmarkResult object with benchmark results
        """
        # Use filename as model name if not provided
        if model_name is None:
            model_name = os.path.basename(model_path)
        
        print(f"Benchmarking CoreML model file: {model_path}")
        
        # Map compute units to backend name for reporting
        backend_map = {
            "CPU_AND_NE": "ANE",
            "CPU_ONLY": "CPU",
            "ALL": "ANE"
        }
        
        backend = backend_map.get(compute_units, "ANE")
        
        # Use the CoreML adapter to benchmark
        results = benchmark_coreml_model_file(
            model_path=model_path,
            num_iterations=num_runs,
            batch_size=batch_size,
            sequence_length=sequence_length,
            hidden_size=hidden_size,
            compute_units=compute_units,
            known_tflops=known_tflops
        )
        
        # Create a BenchmarkResult
        result = BenchmarkResult(
            model_name=model_name,
            input_shape=results.get("primary_input_shape", [batch_size, sequence_length, hidden_size]),
            params_count=int(results.get("params_estimate", 0)),
            backend=backend,
            inference_time_ms=results["avg_inference_time_ms"],
            memory_used_mb=0,  # Memory usage tracking not implemented for direct file benchmarking
            tflops=results["tflops"],
            throughput_gb_s=results["throughput_gb_s"],
            system_info=self.system_info,
            model_size_mb=self._get_model_size_bytes(model) / 1e6  # Convert bytes to MB
        )
        
        self.results.append(result)
        return result
    
    def generate_report(self, output_path: str = "benchmark_report.html", upload: bool = False, upload_service: str = "jsonbin", include_charts: bool = False, auto_open: bool = True):
        """
        Generate an HTML report from benchmark results
        
        Args:
            output_path: Path to save the HTML report
            upload: Whether to upload the report to a sharing service
            upload_service: Service to use for uploading ('jsonbin', 'gist', 'pastebin')
            include_charts: Whether to include charts in the report
            auto_open: Whether to automatically open the report in the default browser
        """
        from anemll_bench.reports.report_generator import generate_report
        from anemll_bench.reports.report_uploader import ReportUploader
        
        if not self.results:
            print("No benchmark results to report")
            return
        
        # Generate the report
        report_html = generate_report(
            results=self.results,
            output_path=output_path,
            include_charts=include_charts
        )
        
        # Save the report (done by generate_report)
        
        # Open the report if requested
        if auto_open and os.path.exists(output_path):
            try:
                import webbrowser
                # Ensure we have an absolute path with the file:// protocol
                abs_path = os.path.abspath(output_path)
                file_url = f"file://{abs_path}"
                print(f"DEBUG: Attempting to open browser with URL: {file_url}")
                print(f"DEBUG: File exists: {os.path.exists(output_path)}")
                print(f"DEBUG: Absolute path: {abs_path}")
                
                # Try to use Safari specifically
                try:
                    # Try to use Safari
                    safari = webbrowser.get('safari')
                    result = safari.open(file_url)
                    print(f"DEBUG: Safari browser open result: {result}")
                except Exception as e:
                    print(f"DEBUG: Could not use Safari: {e}")
                    # Fall back to default browser
                    #result = webbrowser.open(file_url)
                    #print(f"DEBUG: Default browser open result: {result}")
                
                # Comment out the system 'open' command which is causing a second browser window
                # try:
                #     import subprocess
                #     subprocess.run(['open', abs_path], check=True)
                #     print(f"DEBUG: Opened with system 'open' command")
                # except Exception as e:
                #     print(f"DEBUG: Could not open with system command: {e}")
                
                print(f"Opened report in browser: {output_path}")
                print(f"Report URL: {file_url}")
            except Exception as e:
                print(f"Could not open report in browser: {e}")
                print(f"DEBUG: Exception details: {type(e).__name__}: {str(e)}")
                print(f"Report saved to: {output_path}")
        else:
            print(f"Report saved to: {output_path}")
        
        # Upload the report if requested
        if upload:
            try:
                uploader = ReportUploader(output_path)
                if upload_service == "jsonbin":
                    url = uploader.upload_to_jsonbin()
                elif upload_service == "gist":
                    url = uploader.upload_to_gist()
                elif upload_service == "pastebin":
                    url = uploader.upload_to_pastebin()
                else:
                    raise ValueError(f"Unknown upload service: {upload_service}")
                    
                print(f"Report uploaded to: {url}")
                return url
            except Exception as e:
                print(f"Failed to upload report: {e}")
        
        return output_path
    
    def run(self, models_config: Optional[List[Dict]] = None):
        """Run benchmarks on multiple models"""
        config = models_config or self.config.get('models', [])
        
        if not config:
            print("No models specified for benchmarking.")
            return self.results
        
        for model_config in config:
            try:
                model_name = model_config.get('name')
                model_id = model_config.get('id')
                model_path = model_config.get('path')
                input_shape = model_config.get('input_shape')
                backends = model_config.get('backends', ['CPU', 'ANE'])
                
                if not input_shape:
                    print(f"Warning: No input shape specified for {model_name or model_id or model_path}, skipping.")
                    continue
                
                # Load the model
                model, model_info = load_model(model_config)
                actual_model_name = model_info.get('name', model_name or model_id or model_path)
                
                # Run benchmarks on specified backends
                for backend in backends:
                    try:
                        self.benchmark_model(
                            model=model,
                            model_name=actual_model_name,
                            input_shape=input_shape,
                            backend=backend,
                            num_runs=model_config.get('num_runs', 300)
                        )
                    except Exception as e:
                        print(f"Error benchmarking {actual_model_name} on {backend}: {e}")
                
            except Exception as e:
                print(f"Error loading model {model_config.get('name', model_config.get('id', 'unknown'))}: {e}")
        
        return self.results
    
    def benchmark_platform_model(
        self,
        model_name: str,
        num_runs: int = 300,
        batch_size: int = 1,
        sequence_length: int = None,  # Set to None for auto-detection
        check_online: bool = True,
        force_redownload: bool = False,
        use_local_if_exists: bool = False,
    ) -> BenchmarkResult:
        """
        Benchmark a platform-specific model by name.
        
        Args:
            model_name (str): Model name to benchmark.
            num_runs (int): Number of runs to perform (default: 300).
            batch_size (int): Batch size to use.
            sequence_length (int): Sequence length to use (will be auto-detected for LM head models).
            check_online (bool): Whether to check online for a newer version.
            force_redownload (bool): Whether to force re-download the model.
            use_local_if_exists (bool): Whether to use a local model if it exists, even if corrupted.
            
        Returns:
            BenchmarkResult: Benchmark results.
        """
        from anemll_bench.models.model_loader import (
            load_platform_model_by_name,
            get_macos_version,
            download_and_unzip_model,
            read_meta_file,
            download_meta_file
        )
        from anemll_bench.models.model_syncer import ModelSyncer
        
        try:
            # Always use sequence_length of 1 for models with 'lm_head' in their name
            is_lm_head = 'lm_head' in model_name.lower()
            
            if is_lm_head and sequence_length is None:
                sequence_length = 1
                print(f"Auto-detected LM head model, using sequence_length=1")
            elif sequence_length is None:
                # Default sequence length for non-LM head models
                sequence_length = 512
                
            # Get hidden size from meta
            meta_data = download_meta_file() if check_online else read_meta_file()
            macos_version = get_macos_version()
            
            if 'model_info' not in meta_data or macos_version not in meta_data['model_info']:
                # Default to 4096 for hidden size if not found
                hidden_size = 4096
            else:
                # Find the model in the meta data
                model_meta = next((m for m in meta_data['model_info'][macos_version] if m.get('name') == model_name), None)
                hidden_size = model_meta.get('hidden_size', 4096) if model_meta else 4096
            
            if force_redownload:
                print(f"Forcing re-download of {model_name}...")
                # Initialize model syncer to download the model
                syncer = ModelSyncer()
                # Get model URL and type from meta
                if 'model_info' in meta_data and macos_version in meta_data['model_info']:
                    model_meta = next((m for m in meta_data['model_info'][macos_version] if m.get('name') == model_name), None)
                    if model_meta:
                        url = model_meta.get('url')
                        model_type = model_meta.get('type', 'unknown')
                        if url:
                            syncer.download_model(url, model_name, model_type, force_redownload=True, allow_redownload=check_online)
                
            # Try loading the model
            try:
                model, model_info = load_platform_model_by_name(
                    model_name, 
                    check_online=check_online, 
                    use_local_if_exists=use_local_if_exists
                )
                
                # If this is a proxy model, return a predefined result
                if hasattr(model, '_is_proxy_model') and model._is_proxy_model:
                    print(f"Detected proxy model for {model_name}")
                    print(f"Using pre-defined benchmark result for proxy model {model_name}")
                    print(f"  - Inference time: 15.00 ms")
                    # TFLOPs not calculated accurately, so not displaying
                    print(f"  - Throughput: 0.56 GB/s")
                    
                    return BenchmarkResult(
                        model_name=model_name,
                        model_type=model_info.get('type', 'unknown'),
                        backend="ANE",
                        input_shape=[batch_size, sequence_length, hidden_size],
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        latency_ms=15.0,
                        tflops=None,  # Set to None as we're not calculating it correctly
                        throughput_gb_s=0.56,
                        metadata={
                            "hidden_size": hidden_size,
                            "is_proxy_model": True,
                        }
                    )
                    
                # Get the input name for the model
                input_name = None
                if hasattr(model, 'input_names') and model.input_names:
                    input_name = model.input_names[0]
                    print(f"Using input name: {input_name}")
                
                # Try to determine the correct sequence_length from the model
                try:
                    if hasattr(model, 'get_spec') and model.get_spec().description.input[0].type.multiArrayType.shape:
                        # Extract the sequence length from the model specification
                        model_seq_length = model.get_spec().description.input[0].type.multiArrayType.shape[1]
                        if model_seq_length != -1:  # If not dynamic
                            sequence_length = model_seq_length
                            print(f"Using sequence length from model spec: {sequence_length}")
                except Exception as e:
                    print(f"Could not extract sequence length from model spec: {e}")
                    
                    # For LM head models, always force sequence_length to 1
                    if is_lm_head:
                        sequence_length = 1
                        print(f"Forcing sequence_length=1 for LM head model")
                
                # Extract input shape from the model if possible
                input_shape = [batch_size, sequence_length, hidden_size]
                
                # Log the input shape
                print(f"Using input shape: {input_shape}")
                
                # Benchmark the model
                print(f"Warming up model...")
                
                # Always perform benchmark with sequence_length of 1 for LM head models
                result = self.benchmark_model(
                    model=model, 
                    model_name=model_name, 
                    input_shape=input_shape,
                    backend="ANE", 
                    num_runs=num_runs,
                    input_name=input_name
                )
                return result
                
            except Exception as e:
                if check_online:
                    print(f"Error loading model {model_name}, attempting to recover with {'download enabled' if check_online else 'local-only mode'}: {e}")
                    if not force_redownload:
                        # Initialize model syncer to download the model
                        syncer = ModelSyncer()
                        # Get model URL and type from meta
                        if check_online and 'model_info' in meta_data and macos_version in meta_data['model_info']:
                            model_meta = next((m for m in meta_data['model_info'][macos_version] if m.get('name') == model_name), None)
                            if model_meta:
                                url = model_meta.get('url')
                                model_type = model_meta.get('type', 'unknown')
                                if url and check_online:
                                    print(f"Attempting to download model due to check_online={check_online}")
                                    syncer.download_model(url, model_name, model_type, force_redownload=True, allow_redownload=check_online)
                        
                        model, model_info = load_platform_model_by_name(model_name, check_online=check_online)
                        
                        # Get the input name for the model
                        input_name = None
                        if hasattr(model, 'input_names') and model.input_names:
                            input_name = model.input_names[0]
                        
                        # Force sequence_length to 1 for LM head models
                        if is_lm_head:
                            sequence_length = 1
                            print(f"Forcing sequence_length=1 for LM head model")
                        
                        input_shape = [batch_size, sequence_length, hidden_size]
                        print(f"Using input shape: {input_shape}")
                        
                        result = self.benchmark_model(
                            model=model, 
                            model_name=model_name, 
                            input_shape=input_shape,
                            backend="ANE", 
                            num_runs=num_runs,
                            input_name=input_name
                        )
                        return result
                    else:
                        raise
                else:
                    # If check_online is False, we can't download or recover the model
                    print(f"Error loading model {model_name} and --no-sync is enabled, cannot download: {e}")
                    # Return an error result without attempting to download
                    return BenchmarkResult(
                        model_name=model_name,
                        model_type="unknown",
                        backend="ANE",
                        input_shape=[batch_size, sequence_length if sequence_length else 512, hidden_size if hidden_size else 4096],
                        batch_size=batch_size,
                        sequence_length=sequence_length if sequence_length else 512,
                        latency_ms=0,
                        tflops=0,
                        throughput_gb_s=0,
                        metadata={
                            "error": f"Model could not be loaded and --no-sync was specified: {e}",
                        }
                    )
                
        except Exception as e:
            print(f"Error benchmarking model {model_name}: {e}")
            # Create a fake result with error message
            error_result = BenchmarkResult(
                model_name=model_name,
                model_type="unknown",
                backend="ANE",
                input_shape=[batch_size, sequence_length if sequence_length else 512, hidden_size if hidden_size else 4096],
                batch_size=batch_size,
                sequence_length=sequence_length if sequence_length else 512,
                latency_ms=0,
                tflops=0,
                throughput_gb_s=0,
                metadata={
                    "error": str(e),
                }
            )
            return error_result
    
    def benchmark_all_platform_models(self, num_runs: int = 300, batch_size: int = 1, 
                                  sequence_length: int = None,  # Changed from 512 to None to auto-detect
                                  sync_first: bool = True,
                                  include_charts: bool = True, output_path: Optional[str] = None,
                                  force_redownload: bool = False, auto_open: bool = True,
                                  use_local_if_exists: bool = True) -> List[BenchmarkResult]:
        """
        Benchmark all platform-specific models available for the current platform.
        
        Args:
            num_runs (int): Number of benchmark runs (default: 300).
            batch_size (int): Batch size for the benchmarks.
            sequence_length (int): Sequence length for the models. If None, will auto-detect from model.
            sync_first (bool): Whether to synchronize models before benchmarking.
            include_charts (bool): Whether to include charts in the report.
            output_path (str): Path to save the report to.
            force_redownload (bool): Whether to force re-download models.
            auto_open (bool): Whether to auto-open the report.
            use_local_if_exists (bool): Whether to use local models if they exist.
            
        Returns:
            List[BenchmarkResult]: List of benchmark results.
        """
        from anemll_bench.models.model_loader import sync_platform_models, get_platform_specific_models, get_macos_version
        import time
        
        # Get macOS version
        platform_category = get_macos_version()
        
        # Synchronize models if requested
        if sync_first:
            print("Syncing models...")
            # Call sync_platform_models with the correct parameter
            sync_platform_models(force_update=force_redownload)
        
        # Get all available models for this platform
        available_models = get_platform_specific_models(check_online=False)
        
        print(f"Benchmarking {len(available_models)} models for {platform_category}...")
        results = []
        
        # Benchmark each model
        for model_config in available_models:
            model_name = model_config.get('name')
            
            if model_name:
                print(f"Benchmarking model: {model_name}")
                
                try:
                    # Auto-detect sequence length for LM head models if not specified
                    current_sequence_length = sequence_length
                    
                    # For LM head models, always use sequence_length=1 if not specified
                    if "lm_head" in model_name.lower() and current_sequence_length is None:
                        current_sequence_length = 1
                        print(f"Auto-detected LM head model, using sequence_length=1")
                    
                    # Benchmark the model
                    result = self.benchmark_platform_model(
                        model_name=model_name,
                        num_runs=num_runs,
                        batch_size=batch_size,
                        sequence_length=current_sequence_length,  # Pass through the sequence_length parameter
                        check_online=False,  # We already synced
                        force_redownload=force_redownload,
                        use_local_if_exists=use_local_if_exists
                    )
                    
                    results.append(result)
                except Exception as e:
                    print(f"Error benchmarking model {model_name}: {e}")
        
        # If no results, return empty list
        if not results:
            print("No models were successfully benchmarked.")
            # Create a timestamp-based filename for the report
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_path = output_path or f"benchmark_report_{platform_category}_{timestamp}.html"
            
            # Generate an empty report
            self.generate_report(output_path=report_path, include_charts=include_charts, auto_open=auto_open)
            return []
        
        # Generate a report
        print("\nBenchmarking complete. Generated results for", len(results), "models.")
        
        # Create a timestamp-based filename for the report
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = output_path or f"benchmark_report_{platform_category}_{timestamp}.html"
        
        print(f"\nGenerating report: {report_path}")
        self.generate_report(output_path=report_path, include_charts=include_charts, auto_open=auto_open)
        
        return results

    def _add_result_to_history(self, result):
        """
        Add a benchmark result to the history.
        
        Args:
            result: The BenchmarkResult to add
        """
        self.results.append(result)

    def benchmark_dual_models(
        self,
        model1,
        model1_name: str,
        model1_input_shape: List[int],
        model2,
        model2_name: str,
        model2_input_shape: List[int],
        backend: str = 'ANE',
        num_runs: int = 300,
        model1_input_name: str = None,
        model2_input_name: str = None
    ):
        """
        Benchmark two different models simultaneously to measure potential bandwidth improvements.
        
        This function:
        1. First runs each model individually to get baseline performance
        2. Then runs both models simultaneously in separate threads
        3. Compares the performance to identify bandwidth changes
        
        Args:
            model1: First model to benchmark
            model1_name: Name of the first model
            model1_input_shape: Input shape for the first model
            model2: Second model to benchmark
            model2_name: Name of the second model
            model2_input_shape: Input shape for the second model
            backend: Backend to use ('CPU', 'GPU', 'ANE')
            num_runs: Number of runs for each benchmark (default: 300)
            model1_input_name: Input tensor name for model1 (default: None - auto-detect)
            model2_input_name: Input tensor name for model2 (default: None - auto-detect)
            
        Returns:
            Dict containing individual and combined benchmark results along with performance analysis
        """
        import threading
        import queue
        import numpy as np
        import coremltools as ct
        
        print(f"Starting dual model benchmark with {model1_name} and {model2_name}")
        
        # First benchmark each model individually for baseline
        print(f"\n=== Individual Benchmark: {model1_name} ===")
        result1 = self.benchmark_model(
            model=model1,
            model_name=model1_name,
            input_shape=model1_input_shape,
            backend=backend,
            num_runs=num_runs,
            input_name=model1_input_name
        )
        
        print(f"\n=== Individual Benchmark: {model2_name} ===")
        result2 = self.benchmark_model(
            model=model2,
            model_name=model2_name,
            input_shape=model2_input_shape,
            backend=backend,
            num_runs=num_runs,
            input_name=model2_input_name
        )
        
        # Map backend to compute units for setup
        compute_units_map = {
            "CPU": ct.ComputeUnit.CPU_ONLY,
            "GPU": ct.ComputeUnit.CPU_AND_GPU,
            "ANE": ct.ComputeUnit.CPU_AND_NE,
            "ALL": ct.ComputeUnit.ALL
        }
        compute_unit = compute_units_map.get(backend, ct.ComputeUnit.CPU_AND_NE)
        
        # Set compute unit for models if supported
        for model, name in [(model1, model1_name), (model2, model2_name)]:
            if hasattr(model, 'compute_unit'):
                model.compute_unit = compute_unit
                
        # Prepare input data for both models
        input_data1 = np.random.rand(*model1_input_shape).astype(np.float32)
        input_data2 = np.random.rand(*model2_input_shape).astype(np.float32)
        
        # Determine input names if not provided
        if model1_input_name is None:
            try:
                spec = model1.get_spec().description.input
                if spec and len(spec) > 0:
                    model1_input_name = spec[0].name
                else:
                    model1_input_name = "input_ids"
            except:
                model1_input_name = "input_ids"
        
        if model2_input_name is None:
            try:
                spec = model2.get_spec().description.input
                if spec and len(spec) > 0:
                    model2_input_name = spec[0].name
                else:
                    model2_input_name = "input_ids"
            except:
                model2_input_name = "input_ids"
                
        # Create input dictionaries
        inputs1 = {model1_input_name: input_data1}
        inputs2 = {model2_input_name: input_data2}
        
        # Warmup both models
        print(f"Warming up both models...")
        for _ in range(5):
            _ = model1.predict(inputs1)
            _ = model2.predict(inputs2)
            
        # Now run both models in parallel
        print(f"\n=== Parallel Benchmark: Running {model1_name} and {model2_name} simultaneously ===")
        
        # Create queues to capture results
        result_queue1 = queue.Queue()
        result_queue2 = queue.Queue()
        
        # Define worker functions
        def run_model1():
            start_time = time.time()
            for i in range(num_runs):
                _ = model1.predict(inputs1)
            end_time = time.time()
            elapsed_time = end_time - start_time
            inference_time_ms = (elapsed_time * 1000) / num_runs
            result_queue1.put({
                'elapsed_time': elapsed_time,
                'inference_time_ms': inference_time_ms,
                'iterations': num_runs
            })
            
        def run_model2():
            start_time = time.time()
            for i in range(num_runs):
                _ = model2.predict(inputs2)
            end_time = time.time()
            elapsed_time = end_time - start_time
            inference_time_ms = (elapsed_time * 1000) / num_runs
            result_queue2.put({
                'elapsed_time': elapsed_time,
                'inference_time_ms': inference_time_ms,
                'iterations': num_runs
            })
            
        # Create and start threads
        thread1 = threading.Thread(target=run_model1)
        thread2 = threading.Thread(target=run_model2)
        
        parallel_start_time = time.time()
        thread1.start()
        thread2.start()
        
        # Wait for both threads to complete
        thread1.join()
        thread2.join()
        parallel_end_time = time.time()
        
        # Get results from queues
        parallel_result1 = result_queue1.get()
        parallel_result2 = result_queue2.get()
        
        # Calculate the actual total elapsed time for parallel execution
        parallel_total_time = parallel_end_time - parallel_start_time
        
        # Calculate metrics for parallel runs
        model1_size_bytes = self._get_model_size_bytes(model1)
        model2_size_bytes = self._get_model_size_bytes(model2)
        
        # Calculate individual and combined throughputs
        parallel_throughput1_gb_s = self._calculate_throughput(model1_size_bytes, parallel_result1['inference_time_ms'])
        parallel_throughput2_gb_s = self._calculate_throughput(model2_size_bytes, parallel_result2['inference_time_ms'])
        
        # Combined throughput is the sum of data processed divided by the parallel execution time
        model1_total_data_gb = (model1_size_bytes / 1e9) * num_runs
        model2_total_data_gb = (model2_size_bytes / 1e9) * num_runs
        combined_throughput_gb_s = (model1_total_data_gb + model2_total_data_gb) / (parallel_total_time)
        
        # Calculate bandwidth utilization factor
        # (Compare sum of individual throughputs to the combined throughput)
        individual_throughput_sum = result1.throughput_gb_s + result2.throughput_gb_s
        bandwidth_utilization = combined_throughput_gb_s / individual_throughput_sum if individual_throughput_sum > 0 else 0
        
        # Create parallel BenchmarkResult objects
        from .models.benchmark_result import BenchmarkResult
        
        parallel_result1_obj = BenchmarkResult(
            model_name=f"{model1_name} (Parallel Run)",
            backend=backend,
            inference_time_ms=parallel_result1['inference_time_ms'],
            tflops=None,
            throughput_gb_s=parallel_throughput1_gb_s,
            input_shape=model1_input_shape,
            params_count=0,
            memory_used_mb=0.0,
            system_info=self.system_info,
            model_size_mb=model1_size_bytes / 1e6,
            notes="Run simultaneously with another model"
        )
        
        parallel_result2_obj = BenchmarkResult(
            model_name=f"{model2_name} (Parallel Run)",
            backend=backend,
            inference_time_ms=parallel_result2['inference_time_ms'],
            tflops=None,
            throughput_gb_s=parallel_throughput2_gb_s,
            input_shape=model2_input_shape,
            params_count=0,
            memory_used_mb=0.0,
            system_info=self.system_info,
            model_size_mb=model2_size_bytes / 1e6,
            notes="Run simultaneously with another model"
        )
        
        # Add results to history
        self._add_result_to_history(parallel_result1_obj)
        self._add_result_to_history(parallel_result2_obj)
        
        # Calculate average metrics for parallel runs
        avg_inference_time_ms = (parallel_result1['inference_time_ms'] + parallel_result2['inference_time_ms']) / 2
        avg_throughput_gb_s = (parallel_throughput1_gb_s + parallel_throughput2_gb_s) / 2
        
        # Create and add a combined result object to the history
        # For combined throughput, we use the total data processed by both models divided by the total time
        # For time, we use the max time of both models since they run in parallel
        max_parallel_time_ms = max(parallel_result1['inference_time_ms'], parallel_result2['inference_time_ms'])
        combined_result_obj = BenchmarkResult(
            model_name=f"Combined ({model1_name} + {model2_name})",
            backend=backend,
            inference_time_ms=max_parallel_time_ms,  # Use max time since models run in parallel
            tflops=None,
            throughput_gb_s=combined_throughput_gb_s,  # Use the combined throughput we calculated earlier
            input_shape=[],  # Not applicable for combined
            params_count=0,
            memory_used_mb=0.0,
            system_info=self.system_info,
            model_size_mb=(model1_size_bytes + model2_size_bytes) / 1e6,  # Combined size
            notes="Combined performance of both models running simultaneously"
        )
        self._add_result_to_history(combined_result_obj)
        
        # Prepare the performance summary
        performance_summary = {
            'individual_results': {
                model1_name: {
                    'inference_time_ms': result1.inference_time_ms,
                    'throughput_gb_s': result1.throughput_gb_s,
                    'model_size_mb': model1_size_bytes / 1e6
                },
                model2_name: {
                    'inference_time_ms': result2.inference_time_ms,
                    'throughput_gb_s': result2.throughput_gb_s,
                    'model_size_mb': model2_size_bytes / 1e6
                }
            },
            'parallel_results': {
                model1_name: {
                    'inference_time_ms': parallel_result1['inference_time_ms'],
                    'throughput_gb_s': parallel_throughput1_gb_s,
                    'slowdown_factor': parallel_result1['inference_time_ms'] / result1.inference_time_ms
                },
                model2_name: {
                    'inference_time_ms': parallel_result2['inference_time_ms'],
                    'throughput_gb_s': parallel_throughput2_gb_s,
                    'slowdown_factor': parallel_result2['inference_time_ms'] / result2.inference_time_ms
                },
                'average': {
                    'inference_time_ms': avg_inference_time_ms,
                    'throughput_gb_s': avg_throughput_gb_s
                }
            },
            'combined': {
                'total_parallel_time_s': parallel_total_time,
                'combined_throughput_gb_s': combined_throughput_gb_s,
                'individual_throughput_sum_gb_s': individual_throughput_sum,
                'bandwidth_utilization_factor': bandwidth_utilization,
                'efficiency': bandwidth_utilization * 100  # as percentage
            }
        }
        
        # Print detailed performance report
        print("\n=== Dual Model Benchmark Results ===")
        print(f"\nIndividual Performance:")
        print(f"  - {model1_name}: {result1.inference_time_ms:.2f} ms, {result1.throughput_gb_s:.2f} GB/s")
        print(f"  - {model2_name}: {result2.inference_time_ms:.2f} ms, {result2.throughput_gb_s:.2f} GB/s")
        
        print(f"\nParallel Performance:")
        print(f"  - {model1_name}: {parallel_result1['inference_time_ms']:.2f} ms, {parallel_throughput1_gb_s:.2f} GB/s")
        print(f"  - {model2_name}: {parallel_result2['inference_time_ms']:.2f} ms, {parallel_throughput2_gb_s:.2f} GB/s")
        print(f"  - Combined: {max_parallel_time_ms:.2f} ms, {combined_throughput_gb_s:.2f} GB/s (total throughput)")
        
        print(f"\nCombined Analysis:")
        print(f"  - Total Parallel Execution Time: {parallel_total_time:.2f} seconds")
        print(f"  - Combined Throughput: {combined_throughput_gb_s:.2f} GB/s")
        print(f"  - Sum of Individual Throughputs: {individual_throughput_sum:.2f} GB/s")
        print(f"  - Bandwidth Utilization Factor: {bandwidth_utilization:.2f}x")
        print(f"  - Efficiency: {bandwidth_utilization * 100:.2f}%")
        
        if bandwidth_utilization > 0.9:
            print("\nConclusion: High bandwidth utilization detected! Running models in parallel is efficient.")
        elif bandwidth_utilization > 0.7:
            print("\nConclusion: Good bandwidth utilization. Parallel execution provides moderate benefits.")
        else:
            print("\nConclusion: Low bandwidth utilization. Models might be competing for resources.")
            
        return {
            'individual_results': [result1, result2],
            'parallel_results': [parallel_result1_obj, parallel_result2_obj, combined_result_obj],
            'performance_summary': performance_summary
        } 