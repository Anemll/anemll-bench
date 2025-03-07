"""
Main benchmarking module for Apple Neural Engine performance testing
"""

import time
import os
import platform
import json
import psutil
import torch
import numpy as np
from tqdm import tqdm
import coremltools as ct
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
from anemll_bench.utils.system_info import get_system_info
from anemll_bench.reports.report_generator import generate_report
from anemll_bench.models.model_loader import load_model, convert_to_coreml
from anemll_bench.reports.report_uploader import ReportUploader
from anemll_bench.models.coreml_adapter import (
    load_coreml_model, 
    profile_coreml_model, 
    prepare_inputs, 
    get_model_size,
    benchmark_coreml_model_file
)
from anemll_bench.models.benchmark_result import BenchmarkResult


class Benchmark:
    """Benchmark class for measuring Apple Neural Engine performance"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize benchmark with optional config file"""
        self.config = {}
        self.results = []
        self.system_info = get_system_info()
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
    
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
    
    def _calculate_throughput(self, input_size_bytes: float, time_ms: float) -> float:
        """Calculate throughput in GB/s"""
        # Assuming input and output are roughly the same size for simplicity
        total_bytes = input_size_bytes * 2
        return (total_bytes / (time_ms / 1000)) / 1e9
    
    def benchmark_model(self, model, model_name: str, input_shape: List[int], 
                       backend: str = 'ANE', num_runs: int = 100):
        """Benchmark a single model and return results"""
        print(f"Starting benchmark of {model_name} on {backend}...")
        
        # Create dummy input based on shape
        input_size_bytes = np.prod(input_shape) * 4  # Assuming float32 (4 bytes)
        
        if backend == 'ANE':
            # For ANE, we need a CoreML model
            if not isinstance(model, ct.models.MLModel) and not isinstance(model, ct.models.CompiledMLModel):
                print("Converting PyTorch model to CoreML for ANE benchmarking...")
                coreml_path = convert_to_coreml(model, tuple(input_shape), model_name)
                model = ct.models.MLModel(coreml_path)
                
            # Use the coreml_adapter for more accurate profiling
            if isinstance(model, (ct.models.MLModel, ct.models.CompiledMLModel)):
                # Extract sequence length from input shape
                seq_length = input_shape[1] if len(input_shape) > 1 else 512
                hidden_size = input_shape[2] if len(input_shape) > 2 else 2048
                batch_size = input_shape[0] if len(input_shape) > 0 else 1
                
                # Use the direct profiling function
                profile_results = profile_coreml_model(
                    model, 
                    num_iterations=num_runs,
                    batch_size=batch_size,
                    sequence_length=seq_length,
                    hidden_size=hidden_size
                )
                
                # Create a BenchmarkResult from the profile results
                result = BenchmarkResult(
                    model_name=model_name,
                    input_shape=input_shape,
                    params_count=int(profile_results.get("params_estimate", 0)),
                    backend=backend,
                    inference_time_ms=profile_results["avg_inference_time_ms"],
                    memory_used_mb=0,  # Memory usage tracking not implemented for CoreML models
                    tflops=profile_results["tflops"],
                    throughput_gb_s=profile_results["throughput_gb_s"],
                    system_info=self.system_info
                )
                
                self.results.append(result)
                
                print(f"Benchmark complete: {model_name} on {backend}")
                print(f"  - Avg inference time: {result.inference_time_ms:.2f} ms")
                print(f"  - Estimated TFLOPS: {result.tflops:.4f}")
                print(f"  - Throughput: {result.throughput_gb_s:.2f} GB/s")
                
                return result
            else:
                # Create a random input within the expected range
                input_data = {
                    'input': np.random.rand(*input_shape).astype(np.float32)
                }
        else:
            # PyTorch input
            dummy_input = torch.randn(input_shape)
        
        # Estimate parameters count
        params_count = 0
        if hasattr(model, 'parameters'):
            params_count = sum(p.numel() for p in model.parameters())
        
        # Estimate FLOPS
        estimated_flops = self._estimate_flops(model, input_shape)
        
        # Warmup runs
        print("Performing warmup runs...")
        for _ in range(10):
            if backend == 'ANE':
                # CoreML inference
                _ = model.predict(input_data)
            else:
                # PyTorch CPU/GPU inference
                with torch.no_grad():
                    _ = model(dummy_input)
        
        # Measure memory before
        mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        
        # Benchmark runs
        times = []
        print(f"Running {num_runs} benchmark iterations...")
        for _ in tqdm(range(num_runs), desc=f"Benchmarking {model_name} on {backend}"):
            start_time = time.time()
            
            if backend == 'ANE':
                # CoreML inference (runs on ANE if available)
                _ = model.predict(input_data)
            else:
                # PyTorch CPU/GPU inference
                with torch.no_grad():
                    _ = model(dummy_input)
            
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Measure memory after
        mem_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        
        # Calculate stats
        avg_time = np.mean(times)
        memory_used = mem_after - mem_before
        
        # Calculate TFLOPS and GB/s
        tflops = self._calculate_tflops(estimated_flops, avg_time)
        throughput = self._calculate_throughput(input_size_bytes, avg_time)
        
        # Create result object
        result = BenchmarkResult(
            model_name=model_name,
            input_shape=input_shape,
            params_count=params_count,
            backend=backend,
            inference_time_ms=avg_time,
            memory_used_mb=memory_used,
            tflops=tflops,
            throughput_gb_s=throughput,
            system_info=self.system_info
        )
        
        self.results.append(result)
        print(f"Benchmark complete: {model_name} on {backend}")
        print(f"  - Avg inference time: {avg_time:.2f} ms")
        print(f"  - Memory used: {memory_used:.2f} MB")
        print(f"  - Estimated TFLOPS: {tflops:.4f}")
        print(f"  - Throughput: {throughput:.2f} GB/s")
        
        return result
    
    def benchmark_coreml_file(self, model_path: str, model_name: Optional[str] = None, 
                             num_runs: int = 100, batch_size: int = 1, 
                             sequence_length: int = 512, hidden_size: int = 768,
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
            input_shape=[batch_size, sequence_length, hidden_size],
            params_count=int(results.get("params_estimate", 0)),
            backend=backend,
            inference_time_ms=results["avg_inference_time_ms"],
            memory_used_mb=0,  # Memory usage tracking not implemented for direct file benchmarking
            tflops=results["tflops"],
            throughput_gb_s=results["throughput_gb_s"],
            system_info=self.system_info
        )
        
        self.results.append(result)
        return result
    
    def generate_report(self, output_path: str = "benchmark_report.html", upload: bool = False, upload_service: str = "jsonbin", include_charts: bool = False):
        """
        Generate an HTML report from benchmark results
        
        Args:
            output_path: Path to save the HTML report
            upload: Whether to upload the report to a service
            upload_service: Service to upload to (gist, pastebin, jsonbin)
            include_charts: Whether to include performance charts in the report
        
        Returns:
            URL to the uploaded report if upload=True, None otherwise
        """
        if not self.results:
            print("No benchmark results to report.")
            return None
        
        # Generate the HTML report
        html_report = generate_report(self.results, output_path, include_charts)
        print(f"Report generated at {output_path}")
        
        # Upload if requested
        if upload:
            try:
                # Prepare data for upload
                report_data = {
                    "results": [vars(result) for result in self.results],
                    "system_info": self.system_info,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "html": html_report  # Include the HTML report
                }
                
                # Upload the report
                uploader = ReportUploader(service=upload_service)
                url = uploader.upload(report_data)
                print(f"Report uploaded to: {url}")
                return url
                
            except Exception as e:
                print(f"Error uploading report: {e}")
        
        return None
    
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
                actual_model_name = model_info.get('name', model_name or model_id or os.path.basename(model_path))
                
                # Run benchmarks on specified backends
                for backend in backends:
                    try:
                        self.benchmark_model(
                            model=model,
                            model_name=actual_model_name,
                            input_shape=input_shape,
                            backend=backend,
                            num_runs=model_config.get('num_runs', 100)
                        )
                    except Exception as e:
                        print(f"Error benchmarking {actual_model_name} on {backend}: {e}")
                
            except Exception as e:
                print(f"Error loading model {model_config.get('name', model_config.get('id', 'unknown'))}: {e}")
        
        return self.results 