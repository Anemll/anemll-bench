"""
Unit tests for benchmark module
"""

import pytest
import os
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from anemll_bench import Benchmark
from anemll_bench.utils.system_info import get_system_info


class SimpleMLP(torch.nn.Module):
    """Simple MLP model for testing"""
    
    def __init__(self, input_size=128, hidden_size=64, output_size=10):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def test_benchmark_initialization():
    """Test benchmark initialization"""
    benchmark = Benchmark()
    assert benchmark is not None
    assert hasattr(benchmark, 'results')
    assert len(benchmark.results) == 0
    assert benchmark.system_info is not None


def test_system_info():
    """Test system information collection"""
    system_info = get_system_info()
    assert system_info is not None
    assert 'os' in system_info
    assert 'cpu' in system_info
    assert 'ram' in system_info
    assert 'python_version' in system_info
    
    if system_info['os']['name'] == 'Darwin':
        assert 'mac_model' in system_info
        

def test_benchmark_model_cpu():
    """Test benchmarking a model on CPU"""
    # Skip on CI if needed
    if os.environ.get('CI') == 'true':
        pytest.skip("Skipping benchmark test in CI environment")
    
    # Create a simple model
    model = SimpleMLP()
    
    # Initialize benchmark
    benchmark = Benchmark()
    
    # Benchmark on CPU
    result = benchmark.benchmark_model(
        model=model,
        model_name="TestMLP",
        input_shape=[1, 128],
        backend="CPU",
        num_runs=10  # Small number for quick testing
    )
    
    # Check results
    assert result is not None
    assert result.model_name == "TestMLP"
    assert result.backend == "CPU"
    assert result.inference_time_ms > 0
    assert result.memory_used_mb >= 0
    
    # Check TFLOPS calculation
    assert result.tflops is not None
    assert result.tflops >= 0
    
    # Check throughput calculation
    assert result.throughput_gbps is not None
    assert result.throughput_gbps >= 0


def test_generate_report():
    """Test report generation"""
    # Create a simple model and get benchmark results
    model = SimpleMLP()
    benchmark = Benchmark()
    
    # Benchmark on CPU (quick run)
    benchmark.benchmark_model(
        model=model,
        model_name="TestMLP",
        input_shape=[1, 128],
        backend="CPU",
        num_runs=5
    )
    
    # Generate report
    test_report_path = "test_report.html"
    report_html = benchmark.generate_report(test_report_path)
    
    # Check report was created
    assert os.path.exists(test_report_path)
    assert report_html is not None
    assert len(report_html) > 0
    
    # Clean up
    os.remove(test_report_path)


if __name__ == "__main__":
    # Run tests manually
    test_benchmark_initialization()
    test_system_info()
    test_benchmark_model_cpu()
    test_generate_report()
    print("All tests passed!") 