"""
BenchmarkResult class for storing benchmark results
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time


@dataclass
class BenchmarkResult:
    """Class for storing benchmark results"""
    model_name: str
    backend: str
    inference_time_ms: float
    input_shape: List[int]
    tflops: Optional[float] = None
    throughput_gb_s: Optional[float] = None
    params_count: int = 0
    memory_used_mb: float = 0.0
    system_info: Dict[str, Any] = field(default_factory=dict)
    model_size_mb: float = 0.0  # Model size in megabytes
    timestamp: float = field(default_factory=time.time)  # When the benchmark was run
    notes: str = ""  # Additional notes or context about the benchmark

    @property
    def throughput_gbps(self):
        """For backward compatibility"""
        return self.throughput_gb_s 