"""
BenchmarkResult class for storing benchmark results
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class BenchmarkResult:
    """Class for storing benchmark results"""
    model_name: str
    input_shape: List[int]
    params_count: int
    backend: str
    inference_time_ms: float
    memory_used_mb: float
    tflops: Optional[float] = None
    throughput_gb_s: Optional[float] = None
    system_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def throughput_gbps(self):
        """For backward compatibility"""
        return self.throughput_gb_s 