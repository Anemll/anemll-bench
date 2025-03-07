"""
Main entry point for anemll_bench
"""

import sys
import argparse
from anemll_bench import Benchmark
from anemll_bench.utils.system_info import get_system_info


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='ANEMLL-Bench: Apple Neural Engine Benchmarking Tool'
    )
    
    # Model specification
    model_group = parser.add_argument_group('Model Options')
    model_group.add_argument('--model', type=str, 
                            help='Hugging Face model ID to benchmark')
    model_group.add_argument('--config', type=str,
                            help='Path to benchmark configuration JSON file')
    
    # Input parameters
    input_group = parser.add_argument_group('Input Parameters')
    input_group.add_argument('--sequence-length', type=int, default=128,
                            help='Sequence length for transformer models')
    input_group.add_argument('--batch-size', type=int, default=1,
                            help='Batch size for inference')
    input_group.add_argument('--hidden-size', type=int, default=768,
                            help='Hidden size for transformer models')
    
    # Benchmark options
    bench_group = parser.add_argument_group('Benchmark Options')
    bench_group.add_argument('--runs', type=int, default=50,
                            help='Number of benchmark runs')
    bench_group.add_argument('--cpu-only', action='store_true',
                            help='Only benchmark on CPU (skip ANE)')
    bench_group.add_argument('--ane-only', action='store_true',
                            help='Only benchmark on ANE (skip CPU)')
    
    # Report options
    report_group = parser.add_argument_group('Report Options')
    report_group.add_argument('--output', type=str, default="benchmark_report.html",
                            help='Path to save the benchmark report')
    report_group.add_argument('--upload', action='store_true',
                            help='Upload the report to a sharing service')
    report_group.add_argument('--upload-service', type=str, default='jsonbin', 
                            choices=['gist', 'pastebin', 'jsonbin'],
                            help='Service to upload the report to')
    
    # System options
    system_group = parser.add_argument_group('System Options')
    system_group.add_argument('--system-info', action='store_true',
                            help='Display system information and exit')
    
    return parser.parse_args()


def display_system_info():
    """Display system information and exit"""
    system_info = get_system_info()
    
    print("\n=== ANEMLL-Bench System Information ===")
    print(f"Mac Model: {system_info.get('mac_model', 'Unknown')}")
    print(f"OS: {system_info.get('os', {}).get('name', 'Unknown')} "
          f"{system_info.get('os', {}).get('release', '')} "
          f"{system_info.get('os', {}).get('version', '')}")
    print(f"CPU: {system_info.get('cpu', {}).get('brand', 'Unknown')}")
    print(f"CPU Cores: {system_info.get('cpu', {}).get('cores', 'Unknown')} physical, "
          f"{system_info.get('cpu', {}).get('threads', 'Unknown')} logical")
    print(f"RAM: {system_info.get('ram', {}).get('total_gb', 'Unknown')} GB total, "
          f"{system_info.get('ram', {}).get('available_gb', 'Unknown')} GB available")
    print(f"Apple Silicon: {'Yes' if system_info.get('apple_silicon', False) else 'No'}")
    
    if system_info.get('apple_silicon', False):
        print("\nNeural Engine Information:")
        print(f"ANE Available: {'Yes' if system_info.get('neural_engine', {}).get('available', False) else 'No'}")
    
    print(f"Python Version: {system_info.get('python_version', 'Unknown')}")
    print("===================================\n")


def main():
    """Main entry point"""
    args = parse_args()
    
    # Just display system info if requested
    if args.system_info:
        display_system_info()
        return 0
    
    # Check that we have either a model or config
    if not args.model and not args.config:
        print("Error: Either --model or --config must be specified")
        return 1
    
    # Initialize benchmark
    benchmark = Benchmark(config_path=args.config)
    
    # If we have a config file, run that benchmark
    if args.config:
        print(f"Running benchmarks from config file: {args.config}")
        benchmark.run()
    else:
        from transformers import AutoModelForCausalLM, AutoModel
        import torch
        
        # Create backends list
        backends = []
        if not args.ane_only:
            backends.append("CPU")
        if not args.cpu_only:
            backends.append("ANE")
        
        try:
            # Load the model from HF
            print(f"Loading model from Hugging Face: {args.model}")
            try:
                # Try to load as causal LM first
                model = AutoModelForCausalLM.from_pretrained(
                    args.model, 
                    torch_dtype=torch.float16,
                    device_map="cpu"  # Ensure it's loaded on CPU first
                )
                print("Model loaded as Causal LM")
            except Exception as e:
                print(f"Could not load as Causal LM, trying generic model: {e}")
                model = AutoModel.from_pretrained(
                    args.model, 
                    torch_dtype=torch.float16,
                    device_map="cpu"
                )
                print("Model loaded as generic model")
            
            # Get model's hidden size - different models have different attributes
            hidden_size = args.hidden_size  # Default
            if hasattr(model.config, 'hidden_size'):
                hidden_size = model.config.hidden_size
            elif hasattr(model.config, 'd_model'):
                hidden_size = model.config.d_model
            
            # Standard input shape for transformer models
            input_shape = [args.batch_size, args.sequence_length, hidden_size]
            print(f"Using input shape: {input_shape}")
            
            # Benchmark on each backend
            for backend in backends:
                print(f"\n=== Running {backend} Benchmark ===")
                benchmark.benchmark_model(
                    model=model, 
                    model_name=args.model, 
                    input_shape=input_shape,
                    backend=backend, 
                    num_runs=args.runs
                )
            
            # Generate report
            print(f"\nGenerating report to {args.output}...")
            report_url = benchmark.generate_report(
                output_path=args.output,
                upload=args.upload,
                upload_service=args.upload_service
            )
            
            if report_url:
                print(f"Report uploaded to: {report_url}")
                print("You can share this URL to let others view your benchmark results.")
            
            # Calculate speedup if we have both CPU and ANE results
            if len(benchmark.results) >= 2:
                cpu_results = [r for r in benchmark.results if r.backend == "CPU"]
                ane_results = [r for r in benchmark.results if r.backend == "ANE"]
                
                if cpu_results and ane_results:
                    cpu_time = cpu_results[0].inference_time_ms
                    ane_time = ane_results[0].inference_time_ms
                    speedup = cpu_time / ane_time if ane_time > 0 else 0
                    
                    print("\n=== Benchmark Summary ===")
                    print(f"Model: {args.model}")
                    print(f"CPU Inference: {cpu_time:.2f} ms")
                    print(f"ANE Inference: {ane_time:.2f} ms")
                    print(f"Speedup: {speedup:.2f}x")
            
        except Exception as e:
            print(f"Error benchmarking model: {e}")
            return 1
    
    print("\nBenchmark complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 