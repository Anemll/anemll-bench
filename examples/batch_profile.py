#!/usr/bin/env python3
"""
Batch profile script for multiple CoreML models using ANEMLL-Bench
"""

import os
import sys
import argparse
import json
import time
import glob
import webbrowser
from pathlib import Path

# Add parent directory to path for development imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from anemll_bench import Benchmark
from anemll_bench.utils.system_info import get_system_info
from anemll_bench.models.coreml_adapter import get_model_size


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Batch profile multiple CoreML models')
    
    # Model options
    parser.add_argument('--models-dir', type=str, required=True,
                        help='Directory containing CoreML models')
    parser.add_argument('--pattern', type=str, default='*.mlmodel*',
                        help='Glob pattern to match model files (default: *.mlmodel*)')
    
    # Benchmark options
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for profiling')
    parser.add_argument('--sequence-length', type=int, default=512,
                        help='Sequence length for text models')
    parser.add_argument('--hidden-size', type=int, default=4096,
                        help='Hidden size for text models')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of iterations for profiling')
    parser.add_argument('--compute-units', type=str, default='CPU_AND_NE',
                        choices=['CPU_AND_NE', 'CPU_ONLY', 'ALL'],
                        help='Compute units to use for inference')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='./reports',
                        help='Directory to save benchmark results and reports')
    parser.add_argument('--include-charts', action='store_true',
                        help='Include performance charts in HTML report (disabled by default)')
    parser.add_argument('--upload', action='store_true',
                        help='Upload reports to sharing service')
    parser.add_argument('--upload-service', type=str, default='jsonbin',
                        choices=['gist', 'pastebin', 'jsonbin'],
                        help='Service to upload reports to')
    
    # Extra options
    parser.add_argument('--compare-cpu', action='store_true',
                        help='Compare with CPU-only performance (disabled by default)')
    parser.add_argument('--tflops', type=float, default=None,
                        help='Specify the total number of trillion floating point operations (TFLOPs) per iteration (not TFLOPS rate)')
    
    args = parser.parse_args()
    
    return args


def find_models(models_dir, pattern):
    """Find all models matching the pattern in the directory"""
    search_path = os.path.join(models_dir, pattern)
    models = glob.glob(search_path)
    
    # Also look for compiled models in subdirectories
    if '*.mlmodel*' in pattern:
        compiled_models = glob.glob(os.path.join(models_dir, '*.mlmodelc'))
        models.extend(compiled_models)
    
    return sorted(models)


def main():
    """Main entry point"""
    args = parse_args()
    
    # Find all models
    models = find_models(args.models_dir, args.pattern)
    
    if not models:
        print(f"No models found in {args.models_dir} matching pattern {args.pattern}")
        return 1
    
    print(f"Found {len(models)} models to benchmark:")
    for i, model in enumerate(models):
        print(f"  {i+1}. {os.path.basename(model)}")
    
    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get system info (same for all benchmarks)
    system_info = get_system_info()
    print("\n=== System Information ===")
    print(f"Mac Model: {system_info.get('mac_model', 'Unknown')}")
    
    # Use the user-friendly macOS version if available, otherwise fall back to the old format
    if 'macos_version' in system_info:
        print(f"OS: {system_info['macos_version']}")
    else:
        print(f"OS: {system_info.get('os', {}).get('name', 'Unknown')} "
              f"{system_info.get('os', {}).get('release', '')}")
    
    print(f"CPU: {system_info.get('cpu', {}).get('brand', 'Unknown')}")
    print(f"RAM: {system_info.get('ram', {}).get('total_gb', 'Unknown')} GB")
    print(f"Apple Silicon: {'Yes' if system_info.get('apple_silicon', False) else 'No'}")
    print("===========================\n")
    
    # Create a summary of all results
    summary = {
        "system_info": system_info,
        "models": [],
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "hidden_size": args.hidden_size,
        "iterations": args.iterations,
        "compute_units": args.compute_units,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Process each model
    for i, model_path in enumerate(models):
        model_name = os.path.basename(model_path)
        print(f"\n[{i+1}/{len(models)}] Benchmarking {model_name}...")
        
        # Create a new benchmark instance for each model to keep results separate
        benchmark = Benchmark()
        
        try:
            # Benchmark with the requested compute units
            if args.tflops is not None:
                print(f"Using provided TFLOPS value: {args.tflops}")
                
            result = benchmark.benchmark_coreml_file(
                model_path=model_path,
                model_name=model_name,
                num_runs=args.iterations,
                batch_size=args.batch_size,
                sequence_length=args.sequence_length,
                hidden_size=args.hidden_size,
                compute_units=args.compute_units,
                known_tflops=args.tflops
            )
            
            # Also benchmark with CPU only if requested
            cpu_result = None
            if args.compare_cpu and args.compute_units != "CPU_ONLY":
                print("\nComparing with CPU-only performance...")
                cpu_result = benchmark.benchmark_coreml_file(
                    model_path=model_path,
                    model_name=f"{model_name} (CPU)",
                    num_runs=args.iterations,
                    batch_size=args.batch_size,
                    sequence_length=args.sequence_length,
                    hidden_size=args.hidden_size,
                    compute_units="CPU_ONLY",
                    known_tflops=args.tflops
                )
                
                # Calculate speedup
                ane_time = result.inference_time_ms
                cpu_time = cpu_result.inference_time_ms
                speedup = cpu_time / ane_time if ane_time > 0 else 0
                
                print("\n=== Performance Comparison ===")
                print(f"ANE Inference: {ane_time:.2f} ms")
                print(f"CPU Inference: {cpu_time:.2f} ms")
                print(f"Speedup: {speedup:.2f}x")
                print("=============================\n")
            
            # Generate individual report
            report_filename = f"{i+1:02d}_{model_name.replace('.', '_')}_report.html"
            report_path = os.path.join(args.output_dir, report_filename)
            
            report_url = benchmark.generate_report(
                output_path=report_path,
                upload=args.upload,
                upload_service=args.upload_service,
                include_charts=args.include_charts
            )
            
            if report_url:
                print(f"Report uploaded to: {report_url}")
            
            # Add to summary
            model_result = {
                "name": model_name,
                "path": model_path,
                "size_bytes": get_model_size(model_path),
                "size_mb": get_model_size(model_path) / (1024 * 1024),
                "inference_time_ms": result.inference_time_ms,
                "throughput_gbps": result.throughput_gbps,
                "tflops": result.tflops,
                "report_path": report_path,
                "report_url": report_url
            }
            
            if cpu_result:
                model_result["cpu_inference_time_ms"] = cpu_result.inference_time_ms
                model_result["speedup"] = cpu_result.inference_time_ms / result.inference_time_ms
            
            summary["models"].append(model_result)
            
        except Exception as e:
            print(f"Error benchmarking {model_name}: {e}")
            # Add failed model to summary
            summary["models"].append({
                "name": model_name,
                "path": model_path,
                "error": str(e)
            })
    
    # Save summary
    summary_path = os.path.join(args.output_dir, "batch_profile_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nBatch profile complete! Summary saved to {summary_path}")
    
    # Generate summary HTML
    html_summary = generate_html_summary(summary)
    summary_html_path = os.path.join(args.output_dir, "batch_profile_summary.html")
    with open(summary_html_path, 'w') as f:
        f.write(html_summary)
    
    print(f"HTML summary saved to {summary_html_path}")
    
    # Open the summary HTML report
    webbrowser.open(f"file://{os.path.abspath(summary_html_path)}")
    
    return 0


def generate_html_summary(summary):
    """Generate an HTML summary of all benchmarks"""
    system_info = summary["system_info"]
    models = summary["models"]
    
    # Start building HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ANEMLL-Bench Batch Profile Summary</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1, h2, h3 {{
                color: #0056b3;
            }}
            .header {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;
                border-left: 5px solid #007bff;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #dee2e6;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #f8f9fa;
            }}
            tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}
            .success {{
                color: #28a745;
                font-weight: bold;
            }}
            .error {{
                color: #dc3545;
            }}
            .system-info {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }}
            .card {{
                background-color: #f8f9fa;
                border-radius: 5px;
                padding: 15px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>ANEMLL-Bench Batch Profile Summary</h1>
            <p>Generated on: {summary["timestamp"]}</p>
        </div>
        
        <h2>System Information</h2>
        <div class="system-info">
            <div class="card">
                <h3>Hardware</h3>
                <p><strong>Mac Model:</strong> {system_info.get('mac_model', 'Unknown')}</p>
                <p><strong>CPU:</strong> {system_info.get('cpu', {}).get('brand', 'Unknown')}</p>
                <p><strong>CPU Cores:</strong> {system_info.get('cpu', {}).get('cores', 'Unknown')}</p>
                <p><strong>RAM:</strong> {system_info.get('ram', {}).get('total_gb', 'Unknown')} GB</p>
                <p><strong>Apple Silicon:</strong> {'Yes' if system_info.get('apple_silicon', False) else 'No'}</p>
            </div>
            <div class="card">
                <h3>Software</h3>
                <p><strong>OS:</strong> {system_info.get('os', {}).get('name', 'Unknown')} {system_info.get('os', {}).get('release', '')}</p>
                <p><strong>OS Version:</strong> {system_info.get('os', {}).get('version', 'Unknown')}</p>
                <p><strong>Python Version:</strong> {system_info.get('python_version', 'Unknown')}</p>
                <p><strong>Compute Units:</strong> {summary["compute_units"]}</p>
                <p><strong>Batch Size:</strong> {summary["batch_size"]}</p>
                <p><strong>Sequence Length:</strong> {summary["sequence_length"]}</p>
                <p><strong>Hidden Size:</strong> {summary["hidden_size"]}</p>
                <p><strong>Iterations:</strong> {summary["iterations"]}</p>
            </div>
        </div>
        
        <h2>Benchmark Results</h2>
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Model</th>
                    <th>Size (MB)</th>
                    <th>Inference Time (ms)</th>
                    <th>Throughput (GB/s)</th>
                    <th>TFLOPS</th>
                    <th>CPU Speedup</th>
                    <th>Report</th>
                </tr>
            </thead>
            <tbody>
    """
    
    # Add a row for each model
    for i, model in enumerate(models):
        if "error" in model:
            # Error case
            html += f"""
                <tr class="error">
                    <td>{i+1}</td>
                    <td>{model["name"]}</td>
                    <td colspan="6">Error: {model["error"]}</td>
                </tr>
            """
        else:
            # Success case
            report_link = f'<a href="{model["report_url"] if "report_url" in model else model["report_path"]}">View Report</a>'
            speedup = f'{model.get("speedup", "N/A"):.2f}x' if "speedup" in model else "N/A"
            
            # Handle TFLOPS field - show it only if available
            tflops_html = "N/A"
            if "tflops" in model and model["tflops"] is not None:
                tflops_html = f'{model["tflops"]:.4f}'
            
            html += f"""
                <tr>
                    <td>{i+1}</td>
                    <td>{model["name"]}</td>
                    <td>{model["size_mb"]:.2f}</td>
                    <td>{model["inference_time_ms"]:.2f}</td>
                    <td>{model["throughput_gbps"]:.2f}</td>
                    <td>{tflops_html}</td>
                    <td class="success">{speedup}</td>
                    <td>{report_link}</td>
                </tr>
            """
    
    # Close the HTML
    html += """
            </tbody>
        </table>
        
        <h2>Analysis</h2>
        <div class="card">
            <p>This summary report compares the performance of multiple CoreML models on the Apple Neural Engine.</p>
            
            <h3>Key Observations:</h3>
            <ul>
    """
    
    # Add some analysis
    if len(models) > 0:
        # Find fastest and slowest models
        valid_models = [m for m in models if "error" not in m]
        if valid_models:
            fastest = min(valid_models, key=lambda m: m.get("inference_time_ms", float('inf')))
            slowest = max(valid_models, key=lambda m: m.get("inference_time_ms", 0))
            highest_throughput = max(valid_models, key=lambda m: m.get("throughput_gbps", 0))
            
            html += f"""
                <li>Fastest model: <strong>{fastest["name"]}</strong> ({fastest["inference_time_ms"]:.2f} ms)</li>
                <li>Slowest model: <strong>{slowest["name"]}</strong> ({slowest["inference_time_ms"]:.2f} ms)</li>
                <li>Highest memory throughput: <strong>{highest_throughput["name"]}</strong> ({highest_throughput["throughput_gbps"]:.2f} GB/s)</li>
            """
            
            # Add TFLOPS info only if available
            models_with_tflops = [m for m in valid_models if "tflops" in m and m["tflops"] is not None]
            if models_with_tflops:
                highest_tflops = max(models_with_tflops, key=lambda m: m["tflops"])
                html += f"""
                    <li>Highest TFLOPS: <strong>{highest_tflops["name"]}</strong> ({highest_tflops["tflops"]:.4f} TFLOPS)</li>
                """
            
            # Add speedup analysis if available
            speedup_models = [m for m in valid_models if "speedup" in m]
            if speedup_models:
                highest_speedup = max(speedup_models, key=lambda m: m.get("speedup", 0))
                html += f"""
                    <li>Best CPU-to-ANE speedup: <strong>{highest_speedup["name"]}</strong> ({highest_speedup["speedup"]:.2f}x)</li>
                """
    
    # Close the HTML
    html += """
            </ul>
        </div>
        
        <footer style="margin-top: 30px; text-align: center; font-size: 0.9em; color: #6c757d;">
            <p>Generated with ANEMLL-Bench - Apple Neural Engine Machine Learning Benchmark</p>
        </footer>
    </body>
    </html>
    """
    
    return html


if __name__ == "__main__":
    sys.exit(main()) 