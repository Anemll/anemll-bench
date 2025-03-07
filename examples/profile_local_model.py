#!/usr/bin/env python3
"""
Profiles local CoreML models directly from their local paths without triggering downloads.
This is a standalone script that doesn't rely on the platform model mechanisms.
"""

import os
import sys
import argparse
import logging
import time
import platform
import subprocess
import webbrowser
import json
from typing import Optional, Dict, Any
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import coremltools as ct
import numpy as np

def get_model_size(model_path):
    """Get the size of the model in bytes and megabytes."""
    total_size = 0
    weights_size = 0
    
    if os.path.isdir(model_path):
        # Directory (mlmodelc or mlpackage)
        for dirpath, dirnames, filenames in os.walk(model_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                file_size = os.path.getsize(fp)
                total_size += file_size
                
                # Rough approximation of weight files - this is just an estimation
                if "weight" in f.lower() or "model" in f.lower() or f.endswith('.bin'):
                    weights_size += file_size
    else:
        # Single file
        total_size = os.path.getsize(model_path)
        weights_size = total_size  # For single files, assume all are weights
    
    return {
        "size_bytes": total_size,
        "size_mb": total_size / (1024 * 1024),
        "weights_bytes": weights_size,
        "weights_mb": weights_size / (1024 * 1024),
        "weights_percentage": (weights_size / total_size * 100) if total_size > 0 else 0
    }

def get_system_info() -> Dict[str, Any]:
    """Get basic system information"""
    import platform
    import subprocess
    
    system_info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }
    
    # Get more detailed Mac info if on macOS
    if platform.system() == "Darwin":
        # Get Mac model
        try:
            mac_model = subprocess.check_output(["sysctl", "-n", "hw.model"]).decode("utf-8").strip()
            system_info["mac_model"] = mac_model
        except:
            system_info["mac_model"] = "Unknown Mac"
        
        # Check if Apple Silicon
        try:
            processor_info = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode("utf-8").strip()
            system_info["cpu_model"] = processor_info
            system_info["is_apple_silicon"] = "Apple" in processor_info
        except:
            system_info["is_apple_silicon"] = False
            
        # Get memory
        try:
            memory_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode("utf-8").strip())
            system_info["memory_gb"] = round(memory_bytes / (1024**3), 1)
        except:
            system_info["memory_gb"] = "Unknown"
            
        # Get a cleaner macOS version - e.g., "macOS 13.4 Ventura"
        try:
            # Get macOS version using sw_vers
            macos_version = subprocess.check_output(["sw_vers", "-productVersion"]).decode("utf-8").strip()
            macos_name = "macOS"
            
            # Determine macOS name based on version
            version_major = int(macos_version.split('.')[0])
            if version_major == 10:
                minor = int(macos_version.split('.')[1])
                if minor == 15:
                    macos_name = "macOS Catalina"
                elif minor == 14:
                    macos_name = "macOS Mojave"
                elif minor == 13:
                    macos_name = "macOS High Sierra"
                else:
                    macos_name = "macOS"
            elif version_major == 11:
                macos_name = "macOS Big Sur"
            elif version_major == 12:
                macos_name = "macOS Monterey"
            elif version_major == 13:
                macos_name = "macOS Ventura"
            elif version_major == 14:
                macos_name = "macOS Sonoma"
            elif version_major == 15:
                macos_name = "macOS Sequoia"
            
            system_info["os_display"] = f"{macos_name} {macos_version}"
        except:
            system_info["os_display"] = "macOS Unknown Version"
    
    return system_info

def generate_html_report(model_name: str, model_size_info: Dict, system_info: Dict, 
                         inference_time_ms: float, throughput_gb_s: float, 
                         input_shape: tuple, output_path: str = "profile_report.html"):
    """Generate a simple HTML report with profiling results that resembles the original ANEMLL report"""
    
    # Prepare JSON data for email sharing
    json_data = json.dumps({
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'system_info': system_info,
        'model_name': model_name,
        'model_size_mb': model_size_info['size_mb'],
        'inference_time_ms': inference_time_ms,
        'throughput_gb_s': throughput_gb_s,
        'input_shape': input_shape
    }, indent=2)
    
    # Use the friendly OS display if available
    os_display = system_info.get('os_display', system_info.get('os_version', 'Unknown'))
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Apple Neural Engine Benchmark Results - {model_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 20px;
            color: #333;
        }}
        h1, h2 {{
            color: #000;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .system-info {{
            margin: 20px 0;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }}
        .info-section {{
            margin: 20px 0;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        table, th, td {{
            border: 1px solid #ddd;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .metrics {{
            margin: 20px 0;
            padding: 15px;
            background-color: #f0f7ff;
            border-radius: 5px;
            border-left: 5px solid #0066cc;
        }}
        .footer {{
            margin-top: 30px;
            font-size: 12px;
            color: #999;
            text-align: center;
        }}
        .info-table {{
            width: auto;
            min-width: 50%;
        }}
        .info-table td:first-child {{
            width: 150px;
        }}
        .send-button {{
            background-color: #28a745;
            color: white;
            padding: 12px 24px;
            font-size: 18px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin: 20px auto;
            display: block;
        }}
        a {{
            color: #0066cc;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Apple Neural Engine Benchmark Results by <a href="https://www.anemll.com" target="_blank">ANEMLL</a></h1>
        
        <div style="text-align: center; margin: 20px 0;">
            <button id="sendResultsBtn" class="send-button">
                Send Results to ANEMLL Team
            </button>
        </div>
        
        <div class="system-info">
            <h2>System Information</h2>
            <table class="info-table">
                <tr>
                    <td><strong>Mac Model:</strong></td>
                    <td>{system_info.get('mac_model', 'Unknown')}</td>
                </tr>
                <tr>
                    <td><strong>OS:</strong></td>
                    <td>{os_display}</td>
                </tr>
                <tr>
                    <td><strong>CPU:</strong></td>
                    <td>{system_info.get('cpu_model', 'Unknown')}</td>
                </tr>
                <tr>
                    <td><strong>RAM:</strong></td>
                    <td>{system_info.get('memory_gb', 'Unknown')} GB</td>
                </tr>
                <tr>
                    <td><strong>Apple Silicon:</strong></td>
                    <td>{'Yes' if system_info.get('is_apple_silicon', False) else 'No'}</td>
                </tr>
            </table>
        </div>
        
        <div class="info-section">
            <h2>Model Information</h2>
            <table class="info-table">
                <tr>
                    <td><strong>Name:</strong></td>
                    <td>{model_name}</td>
                </tr>
                <tr>
                    <td><strong>Size:</strong></td>
                    <td>{model_size_info['size_mb']:.2f} MB ({model_size_info['size_bytes']:,} bytes)</td>
                </tr>
                <tr>
                    <td><strong>Weights Size:</strong></td>
                    <td>{model_size_info['weights_mb']:.2f} MB ({model_size_info['weights_bytes']:,} bytes)</td>
                </tr>
                <tr>
                    <td><strong>Weights %:</strong></td>
                    <td>{model_size_info['weights_percentage']:.1f}% of total size</td>
                </tr>
                <tr>
                    <td><strong>Input Shape:</strong></td>
                    <td>{input_shape}</td>
                </tr>
            </table>
        </div>
        
        <div class="metrics">
            <h2>Performance Metrics</h2>
            <table class="info-table">
                <tr>
                    <td><strong>Inference Time:</strong></td>
                    <td>{inference_time_ms:.2f} ms</td>
                </tr>
                <tr>
                    <td><strong>Throughput:</strong></td>
                    <td>{throughput_gb_s:.2f} GB/s</td>
                </tr>
            </table>
        </div>
        
        <div class="footer">
            <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong><a href="https://www.anemll.com" target="_blank">ANEMLL BENCH</a></strong></p>
        </div>
    </div>
    
    <script>
        document.getElementById('sendResultsBtn').addEventListener('click', function() {{
            const jsonData = {json_data!r};
            const emailSubject = "ANEMLL BENCH Results";
            const emailBody = "Here are my benchmark results:\\n\\n" + jsonData;
            
            window.location.href = "mailto:realanemll@gmail.com?subject=" + 
                encodeURIComponent(emailSubject) + 
                "&body=" + encodeURIComponent(emailBody);
        }});
    </script>
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Profile a local CoreML model without triggering downloads")
    parser.add_argument("--model", required=True, help="Path to the CoreML model file (.mlpackage or .mlmodelc)")
    parser.add_argument("--hidden-size", type=int, default=4096, help="Hidden size to use for input (default: 4096)")
    parser.add_argument("--sequence-length", type=int, default=1, help="Sequence length to use (default: 1)")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations for profiling (default: 1000)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser with report")
    parser.add_argument("--output", type=str, help="Custom output path for the report (default: profile_report.html)")
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        logger.error(f"Model not found: {args.model}")
        return 1
    
    # Set compute units 
    compute_unit = ct.ComputeUnit.CPU_AND_NE  # Use ANE by default
    
    logger.info(f"Profiling model: {args.model}")
    logger.info(f"Using hidden size: {args.hidden_size}")
    logger.info(f"Using sequence length: {args.sequence_length}")
    logger.info(f"Using compute units: ANE")
    
    # Print system info
    system_info = get_system_info()
    
    # Get user-friendly OS display
    os_display = system_info.get('os_display', system_info.get('os_version', 'Unknown'))
    
    print("\n=== System Information ===")
    print(f"Mac Model: {system_info.get('mac_model', 'Unknown')}")
    print(f"OS: {os_display}")
    print(f"CPU: {system_info.get('cpu_model', 'Unknown')}")
    print(f"RAM: {system_info.get('memory_gb', 'Unknown')} GB")
    print(f"Apple Silicon: {'Yes' if system_info.get('is_apple_silicon', False) else 'No'}")
    print("===========================\n")
    
    # Print model info
    print("\n=== Model Information ===")
    model_size_info = get_model_size(args.model)
    print(f"Path: {args.model}")
    print(f"Total Size: {model_size_info['size_mb']:.2f} MB ({model_size_info['size_bytes']:,} bytes)")
    print(f"Weights Size: {model_size_info['weights_mb']:.2f} MB ({model_size_info['weights_bytes']:,} bytes)")
    print(f"Weights Percentage: {model_size_info['weights_percentage']:.1f}% of total size")
    print("===========================\n")
    
    # Load the model
    print(f"Loading model: {args.model}")
    try:
        model = ct.models.MLModel(args.model)
        print(f"Successfully loaded model: {args.model}")
        
        # Check if this is an LM head model based on name
        is_lm_head = "lm_head" in args.model.lower()
        
        # Create input shape and data
        batch_size = 1
        sequence_length = args.sequence_length
        hidden_size = args.hidden_size
        
        if is_lm_head:
            print("Detected LM head model based on filename")
            input_name = "hidden_states"
            shape = (batch_size, sequence_length, hidden_size)
            print(f"Using LM head specialized input preparation")
            print(f"Using inputs: hidden_states:{shape}")
        else:
            # Try to get input info from model spec
            if hasattr(model, 'get_spec'):
                spec = model.get_spec().description.input
                if spec and len(spec) > 0:
                    input_name = spec[0].name
                    shape = tuple(spec[0].type.multiArrayType.shape)
                    print(f"Using input name and shape from model spec: {input_name}:{shape}")
                else:
                    input_name = "input_ids"
                    shape = (batch_size, sequence_length)
                    print(f"Using default input preparation")
                    print(f"Using inputs: {input_name}:{shape}")
            else:
                input_name = "input_ids"
                shape = (batch_size, sequence_length)
                print(f"Using default input preparation")
                print(f"Using inputs: {input_name}:{shape}")
        
        # Create input data
        input_data = np.random.rand(*shape).astype(np.float32)
        inputs = {input_name: input_data}
        
        # Warmup
        print("Warming up model...")
        try:
            for _ in range(5):
                model.predict(inputs)
        except Exception as e:
            print(f"Warning: Error during warmup: {e}")
            return 1
        
        # Benchmark
        print(f"Running {args.iterations} iterations for profiling...")
        start_time = time.time()
        
        for i in range(args.iterations):
            try:
                prediction = model.predict(inputs)
            except Exception as e:
                print(f"Error during prediction: {e}")
                return 1
        
        end_time = time.time()
        
        # Calculate results
        elapsed_time = end_time - start_time
        inference_time_ms = (elapsed_time * 1000) / args.iterations
        total_data_size = model_size_info['size_bytes'] + np.prod(shape) * 4  # model size + input data size in bytes
        throughput_gb_s = (total_data_size / (inference_time_ms / 1000)) / 1e9  # GB/s
        
        # Print results
        print(f"\nAverage inference time: {inference_time_ms:.2f} ms")
        print(f"Throughput: {throughput_gb_s:.2f} GB/s (based on weights + I/O)")
        print(f"Model size: {model_size_info['size_mb']:.2f} MB")
        print(f"Weights size: {model_size_info['weights_mb']:.2f} MB ({model_size_info['weights_percentage']:.1f}% of total)")
        
        # Generate HTML report
        model_name = os.path.basename(args.model)
        report_path = args.output if args.output else "profile_report.html"
        
        html_path = generate_html_report(
            model_name=model_name,
            model_size_info=model_size_info,
            system_info=system_info,
            inference_time_ms=inference_time_ms,
            throughput_gb_s=throughput_gb_s,
            input_shape=shape,
            output_path=report_path
        )
        
        print(f"\nReport generated at: {html_path}")
        
        # Open the report in the browser if requested
        if not args.no_browser:
            abs_path = os.path.abspath(html_path)
            print(f"Opening report in browser: {abs_path}")
            #webbrowser.open(f"file://{abs_path}")
        
    except Exception as e:
        logger.error(f"Error profiling model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\nProfile complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 