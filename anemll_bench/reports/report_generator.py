"""
Module for generating HTML reports from benchmark results
"""

import os
import json
import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import base64
from io import BytesIO
import numpy as np
import plotly
import plotly.graph_objects as go
from ..models.benchmark_result import BenchmarkResult

# Define the cache directory
CACHE_DIR = os.path.expanduser("~/.cache/anemll-bench")
REPORTS_DIR = os.path.join(CACHE_DIR, "reports")


def _create_report_directory():
    """Create directory for reports if it doesn't exist"""
    # Use the cache directory instead of the current working directory
    os.makedirs(REPORTS_DIR, exist_ok=True)
    return REPORTS_DIR


def _create_plots(results, report_dir, report_id):
    """Generate plots for the report"""
    # Convert results to DataFrame
    rows = []
    for r in results:
        row = {
            'Model': r.model_name,
            'Backend': r.backend,
            'Inference Time (ms)': r.inference_time_ms,
            'Memory (MB)': r.memory_used_mb,
        }
        
        if r.tflops is not None:
            row['TFLOPS'] = r.tflops
            
        if r.throughput_gbps is not None:
            row['Throughput (GB/s)'] = r.throughput_gbps
            
        rows.append(row)
        
    df = pd.DataFrame(rows)
    
    # Create plots directory
    plots_dir = os.path.join(report_dir, f"plots_{report_id}")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create plots
    plot_paths = {}
    
    # Inference time bar plot
    plt.figure(figsize=(10, 3))  # Reduced height from 6 to 3
    sns.barplot(x='Model', y='Inference Time (ms)', hue='Backend', data=df)
    plt.title('Model Inference Time Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    inference_time_plot = os.path.join(plots_dir, 'inference_time.png')
    plt.savefig(inference_time_plot)
    plt.close()
    plot_paths['inference_time'] = inference_time_plot
    
    # Throughput bar plot (if data available)
    if 'Throughput (GB/s)' in df.columns:
        plt.figure(figsize=(10, 3))  # Reduced height from 6 to 3
        sns.barplot(x='Model', y='Throughput (GB/s)', hue='Backend', data=df)
        plt.title('Model Throughput Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        throughput_plot = os.path.join(plots_dir, 'throughput.png')
        plt.savefig(throughput_plot)
        plt.close()
        plot_paths['throughput'] = throughput_plot
    
    return plot_paths


def _create_bar_chart(data: Dict[str, List[float]], title: str) -> str:
    """
    Create a bar chart from the given data and return as HTML img tag with base64 encoded image
    
    Args:
        data: Dictionary mapping labels to values
        title: Chart title
        
    Returns:
        HTML img tag with base64 encoded PNG image
    """
    fig = plt.figure(figsize=(10, 3))  # Reduced height from 6 to 3
    ax = fig.add_subplot(111)
    
    # Create bar chart
    bar_width = 0.35
    index = np.arange(len(list(data.values())[0]))
    
    for i, (label, values) in enumerate(data.items()):
        ax.bar(index + i*bar_width, values, bar_width, label=label)
    
    # Add labels and title
    ax.set_xlabel('Models')
    ax.set_title(title)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels([f"Model {i+1}" for i in range(len(list(data.values())[0]))])
    ax.legend()
    
    # Convert to base64
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    # Close the figure to free memory
    plt.close(fig)
    
    # Encode
    encoded = base64.b64encode(image_png).decode('utf-8')
    # Return as HTML img tag instead of just the data URL
    return f'<img src="data:image/png;base64,{encoded}" alt="{title}" class="chart-image" />'


def _generate_key_findings(grouped_data, grouped_models):
    """Generate key findings from the benchmark results"""
    list_items = []
    
    # Find fastest model
    fastest_model = None
    fastest_time = float('inf')
    
    # Best throughput model
    best_throughput_model = None
    best_throughput = 0
    
    # Best TFLOPS model
    best_tflops_model = None
    best_tflops = 0
    has_tflops = False
    
    for model in grouped_models:
        if 'ANE' in grouped_data[model]:
            time = grouped_data[model]['ANE']['inference_time_ms']
            throughput = grouped_data[model]['ANE'].get('throughput_gb_s', 
                         grouped_data[model]['ANE'].get('throughput_gbps', 0))
            tflops = grouped_data[model]['ANE']['tflops']
            
            if time < fastest_time:
                fastest_model = model
                fastest_time = time
                
            if throughput > best_throughput:
                best_throughput_model = model
                best_throughput = throughput
                
            # Only include in TFLOPS comparison if it's a valid value
            if tflops is not None and tflops > 0:
                has_tflops = True
                if tflops > best_tflops:
                    best_tflops_model = model
                    best_tflops = tflops
    
    # Add findings
    if fastest_model:
        item = f"<li><strong>Fastest Model:</strong> {fastest_model} "
        item += f"with inference time of {fastest_time:.2f} ms"
        
        # Add speedup vs CPU if available
        if fastest_model in grouped_data and 'CPU' in grouped_data[fastest_model]:
            cpu_time = grouped_data[fastest_model]['CPU']['inference_time_ms']
            speedup = cpu_time / fastest_time if fastest_time > 0 else 0
            item += f" ({speedup:.1f}x faster than CPU)"
            
        item += "</li>"
        list_items.append(item)
        
    if best_throughput_model:
        item = f"<li><strong>Best Memory Throughput:</strong> {best_throughput_model} "
        item += f"with throughput of {best_throughput:.2f} GB/s"
        
        # Add comparison vs CPU if available
        if best_throughput_model in grouped_data and 'CPU' in grouped_data[best_throughput_model]:
            cpu_throughput = grouped_data[best_throughput_model]['CPU'].get('throughput_gb_s', 
                            grouped_data[best_throughput_model]['CPU'].get('throughput_gbps', 0))
            ratio = best_throughput / cpu_throughput if cpu_throughput > 0 else 0
            item += f" ({ratio:.1f}x faster than CPU)"
            
        item += "</li>"
        list_items.append(item)
    
    # Only add TFLOPS information if we have valid non-None data
    if has_tflops and best_tflops_model and best_tflops is not None:
        item = f"<li><strong>Best Compute Performance:</strong> {best_tflops_model} "
        item += f"with {best_tflops:.4f} TFLOPS"
        
        # Add comparison vs CPU if available
        if best_tflops_model in grouped_data and 'CPU' in grouped_data[best_tflops_model] and \
           grouped_data[best_tflops_model]['CPU']['tflops'] is not None and \
           grouped_data[best_tflops_model]['CPU']['tflops'] > 0:
            cpu_tflops = grouped_data[best_tflops_model]['CPU']['tflops']
            ratio = best_tflops / cpu_tflops if cpu_tflops > 0 else 0
            item += f" ({ratio:.1f}x faster than CPU)"
            
        item += "</li>"
        list_items.append(item)
    
    return "".join(list_items)


def generate_report(results, output_path=None, include_charts=False):
    """Generate a report from benchmark results
    
    Args:
        results: List of benchmark results
        output_path: Path to save the report
        include_charts: Whether to include performance charts in the report
        
    Returns:
        HTML report as a string
    """
    if not results:
        return None
        
    # Create report directory and ID
    report_dir = _create_report_directory()
    report_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Set output path if not provided
    if output_path is None:
        output_path = os.path.join(report_dir, f"benchmark_report_{report_id}.html")
    else:
        # If a custom path is provided, check if it's absolute or relative
        if not os.path.isabs(output_path):
            # If it's a relative path, store it in the cache directory
            output_path = os.path.join(report_dir, output_path)
    
    # Create plots if requested
    plot_paths = {}
    if include_charts:
        plot_paths = _create_plots(results, report_dir, report_id)
    
    # Extract data for charts
    model_names = [r.model_name for r in results]
    inference_times = [r.inference_time_ms for r in results]
    backends = [r.backend for r in results]
    
    # Group by model name and backend
    grouped_data = {}
    grouped_models = set()
    for r in results:
        model_key = r.model_name
        grouped_models.add(model_key)
        
        # Initialize dictionary for this model if not exists
        if model_key not in grouped_data:
            grouped_data[model_key] = {}
            
        # Handle both old and new attribute names
        throughput = getattr(r, 'throughput_gb_s', None)
        if throughput is None:
            throughput = getattr(r, 'throughput_gbps', 0)
            
        # Use actual model size value or 0 to clearly indicate errors
        model_size = getattr(r, 'model_size_mb', 0)
            
        grouped_data[model_key][r.backend] = {
            'inference_time_ms': r.inference_time_ms,
            'memory_used_mb': r.memory_used_mb,
            'tflops': r.tflops if r.tflops is not None else 0,
            'throughput_gb_s': throughput,
            'model_size_mb': model_size  # Will show 0.0 for missing values
        }
    
    # Prepare data for charts
    chart_data = {
        'inference_time': {backend: [] for backend in set(backends)},
        'throughput': {backend: [] for backend in set(backends)},
    }
    
    for model in grouped_models:
        for backend in set(backends):
            if backend in grouped_data.get(model, {}):
                chart_data['inference_time'][backend].append(grouped_data[model][backend]['inference_time_ms'])
                chart_data['throughput'][backend].append(grouped_data[model][backend]['throughput_gb_s'])
            else:
                chart_data['inference_time'][backend].append(0)
                chart_data['throughput'][backend].append(0)
    
    # Create charts if enabled
    inference_chart = ''
    throughput_chart = ''
    
    if include_charts:
        inference_chart = _create_bar_chart(chart_data['inference_time'], 'Inference Time (ms)')
        throughput_chart = _create_bar_chart(chart_data['throughput'], 'Memory Throughput (GB/s)')
    
    # Check if we have any valid TFLOPS data (non-None values)
    has_valid_tflops = any(r.tflops is not None for r in results)
    
    # Create the table header
    header = "<tr><th>Model</th><th>Backend</th><th>Inference Time (ms)</th><th>Throughput (GB/s)</th>"
    if has_valid_tflops:
        header += "<th>TFLOPS</th>"
    header += "<th>Model Size (MB)</th></tr>"
    
    # Create table rows
    rows = []
    for r in results:
        throughput = r.throughput_gb_s if hasattr(r, 'throughput_gb_s') else r.throughput_gbps if hasattr(r, 'throughput_gbps') else 0
        model_size = r.model_size_mb if hasattr(r, 'model_size_mb') else 0
        
        row = f"<tr><td>{r.model_name}</td><td>{r.backend}</td><td>{r.inference_time_ms:.2f}</td><td>{throughput:.2f}</td>"
        if has_valid_tflops:
            tflops_value = r.tflops if r.tflops is not None else "-"
            tflops_display = f"{tflops_value:.4f}" if isinstance(tflops_value, (int, float)) else tflops_value
            row += f"<td>{tflops_display}</td>"
        row += f"<td>{model_size:.2f}</td></tr>"
        
        rows.append(row)
    
    # Get system info
    system_info = results[0].system_info if results and hasattr(results[0], 'system_info') else {}
    
    # Create JSON data for the email button
    json_results = []
    for r in results:
        # Get model size - explicitly use 0 to indicate missing/error values
        model_size_mb = getattr(r, 'model_size_mb', 0)
        
        result_dict = {
            'model_name': r.model_name,
            'backend': r.backend,
            'inference_time_ms': r.inference_time_ms,
            'memory_used_mb': r.memory_used_mb,
            'throughput_gb_s': getattr(r, 'throughput_gb_s', getattr(r, 'throughput_gbps', 0)),
            'tflops': r.tflops,
            'input_shape': str(r.input_shape),
            'model_size_mb': model_size_mb  # Will show 0.0 for missing values
        }
        json_results.append(result_dict)
    
    # Create system info section
    system_info_html = ""
    if system_info:
        # Get OS information, using the user-friendly macOS version if available
        os_display = system_info.get('macos_version') if 'macos_version' in system_info else \
                    f"{system_info.get('os', {}).get('name', 'Unknown')} {system_info.get('os', {}).get('release', '')}"
        
        system_info_html = f"""
        <div class="system-info">
            <h2>System Information</h2>
            <table class="info-table">
                <tr>
                    <td><strong>Device:</strong></td>
                    <td>{system_info.get('device_name', system_info.get('mac_model', 'Unknown'))}</td>
                </tr>
                <tr>
                    <td><strong>CPU:</strong></td>
                    <td>{system_info.get('cpu', {}).get('brand', 'Unknown')}</td>
                </tr>
                <tr>
                    <td><strong>Memory:</strong></td>
                    <td>{system_info.get('ram', {}).get('total_gb', 'Unknown')} GB</td>
                </tr>
                <tr>
                    <td><strong>OS:</strong></td>
                    <td>{os_display}</td>
                </tr>
                <tr>
                    <td><strong>Apple Silicon:</strong></td>
                    <td>{'Yes' if system_info.get('apple_silicon', False) else 'No'}</td>
                </tr>
            </table>
        </div>
        """
    
    # Generate key findings
    key_findings = _generate_key_findings(grouped_data, grouped_models)
    
    # Format the date for display
    timestamp_text = f"Report generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    # Prepare JSON data for email
    json_data = json.dumps({
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'system_info': system_info,
        'results': json_results
    }, indent=2)
    
    # Create footer with timestamp and ANEMLL branding
    footer_html = f"""
        <div class="footer">
            <p>{timestamp_text}</p>
            <p><strong>ANEMLL BENCH</strong></p>
            <p><a href="https://www.anemll.com" target="_blank">Visit ANEMLL.com</a></p>
            <p>Report stored in: {output_path}</p>
            
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
        </div>
    """
    
    # Create the HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Apple Neural Engine Benchmark Results</title>
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
            .chart-container {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
            }}
            .chart {{
                width: 48%;
                margin-bottom: 20px;
                padding: 10px;
                background-color: #fff;
                border-radius: 5px;
                box-shadow: 0 1px 5px rgba(0,0,0,0.1);
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
            .key-findings {{
                margin: 20px 0;
                padding: 15px;
                background-color: #f0f7ff;
                border-radius: 5px;
                border-left: 5px solid #0066cc;
            }}
            .key-findings ul {{
                margin: 0;
                padding-left: 20px;
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
            .chart-image {{
                max-width: 100%;
                height: auto;
                display: block;
                margin: 0 auto;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Apple Neural Engine Benchmark Results</h1>
            
            <div style="text-align: center; margin: 20px 0;">
                <button id="sendResultsBtn" style="background-color: #28a745; color: white; padding: 12px 24px; font-size: 18px; border: none; border-radius: 5px; cursor: pointer;">
                    Send Results to ANEMLL Team
                </button>
            </div>
            
            {system_info_html}
            
            <div class="key-findings">
                <h2>Key Findings</h2>
                <ul>
                    {key_findings}
                </ul>
            </div>
            
            <h2>Performance Metrics</h2>
            
            {f'''
            <div class="chart-container">
                <div class="chart">{inference_chart}</div>
                <div class="chart">{throughput_chart}</div>
            </div>
            ''' if include_charts else "<p>Charts disabled. Use include_charts=True to enable performance charts.</p>"}
            
            <h2>Detailed Results</h2>
            
            <table>
                {header}
                {"".join(rows)}
            </table>
            
            {footer_html}
        </div>
    </body>
    </html>
    """
    
    # Write the report to file
    with open(output_path, 'w') as f:
        f.write(html)
    
    return html 