#!/usr/bin/env python3

import os
import sys
import webbrowser
import subprocess

# Get the most recent report in ~/.cache/anemll-bench/reports/
reports_dir = os.path.expanduser("~/.cache/anemll-bench/reports/")
if not os.path.exists(reports_dir):
    print(f"Reports directory {reports_dir} does not exist.")
    sys.exit(1)

# List HTML files in the reports directory
html_files = [f for f in os.listdir(reports_dir) if f.endswith('.html')]
if not html_files:
    print(f"No HTML files found in {reports_dir}")
    sys.exit(1)

# Sort by modification time (most recent first)
html_files.sort(key=lambda f: os.path.getmtime(os.path.join(reports_dir, f)), reverse=True)

# Get the most recent report
report_file = os.path.join(reports_dir, html_files[0])
print(f"Opening most recent report: {report_file}")
print(f"File exists: {os.path.exists(report_file)}")

# Try multiple ways to open the file in a browser

# Method 1: webbrowser.open
try:
    file_url = f"file://{os.path.abspath(report_file)}"
    print(f"Method 1: Attempting to open with webbrowser.open: {file_url}")
    #result = webbrowser.open(file_url)
    #print(f"Method 1 result: {result}")
except Exception as e:
    print(f"Method 1 failed: {e}")

# Method 2: webbrowser.get('safari').open
try:
    file_url = f"file://{os.path.abspath(report_file)}"
    print(f"Method 2: Attempting to open with Safari: {file_url}")
    safari = webbrowser.get('safari')
    result = safari.open(file_url)
    print(f"Method 2 result: {result}")
except Exception as e:
    print(f"Method 2 failed: {e}")

# Method 3: subprocess.run with 'open' command
try:
    print(f"Method 3: Attempting to open with 'open' command: {report_file}")
    result = subprocess.run(['open', report_file], check=True)
    print(f"Method 3 result: {result}")
except Exception as e:
    print(f"Method 3 failed: {e}")

print("Done!") 