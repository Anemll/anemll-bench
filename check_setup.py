#!/usr/bin/env python3
"""
Setup Verification Script for ANEMLL-Bench

This script checks if your environment is properly configured for Apple Neural Engine (ANE) access.
"""

import platform
import sys
import subprocess
import os


def check_python_architecture():
    """Check if Python is running on the correct architecture"""
    arch = platform.machine()
    system_arch = os.uname().machine if hasattr(os, 'uname') else 'unknown'
    
    print(f"System Architecture: {system_arch}")
    print(f"Python Architecture: {arch}")
    
    if arch == 'arm64' and system_arch == 'arm64':
        print("✅ Python is running natively on Apple Silicon")
        return True
    elif arch == 'x86_64' and system_arch == 'arm64':
        print("❌ Python is running under Rosetta (x86_64 emulation)")
        print("   This will prevent ANE access!")
        return False
    elif arch == 'x86_64' and system_arch == 'x86_64':
        print("ℹ️  Running on Intel Mac (ANE not available)")
        return False
    else:
        print(f"⚠️  Unknown architecture combination: {arch} on {system_arch}")
        return False


def check_homebrew():
    """Check Homebrew installation and architecture"""
    print("\n=== Homebrew Check ===")
    
    # Check native ARM64 Homebrew
    native_brew = "/opt/homebrew/bin/brew"
    if os.path.exists(native_brew):
        try:
            result = subprocess.run([native_brew, "--version"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("✅ Native ARM64 Homebrew found: /opt/homebrew/bin/brew")
                return True
        except:
            pass
    
    # Check default Homebrew
    try:
        result = subprocess.run(["brew", "--version"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            brew_path = subprocess.run(["which", "brew"], 
                                     capture_output=True, text=True).stdout.strip()
            print(f"⚠️  Default Homebrew found: {brew_path}")
            print("   This may be x86_64 under Rosetta")
            return False
    except:
        pass
    
    print("❌ No Homebrew found")
    return False


def check_python_versions():
    """Check available Python versions"""
    print("\n=== Python Versions Check ===")
    
    python_paths = [
        ("System Python", "/usr/bin/python3"),
        ("ARM64 Homebrew Python", "/opt/homebrew/opt/python@3.9/bin/python3.9"),
        ("x86_64 Homebrew Python", "/usr/local/opt/python@3.9/bin/python3.9"),
    ]
    
    for name, path in python_paths:
        if os.path.exists(path):
            try:
                result = subprocess.run([path, "-c", "import platform; print(platform.machine())"], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    arch = result.stdout.strip()
                    print(f"✅ {name}: {path} (Architecture: {arch})")
                else:
                    print(f"❌ {name}: {path} (Error)")
            except:
                print(f"❌ {name}: {path} (Error)")
        else:
            print(f"❌ {name}: Not found")


def check_coreml():
    """Check CoreML Tools installation"""
    print("\n=== CoreML Tools Check ===")
    
    try:
        import coremltools as ct
        print(f"✅ CoreML Tools: {ct.__version__}")
        
        # Check compute units
        compute_units = [str(cu) for cu in ct.ComputeUnit]
        ane_units = [cu for cu in compute_units if 'NE' in cu or 'ALL' in cu]
        if ane_units:
            print(f"✅ ANE compute units available: {ane_units}")
        else:
            print("❌ No ANE compute units found")
            
        return True
    except ImportError:
        print("❌ CoreML Tools not installed")
        return False


def main():
    print("ANEMLL-Bench Setup Verification")
    print("=" * 40)
    
    # Check Python architecture
    python_ok = check_python_architecture()
    
    # Check Homebrew
    homebrew_ok = check_homebrew()
    
    # Check Python versions
    check_python_versions()
    
    # Check CoreML
    coreml_ok = check_coreml()
    
    # Summary
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    
    if python_ok and coreml_ok:
        print("✅ Your setup looks good for ANE access!")
        print("   You should be able to run models on the Apple Neural Engine.")
    elif not python_ok:
        print("❌ Python architecture issue detected.")
        print("   Rebuild your environment with native ARM64 Python:")
        print("   /usr/bin/python3 -m venv env-anemll-bench")
        print("   OR")
        print("   /opt/homebrew/bin/brew install python@3.9")
        print("   /opt/homebrew/opt/python@3.9/bin/python3.9 -m venv env-anemll-bench")
    elif not coreml_ok:
        print("❌ CoreML Tools not installed.")
        print("   Install with: pip install coremltools")
    else:
        print("⚠️  Mixed issues detected. Check the details above.")
    
    print("\nRun 'python debug_ane.py' for detailed ANE diagnostics.")


if __name__ == "__main__":
    main()
