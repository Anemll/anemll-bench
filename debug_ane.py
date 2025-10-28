#!/usr/bin/env python3
"""
ANE Debugging Script

This script helps diagnose why models might not be running on ANE.
Run this before benchmarking to identify potential issues.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from anemll_bench.utils.ane_verification import run_ane_diagnostic, verify_model_ane_compatibility
from anemll_bench.utils.system_info import get_system_info


def main():
    print("ANE Debugging Tool")
    print("=" * 60)
    
    # Run comprehensive diagnostic
    diagnostic = run_ane_diagnostic()
    
    print("\n" + "=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    
    # Get detailed system info
    system_info = get_system_info()
    
    print(f"Platform: {system_info['os']['name']} {system_info['os']['release']}")
    print(f"Architecture: {system_info['os']['name']}")
    print(f"Apple Silicon: {system_info['apple_silicon']}")
    
    if system_info['apple_silicon']:
        ane_info = system_info['neural_engine']
        print(f"ANE Available: {ane_info['available']}")
        if 'chip_model' in ane_info:
            print(f"Chip Model: {ane_info['chip_model']}")
        if 'cores' in ane_info:
            print(f"ANE Cores: {ane_info['cores']}")
        if 'generation' in ane_info:
            print(f"Chip Generation: {ane_info['generation']}")
        if 'capabilities' in ane_info:
            print(f"ANE Capabilities: {', '.join(ane_info['capabilities'])}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    # Provide recommendations based on diagnostic
    if diagnostic['overall_status'] == 'ANE should be available':
        print("✓ ANE hardware and software support detected")
        print("✓ Your system should be able to run models on ANE")
        print("\nIf models are still not using ANE:")
        print("1. Ensure models are loaded with CPU_AND_NE compute units")
        print("2. Check that models are ML Programs (ANE-optimized)")
        print("3. Verify macOS version compatibility")
        print("4. Try running the benchmark with enhanced debugging")
        
    elif diagnostic['overall_status'] == 'Hardware OK, CoreML issues':
        print("⚠ ANE hardware detected but CoreML issues found")
        print("\nRecommendations:")
        print("1. Update CoreML Tools: pip install --upgrade coremltools")
        print("2. Check CoreML Tools version compatibility")
        print("3. Verify Python environment setup")
        
    elif diagnostic['overall_status'] == 'Hardware not available':
        print("✗ ANE hardware not available")
        print("\nThis system cannot use ANE acceleration")
        print("Models will run on CPU/GPU instead")
        
    else:
        print("? Unknown issues detected")
        print("Please check the diagnostic output above")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    
    print("1. Run your benchmark with enhanced debugging:")
    print("   python -m anemll_bench --model your_model --ane-only")
    print()
    print("2. Check model compatibility:")
    print("   python debug_ane.py --model /path/to/your/model.mlmodel")
    print()
    print("3. For platform models:")
    print("   python examples/benchmark_all_models.py --use-local")
    
    # Check if a model path was provided
    if len(sys.argv) > 1 and sys.argv[1] == '--model' and len(sys.argv) > 2:
        model_path = sys.argv[2]
        print(f"\n" + "=" * 60)
        print(f"MODEL COMPATIBILITY CHECK: {model_path}")
        print("=" * 60)
        
        model_info = verify_model_ane_compatibility(model_path)
        
        print(f"Model Exists: {model_info['model_exists']}")
        if model_info['model_format']:
            print(f"Model Format: {model_info['model_format']}")
        print(f"ANE Optimized: {model_info['ane_optimized']}")
        print(f"ML Program: {model_info['ml_program']}")
        
        if model_info['issues']:
            print(f"\nIssues:")
            for issue in model_info['issues']:
                print(f"  - {issue}")
        else:
            print(f"\n✓ No compatibility issues detected")


if __name__ == "__main__":
    main()
