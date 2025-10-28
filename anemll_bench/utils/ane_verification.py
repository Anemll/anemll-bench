"""
Apple Neural Engine (ANE) Verification Utility

This utility helps diagnose ANE availability and model execution issues.
"""

import platform
import subprocess
import sys
from typing import Dict, Any, Optional


def check_ane_hardware() -> Dict[str, Any]:
    """
    Check if ANE hardware is available on the system.
    
    Returns:
        Dictionary with ANE hardware information
    """
    ane_info = {
        'hardware_available': False,
        'platform': platform.system(),
        'architecture': platform.machine(),
        'macos_version': None,
        'chip_model': None,
        'ane_cores': None,
        'issues': []
    }
    
    # Check if we're on macOS
    if platform.system() != 'Darwin':
        ane_info['issues'].append('Not running on macOS')
        return ane_info
    
    # Check if we're on Apple Silicon
    if platform.machine() != 'arm64':
        ane_info['issues'].append('Not running on Apple Silicon (arm64)')
        return ane_info
    
    ane_info['hardware_available'] = True
    
    # Get macOS version
    try:
        macos_version = platform.mac_ver()[0]
        ane_info['macos_version'] = macos_version
        
        # Check macOS version compatibility
        major_version = int(macos_version.split('.')[0])
        if major_version < 14:
            ane_info['issues'].append(f'macOS {major_version} has limited ANE support')
        elif major_version >= 15:
            ane_info['issues'].append(f'macOS {major_version} has enhanced ANE support')
    except Exception as e:
        ane_info['issues'].append(f'Could not determine macOS version: {e}')
    
    # Try to get chip model
    try:
        result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            chip_model = result.stdout.strip()
            ane_info['chip_model'] = chip_model
            
            # Map chip models to ANE cores
            if 'M1' in chip_model:
                if 'Ultra' in chip_model:
                    ane_info['ane_cores'] = 64
                elif 'Max' in chip_model:
                    ane_info['ane_cores'] = 32
                else:
                    ane_info['ane_cores'] = 16
            elif 'M2' in chip_model:
                if 'Ultra' in chip_model:
                    ane_info['ane_cores'] = 64
                elif 'Max' in chip_model:
                    ane_info['ane_cores'] = 32
                else:
                    ane_info['ane_cores'] = 16
            elif 'M3' in chip_model:
                if 'Ultra' in chip_model:
                    ane_info['ane_cores'] = 64
                elif 'Max' in chip_model:
                    ane_info['ane_cores'] = 32
                else:
                    ane_info['ane_cores'] = 16
            elif 'M4' in chip_model:
                if 'Max' in chip_model:
                    ane_info['ane_cores'] = 32
                else:
                    ane_info['ane_cores'] = 16
        else:
            ane_info['issues'].append('Could not determine chip model')
    except Exception as e:
        ane_info['issues'].append(f'Error getting chip model: {e}')
    
    return ane_info


def check_coreml_ane_support() -> Dict[str, Any]:
    """
    Check CoreML ANE support and configuration.
    
    Returns:
        Dictionary with CoreML ANE information
    """
    coreml_info = {
        'coreml_available': False,
        'version': None,
        'ane_support': False,
        'compute_units': [],
        'issues': []
    }
    
    try:
        import coremltools as ct
        coreml_info['coreml_available'] = True
        coreml_info['version'] = ct.__version__
        
        # Check available compute units
        try:
            compute_units = [str(cu) for cu in ct.ComputeUnit]
            coreml_info['compute_units'] = compute_units
            
            # Check if ANE compute units are available
            ane_units = [cu for cu in compute_units if 'NE' in cu or 'ALL' in cu]
            if ane_units:
                coreml_info['ane_support'] = True
            else:
                coreml_info['issues'].append('No ANE compute units found')
                
        except Exception as e:
            coreml_info['issues'].append(f'Error checking compute units: {e}')
            
    except ImportError:
        coreml_info['issues'].append('CoreML Tools not installed')
    except Exception as e:
        coreml_info['issues'].append(f'Error importing CoreML Tools: {e}')
    
    return coreml_info


def verify_model_ane_compatibility(model_path: str) -> Dict[str, Any]:
    """
    Verify if a model is compatible with ANE execution.
    
    Args:
        model_path: Path to the CoreML model
        
    Returns:
        Dictionary with model ANE compatibility information
    """
    model_info = {
        'model_exists': False,
        'model_format': None,
        'ane_optimized': False,
        'ml_program': False,
        'issues': []
    }
    
    import os
    
    # Check if model exists
    if not os.path.exists(model_path):
        model_info['issues'].append(f'Model file not found: {model_path}')
        return model_info
    
    model_info['model_exists'] = True
    
    # Determine model format
    if model_path.endswith('.mlmodelc'):
        model_info['model_format'] = 'Compiled (.mlmodelc)'
    elif model_path.endswith('.mlmodel'):
        model_info['model_format'] = 'Uncompiled (.mlmodel)'
    elif model_path.endswith('.mlpackage'):
        model_info['model_format'] = 'Package (.mlpackage)'
    else:
        model_info['issues'].append(f'Unknown model format: {model_path}')
    
    # Try to load and analyze the model
    try:
        import coremltools as ct
        
        # Load model
        if model_path.endswith('.mlmodelc'):
            model = ct.models.CompiledMLModel(model_path)
        else:
            model = ct.models.MLModel(model_path)
        
        # Get model spec
        spec = model.get_spec()
        
        # Check if it's an ML Program (ANE-optimized)
        if hasattr(spec, 'mlProgram') and spec.mlProgram:
            model_info['ml_program'] = True
            model_info['ane_optimized'] = True
        else:
            model_info['issues'].append('Model is not an ML Program (may not be ANE-optimized)')
        
        # Check spec version
        spec_version = spec.specificationVersion
        if spec_version < 5:
            model_info['issues'].append(f'Low spec version ({spec_version}), may not support ANE')
        
    except Exception as e:
        model_info['issues'].append(f'Error analyzing model: {e}')
    
    return model_info


def run_ane_diagnostic() -> Dict[str, Any]:
    """
    Run a comprehensive ANE diagnostic.
    
    Returns:
        Dictionary with complete ANE diagnostic information
    """
    print("Running ANE Diagnostic...")
    print("=" * 50)
    
    diagnostic = {
        'hardware': check_ane_hardware(),
        'coreml': check_coreml_ane_support(),
        'overall_status': 'Unknown'
    }
    
    # Determine overall status
    hardware_ok = diagnostic['hardware']['hardware_available']
    coreml_ok = diagnostic['coreml']['coreml_available'] and diagnostic['coreml']['ane_support']
    
    if hardware_ok and coreml_ok:
        diagnostic['overall_status'] = 'ANE should be available'
    elif hardware_ok and not coreml_ok:
        diagnostic['overall_status'] = 'Hardware OK, CoreML issues'
    elif not hardware_ok:
        diagnostic['overall_status'] = 'Hardware not available'
    else:
        diagnostic['overall_status'] = 'Unknown issues'
    
    # Print results
    print(f"\nHardware Check:")
    print(f"  - ANE Hardware Available: {hardware_ok}")
    if diagnostic['hardware']['chip_model']:
        print(f"  - Chip Model: {diagnostic['hardware']['chip_model']}")
    if diagnostic['hardware']['ane_cores']:
        print(f"  - ANE Cores: {diagnostic['hardware']['ane_cores']}")
    if diagnostic['hardware']['macos_version']:
        print(f"  - macOS Version: {diagnostic['hardware']['macos_version']}")
    
    print(f"\nCoreML Check:")
    print(f"  - CoreML Available: {coreml_ok}")
    if diagnostic['coreml']['version']:
        print(f"  - CoreML Version: {diagnostic['coreml']['version']}")
    print(f"  - ANE Support: {diagnostic['coreml']['ane_support']}")
    
    print(f"\nOverall Status: {diagnostic['overall_status']}")
    
    # Print issues
    all_issues = []
    all_issues.extend(diagnostic['hardware']['issues'])
    all_issues.extend(diagnostic['coreml']['issues'])
    
    if all_issues:
        print(f"\nIssues Found:")
        for issue in all_issues:
            print(f"  - {issue}")
    
    return diagnostic


if __name__ == "__main__":
    # Run diagnostic when executed directly
    diagnostic = run_ane_diagnostic()
    
    # Exit with appropriate code
    if diagnostic['overall_status'] == 'ANE should be available':
        sys.exit(0)
    else:
        sys.exit(1)
