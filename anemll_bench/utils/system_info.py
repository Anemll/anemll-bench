"""
Utility functions to collect system information for benchmarking reports
"""

import platform
import os
import subprocess
import json
import psutil


def get_mac_model_identifier():
    """Get the Mac model identifier, e.g., 'MacBookPro18,3'"""
    try:
        result = subprocess.run(['sysctl', '-n', 'hw.model'], 
                               capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except:
        return "Unknown Mac Model"


def get_macos_version():
    """Get a more user-friendly macOS version string"""
    if platform.system() != "Darwin":
        return None
    
    try:
        # Try using sw_vers command for more accurate information
        product_name = subprocess.run(['sw_vers', '-productName'], 
                                    capture_output=True, text=True, check=True).stdout.strip()
        product_version = subprocess.run(['sw_vers', '-productVersion'], 
                                       capture_output=True, text=True, check=True).stdout.strip()
        build_version = subprocess.run(['sw_vers', '-buildVersion'], 
                                      capture_output=True, text=True, check=True).stdout.strip()
        
        # Map of macOS version numbers to marketing names (as backup if productName doesn't work)
        version_names = {
            "10.15": "Catalina",
            "11": "Big Sur",
            "12": "Monterey",
            "13": "Ventura",
            "14": "Sonoma",
            "15": "Sequoia"  
        }
        
        # If product_name is not "macOS", use a more descriptive name
        if product_name and product_name != "macOS":
            return f"{product_name} {product_version} ({build_version})"
        
        # Otherwise determine name from version number
        major_version = product_version.split('.')[0]
        if major_version == "10":
            # For older versions, we need the second digit too
            major_minor = '.'.join(product_version.split('.')[:2])
            version_name = version_names.get(major_minor, "")
        else:
            version_name = version_names.get(major_version, "")
        
        # Format the full version string
        if version_name:
            return f"macOS {version_name} {product_version} ({build_version})"
        else:
            return f"macOS {product_version} ({build_version})"
    except:
        # Fall back to platform.mac_ver() if the commands fail
        try:
            mac_ver = platform.mac_ver()
            return f"macOS {mac_ver[0]}"
        except:
            # Ultimate fallback to platform.version()
            return f"macOS {platform.version()}"


def get_cpu_info():
    """Get detailed CPU information"""
    info = {
        'brand': platform.processor(),
        'architecture': platform.machine(),
        'cores': psutil.cpu_count(logical=False),
        'threads': psutil.cpu_count(logical=True),
    }
    
    # Get more detailed Apple Silicon info if available
    if platform.machine() == "arm64":
        try:
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True, check=True)
            info['brand'] = result.stdout.strip()
        except:
            pass
    
    return info


def get_ram_info():
    """Get RAM information"""
    mem = psutil.virtual_memory()
    return {
        'total_gb': round(mem.total / (1024**3), 2),
        'available_gb': round(mem.available / (1024**3), 2),
    }


def get_ane_info():
    """
    Get detailed Apple Neural Engine (ANE) information for Apple Silicon devices.
    
    Returns:
        Dictionary with ANE capabilities and specifications
    """
    ane_info = {
        'available': True,
        'cores': 'Unknown',
        'tflops': 'Unknown',
        'chip_model': 'Unknown',
        'ane_generation': 'Unknown',
        'capabilities': []
    }
    
    try:
        # Get Mac model identifier
        mac_model = get_mac_model_identifier()
        if mac_model:
            ane_info['chip_model'] = mac_model
            
            # Map Mac models to ANE specifications
            ane_specs = {
                # M1 Series
                'MacBookAir10,1': {'cores': 16, 'generation': 'M1', 'tflops': '11'},
                'MacBookPro17,1': {'cores': 16, 'generation': 'M1', 'tflops': '11'},
                'Macmini9,1': {'cores': 16, 'generation': 'M1', 'tflops': '11'},
                'iMac21,1': {'cores': 16, 'generation': 'M1', 'tflops': '11'},
                'iMac21,2': {'cores': 16, 'generation': 'M1', 'tflops': '11'},
                
                # M1 Pro
                'MacBookPro18,1': {'cores': 16, 'generation': 'M1 Pro', 'tflops': '11'},
                'MacBookPro18,2': {'cores': 16, 'generation': 'M1 Pro', 'tflops': '11'},
                'MacBookPro18,3': {'cores': 16, 'generation': 'M1 Pro', 'tflops': '11'},
                'MacBookPro18,4': {'cores': 16, 'generation': 'M1 Pro', 'tflops': '11'},
                
                # M1 Max
                'MacBookPro18,1': {'cores': 32, 'generation': 'M1 Max', 'tflops': '22'},
                'MacBookPro18,2': {'cores': 32, 'generation': 'M1 Max', 'tflops': '22'},
                'MacBookPro18,3': {'cores': 32, 'generation': 'M1 Max', 'tflops': '22'},
                'MacBookPro18,4': {'cores': 32, 'generation': 'M1 Max', 'tflops': '22'},
                
                # M1 Ultra
                'MacStudio1,1': {'cores': 64, 'generation': 'M1 Ultra', 'tflops': '44'},
                
                # M2 Series
                'MacBookAir13,2': {'cores': 16, 'generation': 'M2', 'tflops': '15.8'},
                'MacBookPro18,1': {'cores': 16, 'generation': 'M2', 'tflops': '15.8'},
                'Macmini9,1': {'cores': 16, 'generation': 'M2', 'tflops': '15.8'},
                
                # M2 Pro
                'MacBookPro19,1': {'cores': 16, 'generation': 'M2 Pro', 'tflops': '15.8'},
                'MacBookPro19,2': {'cores': 16, 'generation': 'M2 Pro', 'tflops': '15.8'},
                
                # M2 Max
                'MacBookPro19,1': {'cores': 32, 'generation': 'M2 Max', 'tflops': '31.6'},
                'MacBookPro19,2': {'cores': 32, 'generation': 'M2 Max', 'tflops': '31.6'},
                
                # M2 Ultra
                'MacPro7,1': {'cores': 64, 'generation': 'M2 Ultra', 'tflops': '63.2'},
                
                # M3 Series
                'MacBookAir15,1': {'cores': 16, 'generation': 'M3', 'tflops': '18'},
                'MacBookPro18,1': {'cores': 16, 'generation': 'M3', 'tflops': '18'},
                'iMac24,1': {'cores': 16, 'generation': 'M3', 'tflops': '18'},
                
                # M3 Pro
                'MacBookPro18,1': {'cores': 16, 'generation': 'M3 Pro', 'tflops': '18'},
                'MacBookPro18,2': {'cores': 16, 'generation': 'M3 Pro', 'tflops': '18'},
                
                # M3 Max
                'MacBookPro18,1': {'cores': 32, 'generation': 'M3 Max', 'tflops': '36'},
                'MacBookPro18,2': {'cores': 32, 'generation': 'M3 Max', 'tflops': '36'},
                
                # M3 Ultra
                'MacStudio1,1': {'cores': 64, 'generation': 'M3 Ultra', 'tflops': '72'},
                
                # M4 Series
                'MacBookAir15,1': {'cores': 16, 'generation': 'M4', 'tflops': '38'},
                'MacBookPro18,1': {'cores': 16, 'generation': 'M4', 'tflops': '38'},
                
                # M4 Pro
                'MacBookPro18,1': {'cores': 16, 'generation': 'M4 Pro', 'tflops': '38'},
                'MacBookPro18,2': {'cores': 16, 'generation': 'M4 Pro', 'tflops': '38'},
                
                # M4 Max
                'MacBookPro18,1': {'cores': 32, 'generation': 'M4 Max', 'tflops': '76'},
                'MacBookPro18,2': {'cores': 32, 'generation': 'M4 Max', 'tflops': '76'},
            }
            
            if mac_model in ane_specs:
                specs = ane_specs[mac_model]
                ane_info.update(specs)
            else:
                # Try to infer from model name patterns
                if 'MacBookAir' in mac_model:
                    ane_info.update({'cores': 16, 'generation': 'Unknown M-series', 'tflops': 'Unknown'})
                elif 'MacBookPro' in mac_model:
                    ane_info.update({'cores': 16, 'generation': 'Unknown M-series Pro/Max', 'tflops': 'Unknown'})
                elif 'MacStudio' in mac_model:
                    ane_info.update({'cores': 64, 'generation': 'Unknown M-series Ultra', 'tflops': 'Unknown'})
                elif 'iMac' in mac_model:
                    ane_info.update({'cores': 16, 'generation': 'Unknown M-series', 'tflops': 'Unknown'})
                elif 'Macmini' in mac_model:
                    ane_info.update({'cores': 16, 'generation': 'Unknown M-series', 'tflops': 'Unknown'})
        
        # Add capabilities based on macOS version
        macos_version = get_macos_version()
        if macos_version:
            if macos_version >= '15.0':
                ane_info['capabilities'].extend([
                    'Enhanced ANE support',
                    'Improved CoreML integration',
                    'Better memory management'
                ])
            elif macos_version >= '14.0':
                ane_info['capabilities'].extend([
                    'Standard ANE support',
                    'CoreML integration',
                    'Basic memory management'
                ])
            else:
                ane_info['capabilities'].extend([
                    'Limited ANE support',
                    'Legacy CoreML integration'
                ])
        
        # Add general capabilities
        ane_info['capabilities'].extend([
            'Neural network acceleration',
            'Machine learning inference',
            'CoreML model execution'
        ])
        
    except Exception as e:
        print(f"Warning: Could not determine detailed ANE info: {e}")
        ane_info['error'] = str(e)
    
    return ane_info


def get_system_info():
    """Collect comprehensive system information"""
    system_info = {
        'os': {
            'name': platform.system(),
            'version': platform.version(),
            'release': platform.release(),
        },
        'cpu': get_cpu_info(),
        'ram': get_ram_info(),
        'python_version': platform.python_version(),
        'mac_model': get_mac_model_identifier() if platform.system() == "Darwin" else None,
    }
    
    # Get a more user-friendly macOS version if on macOS
    if platform.system() == "Darwin":
        system_info['macos_version'] = get_macos_version()
    
    # Check if we're running on Apple Silicon
    system_info['apple_silicon'] = platform.machine() == "arm64" and platform.system() == "Darwin"
    
    # Get detailed ANE information if on Apple Silicon
    if system_info['apple_silicon']:
        ane_info = get_ane_info()
        system_info['neural_engine'] = ane_info
    else:
        system_info['neural_engine'] = {
            'available': False,
            'reason': 'Not running on Apple Silicon'
        }
    
    return system_info


if __name__ == "__main__":
    # Print system info when run directly
    info = get_system_info()
    print(json.dumps(info, indent=2)) 