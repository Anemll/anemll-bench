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
    
    # Try to get Neural Engine information if on Apple Silicon
    if system_info['apple_silicon']:
        # This is a placeholder - Apple doesn't provide direct API for ANE specs
        system_info['neural_engine'] = {
            'available': True,
            # Information below would need to be determined based on device model
            'cores': "Unknown",  # Different Apple chips have different ANE core counts
            'tflops': "Unknown", # Would vary by chip generation
        }
    
    return system_info


if __name__ == "__main__":
    # Print system info when run directly
    info = get_system_info()
    print(json.dumps(info, indent=2)) 