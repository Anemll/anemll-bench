#!/usr/bin/env python3
"""
Tests for the system_info module
"""

import unittest
import os
import sys

# Add parent directory to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from anemll_bench.utils.system_info import get_system_info, get_cpu_info, get_ram_info


class TestSystemInfo(unittest.TestCase):
    """Tests for system information gathering functions"""
    
    def test_get_system_info(self):
        """Test that system info is returned with expected keys"""
        info = get_system_info()
        
        # Check that main keys exist
        self.assertIn('os', info)
        self.assertIn('cpu', info)
        self.assertIn('ram', info)
        self.assertIn('python_version', info)
        
        # Check OS info
        self.assertIn('name', info['os'])
        self.assertIn('version', info['os'])
        self.assertIn('release', info['os'])
        
        # Check CPU info
        self.assertIn('brand', info['cpu'])
        self.assertIn('cores', info['cpu'])
        self.assertIn('threads', info['cpu'])
        
        # Check RAM info
        self.assertIn('total_gb', info['ram'])
        self.assertIn('available_gb', info['ram'])
        
    def test_get_cpu_info(self):
        """Test CPU info retrieval"""
        cpu_info = get_cpu_info()
        
        self.assertIn('brand', cpu_info)
        self.assertIn('architecture', cpu_info)
        self.assertIn('cores', cpu_info)
        self.assertIn('threads', cpu_info)
        
        # Make sure cores and threads are integers
        self.assertIsInstance(cpu_info['cores'], int)
        self.assertIsInstance(cpu_info['threads'], int)
        
        # Threads should be greater than or equal to cores
        self.assertGreaterEqual(cpu_info['threads'], cpu_info['cores'])
        
    def test_get_ram_info(self):
        """Test RAM info retrieval"""
        ram_info = get_ram_info()
        
        self.assertIn('total_gb', ram_info)
        self.assertIn('available_gb', ram_info)
        
        # RAM values should be positive
        self.assertGreater(ram_info['total_gb'], 0)
        self.assertGreater(ram_info['available_gb'], 0)
        
        # Available should be less than or equal to total
        self.assertLessEqual(ram_info['available_gb'], ram_info['total_gb'])


if __name__ == '__main__':
    unittest.main() 