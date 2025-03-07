"""
Unit tests for report uploader module
"""

import pytest
import os
import json
import tempfile
from unittest.mock import patch, MagicMock

# Add parent directory to path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from anemll_bench.reports.report_uploader import ReportUploader


def test_uploader_initialization():
    """Test uploader initialization"""
    uploader = ReportUploader()
    assert uploader is not None
    assert uploader.service == "gist"
    
    # Test with different service
    uploader = ReportUploader(service="jsonbin")
    assert uploader.service == "jsonbin"
    
    # Test with invalid service
    with pytest.raises(ValueError):
        uploader.upload({"test": "data"})


@patch('requests.post')
def test_gist_upload(mock_post):
    """Test uploading to GitHub Gist"""
    # Mock environment variable
    with patch.dict(os.environ, {"GITHUB_TOKEN": "fake_token"}):
        # Create mock response
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"html_url": "https://gist.github.com/test"}
        mock_post.return_value = mock_response
        
        # Create uploader and test
        uploader = ReportUploader(service="gist")
        result = uploader.upload(
            report_data={"test": "data"},
            title="Test Report",
            description="Test Description"
        )
        
        # Verify result
        assert result == "https://gist.github.com/test"
        
        # Verify correct API call
        assert mock_post.called
        args, kwargs = mock_post.call_args
        assert args[0] == "https://api.github.com/gists"
        assert "json" in kwargs
        assert kwargs["json"]["description"] == "Test Description"
        assert kwargs["json"]["public"] is True
        
        # Test error handling
        with pytest.raises(ValueError):
            # Test with no token
            with patch.dict(os.environ, {"GITHUB_TOKEN": ""}, clear=True):
                uploader.upload({"test": "data"})


@patch('requests.post')
def test_jsonbin_upload(mock_post):
    """Test uploading to JSONBin"""
    # Mock environment variable
    with patch.dict(os.environ, {"JSONBIN_API_KEY": "fake_key"}):
        # Create mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"metadata": {"id": "test-bin-id"}}
        mock_post.return_value = mock_response
        
        # Create uploader and test
        uploader = ReportUploader(service="jsonbin")
        result = uploader.upload(
            report_data={"test": "data", "system_info": {"mac_model": "MacPro"}},
            title="Test Report",
            description="Test Description"
        )
        
        # Verify result
        assert result == "https://jsonbin.io/b/test-bin-id"
        
        # Verify correct API call
        assert mock_post.called
        args, kwargs = mock_post.call_args
        assert args[0] == "https://api.jsonbin.io/v3/b"
        assert "json" in kwargs
        assert "_metadata" in kwargs["json"]
        assert kwargs["json"]["_metadata"]["title"] == "Test Report"
        
        # Test error handling
        with pytest.raises(ValueError):
            # Test with no API key
            with patch.dict(os.environ, {"JSONBIN_API_KEY": ""}, clear=True):
                uploader.upload({"test": "data"})


@patch('requests.post')
def test_pastebin_upload(mock_post):
    """Test uploading to Pastebin"""
    # Mock environment variable
    with patch.dict(os.environ, {"PASTEBIN_API_KEY": "fake_key"}):
        # Create mock response
        mock_response = MagicMock()
        mock_response.text = "https://pastebin.com/test"
        mock_post.return_value = mock_response
        
        # Create uploader and test
        uploader = ReportUploader(service="pastebin")
        result = uploader.upload(
            report_data={"test": "data"},
            title="Test Report",
            description="Test Description"
        )
        
        # Verify result
        assert result == "https://pastebin.com/test"
        
        # Verify correct API call
        assert mock_post.called
        args, kwargs = mock_post.call_args
        assert args[0] == "https://pastebin.com/api/api_post.php"
        assert "data" in kwargs
        assert kwargs["data"]["api_paste_name"] == "Test Report"
        
        # Test error handling
        with pytest.raises(ValueError):
            # Test with no API key
            with patch.dict(os.environ, {"PASTEBIN_API_KEY": ""}, clear=True):
                uploader.upload({"test": "data"})
                
        # Test response error
        mock_response.text = "Bad API request"
        with pytest.raises(Exception):
            uploader.upload({"test": "data"})


if __name__ == "__main__":
    # Run tests manually
    test_uploader_initialization()
    print("All tests passed!") 