"""
Report uploader for anemll-bench.
"""

import os
import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class ReportUploader:
    """
    Class for uploading benchmark reports to various services.
    """
    
    def __init__(self, service: str = "jsonbin"):
        """
        Initialize the report uploader.
        
        Args:
            service: The service to upload to (jsonbin, gist, pastebin)
        """
        self.service = service
        self._check_credentials()
    
    def _check_credentials(self):
        """
        Check if the necessary credentials are available for the selected service.
        """
        if self.service == "jsonbin":
            if not os.environ.get("JSONBIN_API_KEY"):
                logger.warning("JSONBIN_API_KEY environment variable not set. Upload will fail.")
        elif self.service == "gist":
            if not os.environ.get("GITHUB_TOKEN"):
                logger.warning("GITHUB_TOKEN environment variable not set. Upload will fail.")
        elif self.service == "pastebin":
            if not os.environ.get("PASTEBIN_API_KEY"):
                logger.warning("PASTEBIN_API_KEY environment variable not set. Upload will fail.")
    
    def upload(self, report_path: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Upload a report to the selected service.
        
        Args:
            report_path: Path to the report file
            metadata: Additional metadata to include with the upload
            
        Returns:
            URL to the uploaded report, or None if upload failed
        """
        logger.info(f"Uploading report {report_path} to {self.service}")
        
        if not os.path.exists(report_path):
            logger.error(f"Report file {report_path} does not exist")
            return None
        
        # Read the report file
        with open(report_path, 'r') as f:
            content = f.read()
        
        # Upload based on the selected service
        if self.service == "jsonbin":
            return self._upload_to_jsonbin(content, metadata)
        elif self.service == "gist":
            return self._upload_to_gist(content, metadata, os.path.basename(report_path))
        elif self.service == "pastebin":
            return self._upload_to_pastebin(content, metadata)
        else:
            logger.error(f"Unknown upload service: {self.service}")
            return None
    
    def _upload_to_jsonbin(self, content: str, metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        """
        Upload to JSONBin.io.
        
        Args:
            content: The content to upload
            metadata: Additional metadata
            
        Returns:
            URL to the uploaded content, or None if upload failed
        """
        logger.info("JSONBin upload not implemented yet")
        return None
    
    def _upload_to_gist(self, content: str, metadata: Optional[Dict[str, Any]], filename: str) -> Optional[str]:
        """
        Upload to GitHub Gist.
        
        Args:
            content: The content to upload
            metadata: Additional metadata
            filename: The filename to use in the gist
            
        Returns:
            URL to the uploaded content, or None if upload failed
        """
        logger.info("GitHub Gist upload not implemented yet")
        return None
    
    def _upload_to_pastebin(self, content: str, metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        """
        Upload to Pastebin.
        
        Args:
            content: The content to upload
            metadata: Additional metadata
            
        Returns:
            URL to the uploaded content, or None if upload failed
        """
        logger.info("Pastebin upload not implemented yet")
        return None 