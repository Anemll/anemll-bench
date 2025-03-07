"""
Utilities for synchronizing and managing model files.
"""

import os
import logging
import pathlib
import yaml
import requests
import zipfile
import shutil
from typing import Dict, Any, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for cache paths
# Use ~/.cache/anemll-bench/ for cache storage
HOME_DIR = str(pathlib.Path.home())
CACHE_DIR = os.path.join(HOME_DIR, ".cache", "anemll-bench")
META_FILE_PATH = os.path.join(CACHE_DIR, "meta.yalm")
MODELS_CACHE_DIR = os.path.join(CACHE_DIR, "models")

# Ensure cache directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODELS_CACHE_DIR, exist_ok=True)

class ModelSyncer:
    """
    Class responsible for synchronizing and managing model files.
    """
    
    def __init__(self):
        """
        Initialize the ModelSyncer.
        """
        # Make sure cache directories exist
        os.makedirs(CACHE_DIR, exist_ok=True)
        os.makedirs(MODELS_CACHE_DIR, exist_ok=True)
    
    def get_model_dir(self) -> str:
        """
        Get the path to the model directory.
        
        Returns:
            Path to the model directory
        """
        return MODELS_CACHE_DIR
    
    def read_meta_file(self) -> Dict:
        """
        Read the meta.yalm file containing model information.
        
        Returns:
            Dictionary containing model information
        """
        if not os.path.exists(META_FILE_PATH):
            logger.warning(f"Meta file not found at {META_FILE_PATH}")
            return {}
        
        try:
            with open(META_FILE_PATH, 'r') as file:
                meta_data = yaml.safe_load(file)
            return meta_data
        except Exception as e:
            logger.error(f"Error reading meta file: {e}")
            return {}
    
    def download_meta_file(self, force_update: bool = False) -> Dict:
        """
        Download the meta.yalm file from Hugging Face and store it locally.
        
        Args:
            force_update: If True, force download even if local file exists
            
        Returns:
            Dictionary containing model information
        """
        # URL to the meta.yalm file on Hugging Face
        meta_url = "https://huggingface.co/anemll/anemll-bench/raw/main/meta.yalm"
        
        # Check if we should download
        if not force_update and os.path.exists(META_FILE_PATH):
            logger.info(f"Using existing meta file at {META_FILE_PATH}")
            return self.read_meta_file()
        
        # Download the meta file
        logger.info(f"Downloading meta file from {meta_url}")
        try:
            response = requests.get(meta_url)
            response.raise_for_status()
            
            # Save the file
            os.makedirs(os.path.dirname(META_FILE_PATH), exist_ok=True)
            with open(META_FILE_PATH, 'w') as file:
                file.write(response.text)
            
            logger.info(f"Meta file downloaded and saved to {META_FILE_PATH}")
            return yaml.safe_load(response.text)
        except Exception as e:
            logger.error(f"Error downloading meta file: {e}")
            
            # If we fail to download but have a local version, use that
            if os.path.exists(META_FILE_PATH):
                logger.warning("Using existing local meta file as fallback")
                return self.read_meta_file()
            
            return {}
    
    def download_model(self, url: str, model_name: str, model_type: str, force_redownload: bool = False, allow_redownload: bool = True) -> Optional[str]:
        """
        Download a model from a URL.
        
        Args:
            url: URL to download the model from
            model_name: Name of the model
            model_type: Type of the model (mlmodelc or mlpackage)
            force_redownload: Whether to force re-download even if the model exists
            allow_redownload: Whether to allow redownloading if the zip file is corrupted
            
        Returns:
            Path to the downloaded model, or None if download failed
        """
        # Define the model directory path
        expected_model_dir = os.path.join(MODELS_CACHE_DIR, f"{model_name}.{model_type}")
        
        # Check if the model already exists
        if os.path.exists(expected_model_dir) and not force_redownload:
            logger.info(f"Model already exists at {expected_model_dir}")
            return expected_model_dir
        
        # Create a temporary directory for downloads
        download_dir = os.path.join(CACHE_DIR, "downloads")
        os.makedirs(download_dir, exist_ok=True)
        
        # Define zip file path
        zip_path = os.path.join(download_dir, f"{model_name}.zip")
        
        # If forcing redownload, remove the existing zip file and model directory
        if force_redownload:
            if os.path.exists(zip_path):
                os.remove(zip_path)
                logger.info(f"Removed existing zip file: {zip_path}")
            if os.path.exists(expected_model_dir):
                shutil.rmtree(expected_model_dir, ignore_errors=True)
                logger.info(f"Removed existing model directory: {expected_model_dir}")
        
        # Check if the zip file already exists
        need_download = True
        if os.path.exists(zip_path) and not force_redownload:
            # Verify it's a valid zip file
            try:
                with zipfile.ZipFile(zip_path) as test_zip:
                    test_zip.testzip()  # This will check the integrity of the zip file
                logger.info(f"Zip file already exists at {zip_path}, using cached version")
                need_download = False
            except zipfile.BadZipFile:
                if allow_redownload:
                    logger.warning(f"Existing zip file is corrupted: {zip_path}, re-downloading")
                    os.remove(zip_path)
                    need_download = True
                else:
                    logger.error(f"Existing zip file is corrupted: {zip_path}, but redownload is disabled")
                    return None
            except Exception as e:
                if allow_redownload:
                    logger.warning(f"Error checking zip file {zip_path}: {e}, re-downloading")
                    os.remove(zip_path)
                    need_download = True
                else:
                    logger.error(f"Error checking zip file {zip_path}: {e}, but redownload is disabled")
                    return None
        
        # Download if needed
        if need_download:
            # Download the model
            logger.info(f"Downloading model from {url}")
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                # Save zip file
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Downloaded zip file to {zip_path}")
            except Exception as e:
                logger.error(f"Error downloading model: {e}")
                return None
        
        # Extract the model
        try:
            # Remove existing model directory if it exists
            if os.path.exists(expected_model_dir):
                logger.info(f"Removing existing model directory: {expected_model_dir}")
                shutil.rmtree(expected_model_dir, ignore_errors=True)
            
            # Verify it's a valid zip file
            try:
                with zipfile.ZipFile(zip_path) as test_zip:
                    test_zip.testzip()
            except zipfile.BadZipFile:
                logger.error(f"Downloaded zip file is corrupted: {zip_path}")
                if not allow_redownload:
                    logger.error("Redownload is disabled. Cannot recover from corrupted zip file.")
                    return None
                elif force_redownload:
                    logger.error("Already tried force redownload but zip is still corrupted. Giving up.")
                    return None
                else:
                    logger.info("Attempting to re-download with force_redownload=True")
                    os.remove(zip_path)
                    return self.download_model(url, model_name, model_type, force_redownload=True, allow_redownload=allow_redownload)
            
            # Extract the model
            with zipfile.ZipFile(zip_path) as zip_ref:
                # Get information about the zip contents
                file_list = zip_ref.namelist()
                logger.info(f"ZIP file contains {len(file_list)} files/directories")
                
                # Print the top-level directories/files
                top_level = set()
                for file_path in file_list:
                    top_dir = file_path.split('/')[0] if '/' in file_path else file_path
                    top_level.add(top_dir)
                
                logger.info(f"Top-level entries in ZIP: {', '.join(top_level)}")
                
                # Extract the model
                logger.info(f"Extracting ZIP file to {MODELS_CACHE_DIR}")
                zip_ref.extractall(MODELS_CACHE_DIR)
            
            # Remove any __MACOSX directory
            macosx_dir = os.path.join(MODELS_CACHE_DIR, "__MACOSX")
            if os.path.exists(macosx_dir):
                logger.info(f"Removing __MACOSX directory: {macosx_dir}")
                shutil.rmtree(macosx_dir, ignore_errors=True)
            
            # Check if the model directory exists
            if os.path.exists(expected_model_dir):
                logger.info(f"Model extracted to {expected_model_dir}")
                return expected_model_dir
            else:
                logger.error(f"Model directory not found after extraction: {expected_model_dir}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting model: {e}")
            return None
    
    def get_model_path(self, model_name: str, check_online: bool = True, force_redownload: bool = False) -> str:
        """
        Get the path to a model, downloading it if necessary.
        
        Args:
            model_name: Name of the model
            check_online: Whether to check online for models not found locally
            force_redownload: Whether to force re-download the model even if it exists
            
        Returns:
            Path to the model
        """
        from .model_loader import get_platform_specific_models
        
        # Try to find the model in platform-specific models
        platform_models = get_platform_specific_models(check_online=False)
        
        # Look for the model in local models
        model_config = None
        for config in platform_models:
            if config.get("name") == model_name:
                model_config = config
                break
        
        # If not found locally and check_online is True, check online
        if model_config is None and check_online:
            logger.info(f"Model '{model_name}' not found locally. Checking online...")
            platform_models = get_platform_specific_models(check_online=True)
            
            for config in platform_models:
                if config.get("name") == model_name:
                    model_config = config
                    break
        
        # If still not found, raise error
        if model_config is None:
            available_models = [m.get("name") for m in platform_models]
            msg = f"No model found with name: {model_name}. Available models: {available_models}"
            logger.error(msg)
            raise ValueError(msg)
        
        # Extract model info
        url = model_config.get("url")
        model_type = model_config.get("type")
        
        if not url or not model_type:
            raise ValueError(f"Invalid model configuration for {model_name}: missing url or type")
        
        # Download or get the model path
        model_path = self.download_model(url, model_name, model_type, force_redownload=force_redownload)
        
        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"Failed to get model path for {model_name}")
        
        return model_path 