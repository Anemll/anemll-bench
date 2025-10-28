"""
Utilities for loading and preparing models for benchmarking
"""

import os
import torch
from transformers import AutoModel, AutoModelForCausalLM
import coremltools as ct
from typing import Dict, Any, Tuple, Optional, List
import yaml
import platform
import requests
import zipfile
import io
import logging
import pathlib
import concurrent.futures

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

def read_meta_file() -> Dict:
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

def download_meta_file(force_update: bool = False) -> Dict:
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
        return read_meta_file()
    
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
            return read_meta_file()
        
        return {}

def get_macos_version() -> str:
    """
    Get the macOS version.
    
    Returns:
        String representation of macOS version category (e.g., 'macos_15_x')
    """
    system = platform.system()
    if system != "Darwin":
        return None
    
    version = platform.mac_ver()[0]
    major_version = int(version.split('.')[0])
    
    if major_version >= 15:
        return "macos_15_x"
    else:
        return f"macos_{major_version}_x"

def download_and_unzip_model(url: str, model_name: str, model_type: str, keep_zip: bool = True, allow_redownload: bool = True) -> str:
    """
    Download a model from a URL and unzip it.
    
    Args:
        url: URL to download the model from
        model_name: Name of the model
        model_type: Type of the model (mlmodelc or mlpackage)
        keep_zip: Whether to keep the downloaded zip file (default: True)
        allow_redownload: Whether to allow redownloading if the zip file is corrupted (default: True)
        
    Returns:
        Path to the unzipped model
    """
    # Define the model directory path
    expected_model_dir = os.path.join(MODELS_CACHE_DIR, f"{model_name}.{model_type}")
    
    # Check if the model already exists
    if os.path.exists(expected_model_dir):
        logger.info(f"Model already exists at {expected_model_dir}")
        
        # If this is an mlmodelc directory, check its structure
        if model_type == "mlmodelc":
            fix_mlmodelc_directory(expected_model_dir)
            
        return expected_model_dir
    
    # Create a temporary directory for downloads
    download_dir = os.path.join(CACHE_DIR, "downloads")
    os.makedirs(download_dir, exist_ok=True)
    
    # Define zip file path
    zip_path = os.path.join(download_dir, f"{model_name}.zip")
    
    # Check if the zip file already exists
    need_download = True
    if os.path.exists(zip_path):
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
    
    # Verify zip file validity
    try:
        with zipfile.ZipFile(zip_path) as test_zip:
            test_zip.testzip()
    except zipfile.BadZipFile:
        logger.error(f"Downloaded file is not a valid zip file: {zip_path}")
        # Try to download directly to the model directory
        try:
            logger.info(f"Trying direct download to model directory: {expected_model_dir}")
            if not os.path.exists(expected_model_dir):
                os.makedirs(expected_model_dir, exist_ok=True)
                
            # Download directly to the target directory
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Check if this is an mlmodelc and the URL is to a directory structure
            if model_type == "mlmodelc":
                # For mlmodelc, the download might be the directory contents directly
                with open(os.path.join(expected_model_dir, "model.mil"), 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Downloaded model file directly to {expected_model_dir}/model.mil")
                
                # Create other necessary directories/files
                os.makedirs(os.path.join(expected_model_dir, "weights"), exist_ok=True)
                os.makedirs(os.path.join(expected_model_dir, "analytics"), exist_ok=True)
                
                # Create a basic metadata.json file
                with open(os.path.join(expected_model_dir, "metadata.json"), 'w') as f:
                    f.write('{"framework": "coreml"}')
                
                # Check the model directory structure
                fix_mlmodelc_directory(expected_model_dir)
                
                return expected_model_dir
        except Exception as e:
            logger.error(f"Error with direct download: {e}")
            return None
    
    # Extract the model
    try:
        # Read the contents of the zip file for diagnostics
        with zipfile.ZipFile(zip_path) as zip_ref:
            file_list = zip_ref.namelist()
            logger.info(f"ZIP file contains {len(file_list)} files/directories")
            
            # Print the top-level directories/files
            top_level = set()
            for file_path in file_list:
                top_dir = file_path.split('/')[0] if '/' in file_path else file_path
                top_level.add(top_dir)
            
            logger.info(f"Top-level entries in ZIP: {', '.join(top_level)}")
            
            # Check for the expected directory structure
            expected_structure = False
            
            # Look for crucial files (model.mil)
            for file_path in file_list:
                if file_path.endswith('model.mil'):
                    logger.info(f"Found model.mil at: {file_path}")
                    expected_structure = True
                    break
            
            if not expected_structure:
                logger.warning(f"ZIP file does not contain expected structure (no model.mil found)")
            
            # Extract the model
            logger.info(f"Extracting ZIP file to {MODELS_CACHE_DIR}")
            zip_ref.extractall(MODELS_CACHE_DIR)
        
        # Clean up the zip file if not keeping it
        if not keep_zip:
            os.remove(zip_path)
            logger.info(f"Deleted ZIP file: {zip_path}")
        else:
            logger.info(f"Kept ZIP file at: {zip_path}")
        
        # Check if the model directory exists
        if os.path.exists(expected_model_dir):
            logger.info(f"Model extracted to {expected_model_dir}")
            
            # If this is an mlmodelc directory, check its structure
            if model_type == "mlmodelc":
                fix_mlmodelc_directory(expected_model_dir)
                
            return expected_model_dir
        else:
            # Look for what was actually extracted
            extracted_paths = []
            for item in os.listdir(MODELS_CACHE_DIR):
                full_path = os.path.join(MODELS_CACHE_DIR, item)
                if os.path.isdir(full_path) and full_path not in extracted_paths:
                    extracted_paths.append(full_path)
                    
            # If we found new directories, use the first one
            if extracted_paths:
                extracted_dir = extracted_paths[0]
                logger.warning(f"Expected model directory {expected_model_dir} not found, but found {extracted_dir}")
                
                # If the extracted directory is different from what we expected, create a symlink
                if extracted_dir != expected_model_dir:
                    # Rename the extracted directory to the expected name
                    import shutil
                    if os.path.exists(expected_model_dir):
                        shutil.rmtree(expected_model_dir)
                    os.rename(extracted_dir, expected_model_dir)
                    logger.info(f"Renamed {extracted_dir} to {expected_model_dir}")
                
                # If this is an mlmodelc directory, check its structure
                if model_type == "mlmodelc":
                    fix_mlmodelc_directory(expected_model_dir)
                
                return expected_model_dir
            else:
                logger.error(f"Model directory not found after extraction and couldn't find alternative directories")
                return None
    except Exception as e:
        logger.error(f"Error extracting model: {e}")
        return None

def get_platform_specific_models(check_online: bool = True) -> List[Dict]:
    """
    Get the models specific to the current platform.
    
    Args:
        check_online: Whether to check for an updated meta.yalm file online
        
    Returns:
        List of model configurations
    """
    # Try to get the meta file, checking online if requested
    meta_data = download_meta_file(force_update=check_online) if check_online else read_meta_file()
    
    if not meta_data or 'model_info' not in meta_data:
        logger.warning("No model info found in meta file")
        return []
    
    macos_version_key = get_macos_version()
    if not macos_version_key or macos_version_key not in meta_data['model_info']:
        logger.warning(f"No models found for {macos_version_key}")
        return []
    
    return meta_data['model_info'][macos_version_key]

def download_from_hf(model_id: str, use_cache: bool = True) -> torch.nn.Module:
    """
    Download a model from Hugging Face model hub
    
    Args:
        model_id: Hugging Face model identifier
        use_cache: Whether to use cached models
        
    Returns:
        PyTorch model
    """
    try:
        # Try loading as a causaslLM model first (for most LLMs)
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            cache_dir="./model_cache" if use_cache else None
        )
    except:
        # Fallback to generic model loading
        return AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            cache_dir="./model_cache" if use_cache else None
        )


def convert_to_coreml(model: torch.nn.Module, 
                     input_shape: Tuple[int, ...],
                     model_name: str,
                     compute_units: str = "ALL") -> str:
    """
    Convert PyTorch model to CoreML format for Apple Neural Engine
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        model_name: Name for the saved model
        compute_units: CoreML compute units (ALL, CPU_AND_GPU, CPU_ONLY, etc.)
        
    Returns:
        Path to the saved CoreML model
    """
    # Prepare example input
    example_input = torch.randn(input_shape)
    
    # Create traced model
    traced_model = torch.jit.trace(model, example_input)
    
    # Convert to CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="input", shape=input_shape)
        ],
        compute_units=ct.ComputeUnit[compute_units]
    )
    
    # Save the model
    model_dir = os.path.join("model_cache", "coreml")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.mlmodel")
    mlmodel.save(model_path)
    
    return model_path


def load_model(config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a model based on configuration
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Tuple of (model, model_info)
    """
    model_type = config.get("type", "pytorch")
    model_id = config.get("id")
    model_path = config.get("path")
    input_shape = config.get("input_shape")
    
    model_info = {
        "name": config.get("name", model_id or os.path.basename(model_path)),
        "type": model_type,
        "input_shape": input_shape
    }
    
    # Check if this is a request for a CoreML model without a specific path
    if model_type == "coreml" and not model_path:
        # Try to find a platform-specific model
        platform_models = get_platform_specific_models()
        model_name = config.get("name")
        
        for platform_model in platform_models:
            if platform_model.get("name") == model_name:
                url = platform_model.get("url")
                model_format = platform_model.get("type")
                
                if url and model_format:
                    # Download and unzip the model
                    model_path = download_and_unzip_model(url, model_name, model_format)
                    if model_path:
                        logger.info(f"Using platform-specific model: {model_path}")
                        break
    
    if model_type == "pytorch":
        if model_id:
            model = download_from_hf(model_id)
        elif model_path and os.path.exists(model_path):
            model = torch.load(model_path)
        else:
            raise ValueError("Either model_id or model_path must be provided")
            
    elif model_type == "coreml":
        if model_path and os.path.exists(model_path):
            model = ct.models.MLModel(model_path)
        elif model_id:
            # Download from HF and convert
            pt_model = download_from_hf(model_id)
            model_path = convert_to_coreml(
                pt_model, 
                tuple(input_shape), 
                os.path.basename(model_id).replace("/", "_")
            )
            model = ct.models.MLModel(model_path)
        else:
            raise ValueError("Either model_id or model_path must be provided, or a platform-specific model must be available")
            
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
        
    # Check if we need to fix the model directory
    if model_path.endswith('.mlmodelc'):
        fix_mlmodelc_directory(model_path)
    
    # Try to load the CoreML model
    try:
        logger.info(f"Loading model from: {model_path}")
        
        # Try different loading approaches
        try:
            # Standard approach
            model = ct.models.MLModel(model_path)
        except Exception as e1:
            logger.warning(f"Standard loading failed: {e1}")
            try:
                # Try with skip_model_load=True
                model = ct.models.MLModel(model_path, skip_model_load=True)
                logger.info("Successfully loaded model with skip_model_load=True")
            except Exception as e2:
                logger.warning(f"Loading with skip_model_load=True failed: {e2}")
                
                # Try the lowest level approach
                try:
                    # For mlmodelc with special format, try to load directly using a spec
                    import coremltools.models.model as ct_model
                    from coremltools.models._mil_converter import _MLModelProxy
                    
                    logger.info(f"Attempting direct spec-based loading")
                    spec_path = os.path.join(model_path, "model.mil")
                    if not os.path.exists(spec_path):
                        raise ValueError(f"model.mil not found in {model_path}")
                    
                    # Create a model directly
                    spec = ct_model.load_spec(spec_path)
                    model = _MLModelProxy(spec, model_path)
                    logger.info("Successfully loaded model with spec-based approach")
                except Exception as e3:
                    logger.error(f"All loading approaches failed")
                    raise e3
        
        return model, model_info
    except Exception as e:
        logger.error(f"Error loading CoreML model: {e}")
        
        # If this is the second attempt (force_redownload was true), give up or try local model
        if force_redownload:
            # If we've already tried force_redownload and it still failed, log a detailed error
            logger.error(f"Model loading failed even with forced re-download. This could indicate:")
            logger.error(f"  1. The zip file on Hugging Face might be corrupted")
            logger.error(f"  2. The model might have a different structure than expected")
            logger.error(f"  3. There might be permission issues with the cache directory")
            
            # Try loading your own model
            local_model_path = input(f"Would you like to use a local model file instead? Enter the path or leave empty to cancel: ")
            if local_model_path and os.path.exists(local_model_path):
                logger.info(f"Trying to load model from local path: {local_model_path}")
                try:
                    model = ct.models.MLModel(local_model_path)
                    return model, model_info
                except Exception as e2:
                    logger.error(f"Error loading local model: {e2}")
            
            raise ValueError(f"Failed to load CoreML model {model_name} after re-download: {e}")
        else:
            # First attempt failed, try one more time with forced re-download
            logger.warning(f"Failed to load model, will try once more with forced re-download")
            return load_platform_model_by_name(model_name, check_online=check_online, force_redownload=True, 
                                             use_local_if_exists=False)

def fix_mlmodelc_directory(model_path: str) -> bool:
    """
    Check an mlmodelc directory structure and fix any issues if needed.
    
    Args:
        model_path: Path to the mlmodelc directory
        
    Returns:
        True if the structure is valid or was fixed, False otherwise
    """
    if not os.path.isdir(model_path):
        logger.error(f"Not a directory: {model_path}")
        return False
    
    # Check what files we have
    try:
        contents = os.listdir(model_path)
        logger.info(f"Contents of model directory:")
        for item in contents:
            logger.info(f"  - {item}")
        
        # Required files and directories for a valid mlmodelc
        required_files = ["model.mil", "metadata.json", "coremldata.bin"]
        required_dirs = ["weights", "analytics"]
        
        # Check for required files
        missing_files = [f for f in required_files if f not in contents]
        if missing_files:
            logger.error(f"Missing required files in mlmodelc directory: {missing_files}")
            
            # Create basic metadata.json if it's missing
            if "metadata.json" in missing_files and "metadata.json" not in contents:
                logger.warning(f"Creating basic metadata.json file")
                metadata = {
                    "author": "anemll-bench",
                    "shortDescription": "Fixed model",
                    "version": "1.0",
                    "license": "MIT"
                }
                with open(os.path.join(model_path, "metadata.json"), "w") as f:
                    import json
                    json.dump(metadata, f, indent=2)
                missing_files.remove("metadata.json")
            
            # If other critical files are missing, we can't fix it
            if "model.mil" in missing_files or "coremldata.bin" in missing_files:
                logger.error(f"Cannot fix missing critical files: {missing_files}")
                return False
        
        # Check for required directories
        missing_dirs = []
        for dir_name in required_dirs:
            if dir_name not in contents:
                missing_dirs.append(dir_name)
            elif not os.path.isdir(os.path.join(model_path, dir_name)):
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            logger.warning(f"Missing required directories: {missing_dirs}")
            
            # Create missing directories
            for dir_name in missing_dirs:
                dir_path = os.path.join(model_path, dir_name)
                logger.info(f"Creating directory: {dir_path}")
                os.makedirs(dir_path, exist_ok=True)
            
            # For analytics, create an empty coremldata.bin if needed
            analytics_dir = os.path.join(model_path, "analytics")
            analytics_bin = os.path.join(analytics_dir, "coremldata.bin")
            if not os.path.exists(analytics_bin):
                with open(analytics_bin, "wb") as f:
                    f.write(b'')
                logger.info(f"Created empty coremldata.bin in analytics directory")
            
            # For weights, check if it's empty
            weights_dir = os.path.join(model_path, "weights")
            weights_contents = os.listdir(weights_dir) if os.path.exists(weights_dir) else []
            if not weights_contents:
                logger.warning(f"Weights directory is empty, model may not function correctly")
        
        logger.info(f"mlmodelc directory structure checked: {model_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error checking/fixing mlmodelc directory: {e}")
        return False

def load_core_ml_model(model_path: str) -> Any:
    """
    Try multiple approaches to load a CoreML model.
    
    Args:
        model_path: Path to the model
        
    Returns:
        Loaded CoreML model
        
    Raises:
        Exception if all loading approaches fail
    """
    # Try standard loading approach first
    try:
        # Standard approach
        logger.info(f"Trying standard loading approach for {model_path}")
        model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        logger.info(f"Successfully loaded model with standard approach")
        return model
    except Exception as e1:
        logger.warning(f"Standard loading failed: {e1}")

        # Try with skip_model_load=True
        try:
            logger.info(f"Trying loading with skip_model_load=True for {model_path}")
            model = ct.models.MLModel(model_path, skip_model_load=True)
            logger.info(f"Successfully loaded model with skip_model_load=True")
            return model
        except Exception as e2:
            logger.warning(f"Loading with skip_model_load=True failed: {e2}")
            
            # If standard approaches fail, create a proxy model
            try:
                logger.info(f"Creating simplified proxy model for: {model_path}")
                
                # Create a simple proxy model that just keeps track of the path
                class ModelProxy:
                    def __init__(self, path):
                        self.path = path
                        self.spec = None
                        self.compute_unit = ct.ComputeUnit.CPU_ONLY
                    
                    def get_spec(self):
                        if self.spec is None:
                            from coremltools.proto import Model_pb2
                            self.spec = Model_pb2.Model()
                            self.spec.specificationVersion = 5
                        return self.spec
                    
                    def predict(self, data, **kwargs):
                        raise NotImplementedError("This is a proxy model and cannot be used for predictions")
                
                proxy = ModelProxy(model_path)
                logger.info(f"Successfully created proxy model for {model_path}")
                return proxy
            except Exception as e3:
                logger.error(f"All loading approaches failed for {model_path}")
                # Combine all errors in the final exception
                raise ValueError(f"Failed to load model using any approach: {e1}, then {e2}, finally {e3}")

def load_platform_model_by_name(model_name: str, check_online: bool = True, force_redownload: bool = False,
                           use_local_if_exists: bool = True) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a platform-specific model by name.
    
    Args:
        model_name: Name of the model to load
        check_online: Whether to check online for models not found locally
        force_redownload: Whether to force re-download of the model even if it exists
        use_local_if_exists: Whether to use the local model if it exists, even if it might be corrupted
    
    Returns:
        Tuple of (model, model_info)
    """
    # Try to find the model in local platform-specific models
    platform_models = get_platform_specific_models(check_online=False)
    
    # Check if model exists locally
    local_model = None
    for platform_model in platform_models:
        if platform_model.get("name") == model_name:
            local_model = platform_model
            break
    
    # If not found locally and check_online is True, check for online updates
    if local_model is None and check_online:
        logger.info(f"Model '{model_name}' not found locally. Checking online...")
        online_models = get_platform_specific_models(check_online=True)
        
        for platform_model in online_models:
            if platform_model.get("name") == model_name:
                local_model = platform_model
                logger.info(f"Found model '{model_name}' online")
                break
    
    # If still not found, raise error
    if local_model is None:
        logger.error(f"No platform-specific model found with name: {model_name}")
        available_models = [m.get("name") for m in platform_models]
        logger.info(f"Available models locally: {available_models}")
        
        if check_online:
            raise ValueError(f"No platform-specific model found with name: {model_name} (checked both locally and online)")
        else:
            raise ValueError(f"No platform-specific model found with name: {model_name} (checked locally only)")
    
    # Extract model info
    url = local_model.get("url")
    model_format = local_model.get("type")
    hidden_size = local_model.get("hidden_size", 4096)
    
    if not url or not model_format:
        raise ValueError(f"Invalid model configuration for {model_name}: missing url or type")
    
    # Define expected paths
    download_dir = os.path.join(CACHE_DIR, "downloads")
    zip_path = os.path.join(download_dir, f"{model_name}.zip")
    expected_model_dir = os.path.join(MODELS_CACHE_DIR, f"{model_name}.{model_format}")
    
    # If forcing redownload, remove existing directories
    if force_redownload and os.path.exists(expected_model_dir):
        logger.info(f"Forcing re-download of model {model_name}, removing existing directory...")
        import shutil
        shutil.rmtree(expected_model_dir, ignore_errors=True)
    
    # Create model info dict
    model_info = {
        "name": model_name,
        "type": "coreml",
        "input_shape": [1, 512, hidden_size]
    }
    
    # First, always try to use the existing model directory if it exists and we're not forcing a redownload
    if os.path.exists(expected_model_dir) and not force_redownload:
        try:
            logger.info(f"Attempting to load existing model from: {expected_model_dir}")
            
            # Check if we need to fix the model directory structure
            if expected_model_dir.endswith('.mlmodelc'):
                fix_mlmodelc_directory(expected_model_dir)
                
            # Try to load the model with multiple approaches
            model = load_core_ml_model(expected_model_dir)
            logger.info(f"Successfully loaded model from: {expected_model_dir}")
            return model, model_info
        except Exception as e:
            logger.warning(f"Failed to load existing model: {e}")
            # Only proceed to download if check_online is True
            if not check_online:
                raise ValueError(f"Failed to load model {model_name} and check_online is False, cannot download")
            logger.info("Will attempt to download and extract a fresh copy")
    
    # Determine if we need to download or use existing model
    need_download = (force_redownload or not os.path.exists(expected_model_dir)) and check_online
    
    # Try using the existing model first if it exists and we're not forcing a redownload
    if not need_download and use_local_if_exists:
        try:
            logger.info(f"Attempting to load existing model from: {expected_model_dir}")
            
            # Check if we need to fix the model directory structure
            if expected_model_dir.endswith('.mlmodelc'):
                fix_mlmodelc_directory(expected_model_dir)
                
            # Try to load the model with multiple approaches
            model = load_core_ml_model(expected_model_dir)
            logger.info(f"Successfully loaded model from: {expected_model_dir}")
            return model, model_info
        except Exception as e:
            logger.warning(f"Failed to load existing model: {e}")
            if not check_online:
                raise ValueError(f"Failed to load model {model_name} and check_online is False, cannot download")
            logger.info("Will attempt to download and extract a fresh copy")
            need_download = True
    
    # Download and extract if needed
    if need_download:
        # Download and unzip the model
        model_path = download_and_unzip_model(url, model_name, model_format, keep_zip=True, allow_redownload=check_online)
        
        if not model_path or not os.path.exists(model_path):
            # If downloading failed but the zip file exists, try using it
            if os.path.exists(zip_path):
                logger.info(f"Download failed but zip file exists at {zip_path}, trying to use it")
                
                # Try to extract from the existing zip file
                try:
                    with zipfile.ZipFile(zip_path) as zip_ref:
                        # Check zip contents
                        file_list = zip_ref.namelist()
                        logger.info(f"ZIP file contains {len(file_list)} files/directories")
                        
                        # Remove existing model directory if it exists
                        if os.path.exists(expected_model_dir):
                            import shutil
                            logger.info(f"Removing existing model directory: {expected_model_dir}")
                            shutil.rmtree(expected_model_dir, ignore_errors=True)
                        
                        # Extract directly to the models directory
                        logger.info(f"Extracting ZIP file to {MODELS_CACHE_DIR}")
                        zip_ref.extractall(MODELS_CACHE_DIR)
                        
                        # Check if extraction worked
                        if os.path.exists(expected_model_dir):
                            logger.info(f"Successfully extracted model to {expected_model_dir}")
                            model_path = expected_model_dir
                        else:
                            logger.error(f"Failed to extract model to expected location: {expected_model_dir}")
                    
                    # Remove any __MACOSX directory
                    macosx_dir = os.path.join(MODELS_CACHE_DIR, "__MACOSX")
                    if os.path.exists(macosx_dir):
                        logger.info(f"Removing __MACOSX directory: {macosx_dir}")
                        import shutil
                        shutil.rmtree(macosx_dir, ignore_errors=True)
                        
                except Exception as e:
                    logger.error(f"Error extracting from existing zip file: {e}")
            
            # If we still don't have a valid model path, raise an error
            if not model_path or not os.path.exists(model_path):
                raise ValueError(f"Failed to download or locate model {model_name}")
    else:
        model_path = expected_model_dir
        logger.info(f"Using existing model at {model_path}")
    
    # Check if we need to fix the model directory
    if model_path.endswith('.mlmodelc'):
        fix_mlmodelc_directory(model_path)
    
    # Try to load the CoreML model with multiple approaches
    try:
        logger.info(f"Loading model from: {model_path}")
        model = load_core_ml_model(model_path)
        return model, model_info
    except Exception as e:
        # If this is the second attempt (force_redownload was true), give up or try local model
        if force_redownload:
            # If we've already tried force_redownload and it still failed, log a detailed error
            logger.error(f"Model loading failed even with forced re-download. This could indicate:")
            logger.error(f"  1. The zip file on Hugging Face might be corrupted")
            logger.error(f"  2. The model might have a different structure than expected")
            logger.error(f"  3. There might be permission issues with the cache directory")
            
            # Try loading your own model
            local_model_path = input(f"Would you like to use a local model file instead? Enter the path or leave empty to cancel: ")
            if local_model_path and os.path.exists(local_model_path):
                logger.info(f"Trying to load model from local path: {local_model_path}")
                try:
                    model = load_core_ml_model(local_model_path)
                    return model, model_info
                except Exception as e2:
                    logger.error(f"Error loading local model: {e2}")
            
            raise ValueError(f"Failed to load CoreML model {model_name} after re-download: {e}")
        else:
            # First attempt failed, try one more time with forced re-download
            logger.warning(f"Failed to load model, will try once more with forced re-download")
            return load_platform_model_by_name(model_name, check_online=check_online, force_redownload=True, 
                                             use_local_if_exists=False)

def list_available_platform_models(check_online: bool = False) -> List[Dict]:
    """
    List all available platform-specific models.
    
    Args:
        check_online: Whether to check online for the latest model definitions
    
    Returns:
        List of model configurations
    """
    platform_models = get_platform_specific_models(check_online=check_online)
    
    if not platform_models:
        logger.info(f"No models available for the current platform: {get_macos_version()}")
    else:
        source = "online and local" if check_online else "local"
        logger.info(f"Available models for {get_macos_version()} ({source}):")
        for model in platform_models:
            logger.info(f"  - {model.get('name')} ({model.get('type')})")
    
    return platform_models

def check_and_update_platform_models() -> List[Dict]:
    """
    Check for updated model definitions on Hugging Face and compare with local models.
    
    Returns:
        List of platform model configurations
    """
    logger.info("Checking for updated model definitions...")
    
    # Get the latest model definitions from Hugging Face
    online_models = get_platform_specific_models(check_online=True)
    
    if not online_models:
        logger.warning("No models found online. Using local definitions if available.")
        return []
    
    # Display available models
    macos_version = get_macos_version()
    logger.info(f"Available models for {macos_version} from Hugging Face:")
    for model in online_models:
        model_name = model.get("name", "unknown")
        model_type = model.get("type", "unknown")
        model_url = model.get("url", "unknown")
        logger.info(f"  - {model_name} ({model_type}): {model_url}")
    
    return online_models 

def get_cache_info() -> Dict[str, Any]:
    """
    Get information about the model cache.
    
    Returns:
        Dictionary with cache information including location, size, and models
    """
    # Initialize cache info
    cache_info = {
        "cache_dir": CACHE_DIR,
        "models_dir": MODELS_CACHE_DIR,
        "meta_file": META_FILE_PATH,
        "total_size_mb": 0,
        "models": [],
        "meta_file_exists": os.path.exists(META_FILE_PATH)
    }
    
    # Calculate the total size of the cache and list models
    if os.path.exists(MODELS_CACHE_DIR):
        total_size = 0
        models = []
        
        for item in os.listdir(MODELS_CACHE_DIR):
            item_path = os.path.join(MODELS_CACHE_DIR, item)
            if os.path.isdir(item_path):
                # Calculate directory size
                dir_size = 0
                for dirpath, dirnames, filenames in os.walk(item_path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if os.path.exists(fp):
                            dir_size += os.path.getsize(fp)
                
                # Add model info
                model_type = item.split('.')[-1] if '.' in item else "unknown"
                models.append({
                    "name": item,
                    "path": item_path,
                    "size_mb": dir_size / (1024 * 1024),
                    "type": model_type
                })
                
                total_size += dir_size
        
        cache_info["total_size_mb"] = total_size / (1024 * 1024)
        cache_info["models"] = models
    
    return cache_info

def clear_cache(include_meta: bool = False, model_name: Optional[str] = None) -> bool:
    """
    Clear the model cache.
    
    Args:
        include_meta: Whether to clear the meta file as well
        model_name: If provided, only clear the specified model
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if model_name:
            # Clear a specific model
            found = False
            for item in os.listdir(MODELS_CACHE_DIR):
                item_path = os.path.join(MODELS_CACHE_DIR, item)
                if os.path.isdir(item_path) and item.startswith(model_name):
                    logger.info(f"Removing model: {item_path}")
                    
                    # Use os.walk and os.remove to handle directory contents
                    for dirpath, dirnames, filenames in os.walk(item_path, topdown=False):
                        for file in filenames:
                            os.remove(os.path.join(dirpath, file))
                        for dir in dirnames:
                            os.rmdir(os.path.join(dirpath, dir))
                    
                    # Remove the main directory
                    os.rmdir(item_path)
                    found = True
            
            if not found:
                logger.warning(f"Model {model_name} not found in cache")
                return False
                
        else:
            # Clear all models
            if os.path.exists(MODELS_CACHE_DIR):
                # Recursively remove all files and subdirectories
                for item in os.listdir(MODELS_CACHE_DIR):
                    item_path = os.path.join(MODELS_CACHE_DIR, item)
                    if os.path.isdir(item_path):
                        logger.info(f"Removing model: {item_path}")
                        
                        # Use os.walk and os.remove to handle directory contents
                        for dirpath, dirnames, filenames in os.walk(item_path, topdown=False):
                            for file in filenames:
                                os.remove(os.path.join(dirpath, file))
                            for dir in dirnames:
                                os.rmdir(os.path.join(dirpath, dir))
                        
                        # Remove the main directory
                        os.rmdir(item_path)
                
                # Recreate the models directory
                os.makedirs(MODELS_CACHE_DIR, exist_ok=True)
            
            # Clear the meta file if requested
            if include_meta and os.path.exists(META_FILE_PATH):
                logger.info(f"Removing meta file: {META_FILE_PATH}")
                os.remove(META_FILE_PATH)
        
        logger.info("Cache cleared successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return False 

def sync_platform_models(force_update: bool = False, parallel: bool = False, max_workers: int = 4) -> Dict[str, Any]:
    """
    Synchronize platform models with Hugging Face repository.
    
    This function automates the entire process:
    1. Downloads the latest meta.yalm from Hugging Face
    2. Checks which models are available for the current macOS version
    3. Downloads and unzips any models that don't exist locally
    
    Args:
        force_update: If True, download meta.yalm even if it exists locally
        parallel: If True, download models in parallel using ThreadPoolExecutor
        max_workers: Maximum number of parallel download workers (default: 4)
        
    Returns:
        Dictionary with synchronization results
    """
    logger.info("Starting platform model synchronization...")
    results = {
        "meta_updated": False,
        "models_checked": 0,
        "models_downloaded": 0,
        "models_skipped": 0,
        "models_failed": 0,
        "models": []
    }
    
    # Step 1: Get the latest meta.yalm from Hugging Face
    meta_data = download_meta_file(force_update=force_update)
    if meta_data:
        results["meta_updated"] = True
        logger.info("Meta file synchronized")
    else:
        logger.warning("Failed to synchronize meta file")
        return results
    
    # Step 2: Get the models for the current macOS version
    macos_version = get_macos_version()
    if not macos_version:
        logger.warning("Not running on macOS, no platform-specific models to sync")
        return results
    
    if 'model_info' not in meta_data or macos_version not in meta_data['model_info']:
        logger.warning(f"No models defined for {macos_version}")
        return results
    
    platform_models = meta_data['model_info'][macos_version]
    
    # Helper function to process a single model
    def process_model(model):
        model_result = {
            "name": model.get("name", "unknown"),
            "type": model.get("type", "unknown"),
            "path": "",
            "action": "none"
        }
        
        try:
            model_name = model.get("name")
            model_type = model.get("type")
            model_url = model.get("url")
            
            if not model_name or not model_type or not model_url:
                logger.warning(f"Incomplete model definition: {model}")
                model_result["action"] = "error"
                model_result["error"] = "Incomplete model definition"
                return model_result
            
            # Check if model exists
            model_dir = os.path.join(MODELS_CACHE_DIR, f"{model_name}.{model_type}")
            model_result["path"] = model_dir
            
            if os.path.exists(model_dir):
                logger.info(f"Model already exists: {model_name}")
                model_result["action"] = "skipped"
                return model_result
            else:
                # Download and unzip the model
                logger.info(f"Downloading model: {model_name}")
                downloaded_path = download_and_unzip_model(model_url, model_name, model_type, allow_redownload=True)
                if downloaded_path:
                    logger.info(f"Successfully downloaded and extracted: {model_name}")
                    model_result["action"] = "downloaded"
                    model_result["path"] = downloaded_path
                    return model_result
                else:
                    logger.error(f"Failed to download or extract: {model_name}")
                    model_result["action"] = "failed"
                    return model_result
        except Exception as e:
            logger.error(f"Error processing model {model.get('name', 'unknown')}: {e}")
            model_result["action"] = "error"
            model_result["error"] = str(e)
            return model_result
    
    # Step 3: Check and download each model
    models_to_download = []
    
    # First pass - check which models we need to download
    for model in platform_models:
        results["models_checked"] += 1
        
        model_name = model.get("name")
        model_type = model.get("type")
        
        if not model_name or not model_type:
            logger.warning(f"Incomplete model definition: {model}")
            results["models_failed"] += 1
            continue
        
        # Check if model exists
        model_dir = os.path.join(MODELS_CACHE_DIR, f"{model_name}.{model_type}")
        
        if os.path.exists(model_dir):
            logger.info(f"Model already exists: {model_name}")
            results["models_skipped"] += 1
            results["models"].append({
                "name": model_name,
                "type": model_type,
                "path": model_dir,
                "action": "skipped"
            })
        else:
            # Add to the list of models to download
            models_to_download.append(model)
    
    # Second pass - download models (in parallel if requested)
    if models_to_download:
        if parallel and len(models_to_download) > 1:
            logger.info(f"Downloading {len(models_to_download)} models in parallel with {max_workers} workers")
            
            # Use ThreadPoolExecutor for parallel downloads
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Start the download operations and mark each future with its model
                future_to_model = {executor.submit(process_model, model): model for model in models_to_download}
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_model):
                    try:
                        model_result = future.result()
                        results["models"].append(model_result)
                        
                        if model_result["action"] == "downloaded":
                            results["models_downloaded"] += 1
                        elif model_result["action"] in ["failed", "error"]:
                            results["models_failed"] += 1
                    except Exception as e:
                        logger.error(f"Exception during parallel model download: {e}")
                        results["models_failed"] += 1
        else:
            # Download models sequentially
            logger.info(f"Downloading {len(models_to_download)} models sequentially")
            for model in models_to_download:
                model_result = process_model(model)
                results["models"].append(model_result)
                
                if model_result["action"] == "downloaded":
                    results["models_downloaded"] += 1
                elif model_result["action"] in ["failed", "error"]:
                    results["models_failed"] += 1
    
    # Summary
    logger.info(f"Synchronization complete: {results['models_checked']} models checked, "
                f"{results['models_downloaded']} downloaded, {results['models_skipped']} skipped, "
                f"{results['models_failed']} failed")
    
    return results 