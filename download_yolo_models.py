"""
Script to download all YOLO model sizes in advance with improved reliability
"""
import os
import sys
import logging
import time
import urllib.request
import urllib.error
import ssl

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("download_models")

# Model URLs - hardcoded to ensure reliable downloads
MODEL_URLS = {
    "tiny": {
        "config": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
        "weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
    },
    "medium": {
        "config": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg",
        "weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4.weights"
    },
    "large": {
        "config": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-p6.cfg",
        "weights": "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-p6.weights"
    }
}

# Expected file sizes in bytes for verification
EXPECTED_SIZES = {
    "tiny": {
        "config": 12222,
        "weights": 23651024
    },
    "medium": {
        "config": 13509,
        "weights": 257717640
    },
    "large": {
        "config": 14970,
        "weights": 470643400
    }
}

class DownloadProgressBar:
    def __init__(self, total_size, desc="Downloading"):
        self.total_size = total_size
        self.downloaded = 0
        self.last_percent = -1
        self.desc = desc
        self.start_time = time.time()
        
    def __call__(self, count, block_size, total_size):
        self.downloaded += block_size
        percent = int(self.downloaded * 100 / self.total_size)
        
        # Only update when percent changes to avoid too many prints
        if percent != self.last_percent and percent % 5 == 0:
            self.last_percent = percent
            elapsed_time = time.time() - self.start_time
            speed = self.downloaded / (elapsed_time if elapsed_time > 0 else 1) / 1024 / 1024  # MB/s
            
            # Calculate ETA
            if speed > 0:
                eta = (self.total_size - self.downloaded) / (speed * 1024 * 1024)
                eta_str = f", ETA: {int(eta//60)}m {int(eta%60)}s"
            else:
                eta_str = ""
                
            logger.info(f"{self.desc}: {percent}% ({self.downloaded / 1024 / 1024:.1f}MB/{self.total_size / 1024 / 1024:.1f}MB) at {speed:.2f} MB/s{eta_str}")

def download_file(url, dest_path, desc, expected_size, retries=3):
    """Download a file with progress bar and retries"""
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    # Check if file already exists and has correct size
    if os.path.exists(dest_path) and os.path.getsize(dest_path) == expected_size:
        logger.info(f"{desc} already exists with correct size. Skipping download.")
        return True
    
    # If file exists but has wrong size, delete it
    if os.path.exists(dest_path):
        logger.warning(f"{desc} exists but has wrong size. Deleting and re-downloading.")
        os.remove(dest_path)
    
    # Try downloading with retries
    for attempt in range(retries):
        try:
            # Create SSL context to handle certificate issues
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            # Use progress bar
            progress_bar = DownloadProgressBar(expected_size, desc=desc)
            
            # Download the file
            urllib.request.urlretrieve(
                url, 
                dest_path,
                reporthook=progress_bar
            )
            
            # Verify file size
            actual_size = os.path.getsize(dest_path)
            if actual_size != expected_size:
                logger.error(f"Download incomplete. Expected size: {expected_size}, got: {actual_size}")
                os.remove(dest_path)
                continue
            
            logger.info(f"Successfully downloaded {desc}")
            return True
            
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionResetError) as e:
            logger.error(f"Download failed on attempt {attempt+1}/{retries}: {e}")
            if os.path.exists(dest_path):
                os.remove(dest_path)
            
            # Wait before retrying
            if attempt < retries - 1:
                wait_time = 5 * (attempt + 1)
                logger.info(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
    
    logger.error(f"Failed to download {desc} after {retries} attempts")
    return False

def download_model(model_size):
    """Download a specific YOLO model size"""
    
    if model_size not in MODEL_URLS:
        logger.error(f"Unknown model size: {model_size}")
        return False
    
    # Define paths
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detection", "models")
    config_path = os.path.join(models_dir, f"yolov4{'-tiny' if model_size == 'tiny' else '-p6' if model_size == 'large' else ''}.cfg")
    weights_path = os.path.join(models_dir, f"yolov4{'-tiny' if model_size == 'tiny' else '-p6' if model_size == 'large' else ''}.weights")
    
    # Download config file
    config_success = download_file(
        MODEL_URLS[model_size]["config"],
        config_path,
        f"{model_size} config file",
        EXPECTED_SIZES[model_size]["config"]
    )
    
    # Download weights file
    weights_success = download_file(
        MODEL_URLS[model_size]["weights"],
        weights_path,
        f"{model_size} weights file",
        EXPECTED_SIZES[model_size]["weights"]
    )
    
    return config_success and weights_success

def download_all_models():
    """Download all YOLO model sizes"""
    model_sizes = ["tiny", "medium", "large"]
    
    # First download the tiny model as it's fastest
    results = {}
    
    for size in model_sizes:
        logger.info(f"=== Downloading {size.upper()} model ===")
        success = download_model(size)
        results[size] = success
        logger.info(f"{'✅' if success else '❌'} {size.upper()} model: {'SUCCESSFUL' if success else 'FAILED'}")
    
    # Print summary
    logger.info("=== Download Summary ===")
    for size, success in results.items():
        logger.info(f"{'✅' if success else '❌'} {size.upper()} model: {'SUCCESSFUL' if success else 'FAILED'}")

if __name__ == "__main__":
    logger.info("Starting YOLO model downloads...")
    download_all_models()
    logger.info("Download process complete")
