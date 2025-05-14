"""
SmartVan Monitor - Main Application
----------------------------------
Main application entry point for the SmartVan Monitor system.
This version implements basic motion detection for Phase 1 MVP.
"""

import os
import cv2
import time
import argparse
import logging
import json
import datetime
from pathlib import Path

# Import local modules
from utils.camera import CameraManager
from detection.motion_detection import MotionDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("smartvan_monitor.log")
    ]
)
logger = logging.getLogger("main")

class SmartVanMonitor:
    """Main application class for SmartVan Monitor system."""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the SmartVan Monitor application.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Create output directories
        self.output_dir = Path(self.config.get("output_dir", "output"))
        self.output_dir.mkdir(exist_ok=True)
        self.clips_dir = self.output_dir / "clips"
        self.clips_dir.mkdir(exist_ok=True)
        
        # Initialize camera manager
        self.camera_manager = CameraManager()
        self._setup_cameras()
        
        # Initialize motion detectors (one per camera)
        self.motion_detectors = {}
        self._setup_motion_detectors()
        
        # Initialize detection state
        self.last_save_time = {}
        self.recording_active = {}
        
        logger.info("SmartVan Monitor initialized")
    
    def _load_config(self) -> dict:
        """
        Load configuration from file or create default.
        
        Returns:
            Configuration dictionary
        """
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        
        # Default configuration
        config = {
            "cameras": [
                {
                    "id": 0,
                    "name": "Rear_Inventory",
                    "resolution": [640, 480],
                    "fps": 10,
                    "roi": [
                        [10, 10, 620, 460]  # Example ROI, will need adjustment
                    ]
                },
                {
                    "id": 1,
                    "name": "Side_Wall",
                    "resolution": [640, 480],
                    "fps": 10,
                    "roi": [
                        [10, 10, 620, 460]  # Example ROI, will need adjustment
                    ]
                }
            ],
            "motion_detection": {
                "algorithm": "MOG2",
                "learning_rate": 0.001,
                "motion_threshold": 25,
                "min_area": 200,
                "blur_size": 21
            },
            "recording": {
                "min_intensity": 20,
                "pre_motion_seconds": 2,
                "post_motion_seconds": 3,
                "min_recording_gap": 10
            },
            "output_dir": "output"
        }
        
        # Save default configuration
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)
            logger.info(f"Created default configuration at {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving default configuration: {e}")
        
        return config
    
    def _setup_cameras(self) -> None:
        """Set up cameras from configuration."""
        for camera_config in self.config.get("cameras", []):
            camera_id = camera_config.get("id", 0)
            name = camera_config.get("name", f"Camera_{camera_id}")
            resolution = tuple(camera_config.get("resolution", [640, 480]))
            fps = camera_config.get("fps", 10)
            
            self.camera_manager.add_camera(
                camera_id=camera_id,
                name=name,
                resolution=resolution,
                fps=fps
            )
            
            # Initialize state tracking for this camera
            self.last_save_time[name] = 0
            self.recording_active[name] = False
    
    def _setup_motion_detectors(self) -> None:
        """Set up motion detectors for each camera."""
        motion_config = self.config.get("motion_detection", {})
        
        for camera_config in self.config.get("cameras", []):
            name = camera_config.get("name", f"Camera_{camera_config.get('id', 0)}")
            
            detector = MotionDetector(
                algorithm=motion_config.get("algorithm", "MOG2"),
                learning_rate=motion_config.get("learning_rate", 0.001),
                motion_threshold=motion_config.get("motion_threshold", 25),
                min_area=motion_config.get("min_area", 200),
                blur_size=motion_config.get("blur_size", 21)
            )
            
            # Add ROIs from configuration
            for roi in camera_config.get("roi", []):
                if len(roi) == 4:
                    detector.add_roi(*roi)
            
            self.motion_detectors[name] = detector
            logger.info(f"Created motion detector for {name}")
    
    def start(self) -> None:
        """Start the SmartVan Monitor system."""
        logger.info("Starting SmartVan Monitor")
        
        # Start all cameras
        self.camera_manager.start_all_cameras()
        
        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down")
        finally:
            self.stop()
    
    def stop(self) -> None:
        """Stop the SmartVan Monitor system."""
        logger.info("Stopping SmartVan Monitor")
        self.camera_manager.stop_all_cameras()
        self.camera_manager.cleanup()
    
    def _main_loop(self) -> None:
        """Main processing loop."""
        logger.info("Entering main loop")
        
        # FPS tracking
        frame_count = 0
        start_time = time.time()
        fps_display_interval = 5  # seconds
        
        while True:
            # Get frames from all cameras
            frames = self.camera_manager.get_all_frames()
            
            # Process each camera
            for camera_name, (frame, timestamp, fps) in frames.items():
                if frame is None:
                    continue
                
                # Motion detection
                detector = self.motion_detectors.get(camera_name)
                if detector is None:
                    continue
                
                # Detect motion
                motion_result = detector.detect(frame)
                
                # Process detection result
                self._process_detection(camera_name, frame, motion_result, timestamp)
                
                # Display frame (for development/debugging)
                if self.config.get("display", True):
                    display_frame = detector.visualize(frame, motion_result)
                    cv2.imshow(f"SmartVan Monitor - {camera_name}", display_frame)
            
            # FPS calculation and display
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= fps_display_interval:
                fps = frame_count / elapsed_time
                logger.info(f"Processing FPS: {fps:.2f}")
                frame_count = 0
                start_time = time.time()
            
            # Check for exit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    def _process_detection(self, camera_name: str, frame: any, 
                          motion_result: dict, timestamp: float) -> None:
        """
        Process motion detection results and record if necessary.
        
        Args:
            camera_name: Name of the camera
            frame: Current video frame
            motion_result: Detection result from motion detector
            timestamp: Frame timestamp
        """
        # Get recording configuration
        recording_config = self.config.get("recording", {})
        min_intensity = recording_config.get("min_intensity", 20)
        min_recording_gap = recording_config.get("min_recording_gap", 10)
        
        # Check if motion is significant enough to trigger recording
        motion_detected = motion_result["motion_detected"]
        intensity = motion_result["intensity"]
        significant_motion = motion_detected and intensity >= min_intensity
        
        # Get current time
        current_time = time.time()
        time_since_last_save = current_time - self.last_save_time.get(camera_name, 0)
        
        # Determine if we should start/continue recording
        if significant_motion:
            # If not already recording or minimum gap satisfied
            if not self.recording_active.get(camera_name, False) or time_since_last_save >= min_recording_gap:
                # Start a new recording
                self._save_event(camera_name, frame, motion_result, timestamp)
                self.recording_active[camera_name] = True
                self.last_save_time[camera_name] = current_time
                logger.info(f"Motion detected on {camera_name}: Intensity {intensity}")
        else:
            # No significant motion, mark as not recording
            self.recording_active[camera_name] = False
    
    def _save_event(self, camera_name: str, frame: any, 
                   motion_result: dict, timestamp: float) -> None:
        """
        Save a motion event (image and metadata).
        
        Args:
            camera_name: Name of the camera
            frame: Current video frame
            motion_result: Detection result from motion detector
            timestamp: Frame timestamp
        """
        # Create timestamp string
        dt = datetime.datetime.fromtimestamp(timestamp)
        timestamp_str = dt.strftime("%Y%m%d_%H%M%S")
        
        # Save image
        image_filename = f"{camera_name}_{timestamp_str}_intensity{motion_result['intensity']}.jpg"
        image_path = self.clips_dir / image_filename
        cv2.imwrite(str(image_path), frame)
        
        # Save metadata
        metadata = {
            "camera": camera_name,
            "timestamp": timestamp,
            "datetime": dt.isoformat(),
            "motion_detected": motion_result["motion_detected"],
            "intensity": motion_result["intensity"],
            "duration": motion_result["duration"],
            "bounding_boxes": motion_result["bounding_boxes"],
            "image_path": str(image_path)
        }
        
        metadata_filename = f"{camera_name}_{timestamp_str}_intensity{motion_result['intensity']}.json"
        metadata_path = self.clips_dir / metadata_filename
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Saved event to {image_path}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SmartVan Monitor System")
    parser.add_argument("--config", default="config.json", help="Path to configuration file")
    parser.add_argument("--no-display", action="store_true", help="Disable video display")
    args = parser.parse_args()
    
    # Load configuration
    config_path = args.config
    
    # Create and start the monitor
    monitor = SmartVanMonitor(config_path=config_path)
    
    # Override display setting if specified
    if args.no_display:
        monitor.config["display"] = False
    
    # Start the monitor
    monitor.start()
