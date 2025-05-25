"""
SmartVan Monitor - Main Application
----------------------------------
Main application entry point for the SmartVan Monitor system.
This version implements enhanced motion detection, event classification, and object detection for Phase 1 MVP.
"""

import os
import sys
import time
import json
import logging
import cv2
import numpy as np
import argparse
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Custom JSON encoder to handle NumPy types
class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyJSONEncoder, self).default(obj)

# Import local modules
from utils.camera import CameraManager
from utils.backend_client import BackendClient
from detection.motion_detection import MotionDetector
from detection.event_classification import EventClassifier
from detection.object_detection.yolo_detector import YOLODetector
from detection.integrated_detector import IntegratedDetector
from detection.enhanced_monitor import EnhancedDetectionMonitor
from detection.inventory_detection import InventoryChangeDetector
from detection.inventory_tracker import InventoryTracker
from detection.event_framework import EnhancedEventClassifier

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
        
        # Initialize detection state
        self.last_save_time = {}
        self.recording_active = {}
        
        # Initialize camera manager
        self.camera_manager = CameraManager()
        self._setup_cameras()
        
        # Initialize backend client for edge computing approach
        self.backend_client = BackendClient(self.config, self.output_dir)
        
        # Initialize enhanced detection monitor (replaces individual detectors)
        try:
            logger.info("Initializing EnhancedDetectionMonitor with optimized parameters...")
            # Add vibration detection settings to config if not already present
            if "motion_detection" not in self.config:
                self.config["motion_detection"] = {}
            if "event_classification" not in self.config:
                self.config["event_classification"] = {}
                
            # Make sure the fine-tuned thresholds are set
            self.config["event_classification"]["motion_intensity_threshold"] = 40  # Ignore vibrations below 40
            self.config["motion_detection"]["vibration_threshold"] = 40  # Configurable from command line
                
            # Initialize enhanced monitor with updated config
            self.enhanced_monitor = EnhancedDetectionMonitor(self.config)
            logger.info("Successfully initialized EnhancedDetectionMonitor")
        except Exception as e:
            logger.error(f"Error initializing EnhancedDetectionMonitor: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        
        # Keep references to individual components for backward compatibility
        self.motion_detectors = {}
        self.integrated_detectors = {}
        self.event_classifier = None
        
        # Initialize cameras in the enhanced monitor
        # Handle cameras list format from the existing config
        for camera_config in self.config.get("cameras", []):
            # Get camera name or use ID if name not available
            camera_name = camera_config.get("name", f"Camera_{camera_config.get('id', 0)}")
            
            # Setup camera with ROI if available
            if "roi" in camera_config:
                roi_areas = camera_config["roi"]
                self.enhanced_monitor.setup_camera(camera_name, roi_areas)
            else:
                self.enhanced_monitor.setup_camera(camera_name)
        
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
            # Add van_id for fleet identification
            "van_id": "VAN001",  # Default identifier, should be unique per vehicle
            
            "cameras": [
                {
                    "id": 0,
                    "name": "Rear_Inventory",
                    "resolution": [640, 480],
                    "fps": 2,  # 500ms between frames
                    "roi": [
                        [10, 10, 620, 460]  # Example ROI, will need adjustment
                    ]
                },
                {
                    "id": 1,
                    "name": "Side_Wall",
                    "resolution": [640, 480],
                    "fps": 2,  # 500ms between frames
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
                "blur_size": 21,
                "noise_filter": 0.6,
                "movement_stability": 7,
                "human_detection": {
                    "enabled": true,
                    "flow_magnitude_threshold": 3.5,
                    "flow_uniformity_threshold": 0.5,
                    "significant_motion_threshold": 6.0,
                    "min_duration": 1.5
                }
            },
            "event_classification": {
                "business_hours_start": 7,
                "business_hours_end": 19,
                "high_frequency_threshold": 5,
                "high_frequency_window": 900,
                "event_memory_window": 3600,
                "min_alert_interval": 120,
                "motion_intensity_threshold": 40
            },
            "object_detection": {
                "enabled": True,
                "model_size": "tiny",  # tiny, medium, or large
                "confidence_threshold": 0.5,
                "nms_threshold": 0.4,
                "only_on_motion": True,  # Only run YOLO when motion is detected
                "gpu_enabled": False,  # Set to true if GPU is available
                "yolo_interval": 5,  # Run YOLO every X frames when no motion
                "human_confirmation_needed": True,  # Require both motion and YOLO to confirm human
                "object_recording_threshold": 0.6,  # Confidence threshold for recording
                "classes_of_interest": ["person", "backpack", "handbag", "suitcase", "bottle"]
            },
            "recording": {
                "min_intensity": 20,
                "pre_motion_seconds": 2,
                "post_motion_seconds": 3,
                "min_recording_gap": 10
            },
            "output_dir": "output",
            # Backend communication settings
            "backend": {
                "server_url": "https://api.smartvan-monitor.com/v1",
                "api_key": "",  # Must be provided for backend communication
                "heartbeat_interval": 300,  # 5 minutes
                "summary_interval": 3600,  # 1 hour
                "max_retries": 3,
                "retry_delay": 30
            }
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
        """Set up motion detectors for each camera with enhanced features."""
        motion_config = self.config.get("motion_detection", {})
        
        for camera_name in self.camera_manager.cameras:
            # Find camera config that matches this camera name
            camera_config = None
            roi_areas = None
            for cam_cfg in self.config.get("cameras", []):
                if cam_cfg.get("name") == camera_name:
                    camera_config = cam_cfg
                    roi_areas = camera_config.get("roi", [])
                    break
            
            if not camera_config:
                logger.warning(f"No configuration found for camera {camera_name}, using defaults")
            
            # Create motion detector with configurable human detection
            self.motion_detectors[camera_name] = MotionDetector(
                algorithm=motion_config.get("algorithm", "MOG2"),
                learning_rate=motion_config.get("learning_rate", 0.001),
                motion_threshold=motion_config.get("motion_threshold", 25),
                min_area=motion_config.get("min_area", 300),
                blur_size=motion_config.get("blur_size", 21),
                roi_areas=roi_areas,  # Pass the ROI areas directly
                noise_filter=motion_config.get("noise_filter", 0.6),
                movement_stability=motion_config.get("movement_stability", 7),
                human_detection_config=motion_config.get("human_detection", {})
            )
            
            logger.info(f"Set up motion detector for {camera_name} with human detection settings: {motion_config.get('human_detection', {})}")
        
        # Initialize tracking state for all cameras
        for camera_name in self.camera_manager.cameras:
            self.last_save_time[camera_name] = 0
            self.recording_active[camera_name] = False
            
        logger.info(f"Set up {len(self.motion_detectors)} motion detectors")
    
    def _setup_integrated_detectors(self) -> None:
        """Set up integrated detectors (motion + YOLO) for each camera."""
        # Get object detection configuration
        object_detection_config = self.config.get("object_detection", {})
        
        # Skip if object detection is not enabled
        if not object_detection_config.get("enabled", True):
            logger.info("Object detection disabled in configuration")
            return
        
        # Prepare YOLO configuration
        yolo_config = {
            "model_size": object_detection_config.get("model_size", "tiny"),
            "confidence_threshold": object_detection_config.get("confidence_threshold", 0.5),
            "nms_threshold": object_detection_config.get("nms_threshold", 0.4),
            "only_on_motion": object_detection_config.get("only_on_motion", True),
            "gpu_enabled": object_detection_config.get("gpu_enabled", False)
        }
        
        # Configure each camera's integrated detector
        for camera_name, motion_detector in self.motion_detectors.items():
            # Initialize integrated detector
            self.integrated_detectors[camera_name] = IntegratedDetector(
                motion_detector=motion_detector,
                yolo_config=yolo_config,
                always_run_yolo=not object_detection_config.get("only_on_motion", True),
                yolo_interval=object_detection_config.get("yolo_interval", 5),
                human_confirmation_needed=object_detection_config.get("human_confirmation_needed", True),
                object_recording_threshold=object_detection_config.get("object_recording_threshold", 0.6)
            )
            
            logger.info(f"Initialized integrated detector (motion + YOLO) for {camera_name}")
        
        logger.info(f"Object detection enabled using {yolo_config['model_size']} YOLO model")
    
    def start(self) -> None:
        """Start the SmartVan Monitor system."""
        logger.info("Starting SmartVan Monitor")
        
        # Start all cameras
        self.camera_manager.start_all_cameras()
        
        # Start backend client
        self.backend_client.start()
        
        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down")
        finally:
            self.stop()
    
    def stop(self) -> None:
        """Stop the SmartVan Monitor system."""
        logger.info("Stopping SmartVan Monitor")
        
        # Stop backend client
        self.backend_client.stop()
        
        # Stop cameras
        self.camera_manager.stop_all_cameras()
        self.camera_manager.cleanup()
    
    def _main_loop(self) -> None:
        """Main processing loop."""
        logger.info("Starting main processing loop")
        
        # Settings from configuration
        display = self.config.get("display", True)
        
        # Use the fine-tuned FPS settings from your optimization work
        # Get from first camera or use the default of 2 FPS for optimal detection
        cameras_config = self.config.get("cameras", [])
        fps_target = 2  # Default to 2 FPS (500ms intervals) as per your optimized settings
        
        if cameras_config and isinstance(cameras_config, list) and len(cameras_config) > 0:
            fps_target = cameras_config[0].get("fps", fps_target)
            
        frame_interval_ms = max(1, int(1000 / fps_target))
        logger.info(f"Using frame interval of {frame_interval_ms}ms ({fps_target} FPS) for optimal detection")
        
        # FPS tracking
        frame_count = 0
        start_time = time.time()
        fps_display_interval = 5  # seconds
        
        # Main loop
        while True:
            try:
                # Get frames from all cameras
                frames = self.camera_manager.get_all_frames()
                
                # Process each camera
                for camera_name, (frame, timestamp, fps) in frames.items():
                    if frame is None:
                        logger.warning(f"No frame received from {camera_name}")
                        continue
                    
                    try:
                        # Process frame with enhanced detection monitor
                        detection_result = self.enhanced_monitor.process_frame(camera_name, frame, timestamp)
                        
                        # Determine if we should record this event
                        should_record = self.enhanced_monitor.should_record(detection_result)
                        detection_result["should_record"] = should_record
                    except Exception as e:
                        logger.error(f"Error processing frame from {camera_name}: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        continue
                    
                    # Handle recording if needed
                    if should_record:
                        # Start a new recording
                        self._save_event(camera_name, frame, detection_result, timestamp)
                        self.recording_active[camera_name] = True
                        self.last_save_time[camera_name] = timestamp
                    elif camera_name in self.recording_active and self.recording_active[camera_name]:
                        # Check if we need to end recording
                        time_since_last = timestamp - self.last_save_time.get(camera_name, 0)
                        min_recording_gap = self.config.get("recording", {}).get("min_recording_gap", 10)
                        
                        if time_since_last > min_recording_gap:
                            self.recording_active[camera_name] = False
                    
                    # Display frame with visualization
                    if display:
                        # Get enhanced visualization
                        vis_frame = self.enhanced_monitor.visualize(camera_name, frame, detection_result)
                        
                        # Show frame
                        cv2.imshow(f"SmartVan Monitor - {camera_name}", vis_frame)
                
                # FPS calculation and display
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= fps_display_interval:
                    fps = frame_count / elapsed_time
                    logger.info(f"Processing FPS: {fps:.2f}")
                    frame_count = 0
                    start_time = time.time()
                
                # Check for exit key (wait short time between frames)
                key = cv2.waitKey(frame_interval_ms) & 0xFF
                if key == 27 or key == ord('q'):  # ESC or 'q' to quit
                    break
                    
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                import traceback
                logger.error(traceback.format_exc())
                time.sleep(1)  # Prevent error flooding
                
        # Clean up
        for camera_name, camera in self.camera_manager.cameras.items():
            if display:
                cv2.destroyWindow(f"SmartVan Monitor - {camera_name}")
        self.camera_manager.release_all()
    
    def _save_event(self, camera_name: str, frame: np.ndarray, detection_result: Dict[str, Any], timestamp: float) -> None:
        """
        Save an event frame and metadata.
        
        Args:
            camera_name: Name of the camera
            frame: Current video frame
            detection_result: Result from enhanced detection monitor
            timestamp: Frame timestamp
        """
        # Get recording configuration
        recording_config = self.config.get("recording", {})
        
        # Create timestamp string
        dt = datetime.fromtimestamp(timestamp)
        timestamp_str = dt.strftime("%Y%m%d_%H%M%S")
        
        # Get event details from the detection result
        motion_detected = detection_result.get("motion_detected", False)
        motion_intensity = detection_result.get("intensity", 0)
        motion_is_vibration = detection_result.get("is_vibration", False)
        human_detected = detection_result.get("human_detected", False)
        human_confidence = detection_result.get("human_confidence", 0)
        
        # Get inventory tracking data
        inventory_tracking = detection_result.get("inventory_tracking", {})
        inventory_access = inventory_tracking.get("inventory_access_detected", False)
        inventory_change = inventory_tracking.get("inventory_change_detected", False)
        
        # Determine event type for filename
        event_type = "NORMAL"
        if human_detected and human_confidence > 0.7:
            event_type = "HUMAN_PRESENT"
        elif inventory_change:
            event_type = "INVENTORY_CHANGE"
        elif inventory_access:
            event_type = "INVENTORY_ACCESS"
        elif motion_detected and not motion_is_vibration:
            event_type = "MOTION"
        
        # Build filename with descriptive information
        filename_prefix = f"{camera_name}_{timestamp_str}_{event_type}"
        if human_detected:
            filename_prefix += f"_HUMAN{int(human_confidence*100)}"
        if motion_detected:
            filename_prefix += f"_MOTION{motion_intensity}"
        if inventory_change:
            change_type = inventory_tracking.get("change_type", "UNKNOWN")
            filename_prefix += f"_{change_type}"
        
        # Save the frame image
        image_filename = f"{filename_prefix}.jpg"
        image_path = str(self.clips_dir / image_filename)
        cv2.imwrite(image_path, frame)
        
        # Build comprehensive metadata
        metadata = {
            "van_id": detection_result.get("van_id", self.config.get("van_id", "VAN001")),  # Include van_id for fleet identification
            "camera": camera_name,
            "timestamp": timestamp,
            "datetime": dt.isoformat(),
            "image_path": image_path,
            "event_type": event_type,
            
            # Motion data
            "motion": {
                "detected": motion_detected,
                "intensity": motion_intensity,
                "is_vibration": motion_is_vibration,
                "duration": detection_result.get("duration", 0),
            },
            
            # Human detection data
            "human": {
                "detected": human_detected,
                "confidence": human_confidence,
                "bounding_boxes": detection_result.get("bounding_boxes", [])
            }
        }
        
        # Add inventory tracking data if available
        if inventory_tracking:
            # Extract relevant inventory data
            inventory_data = {
                "access_detected": inventory_access,
                "change_detected": inventory_change,
                "zone_id": inventory_tracking.get("zone_id", ""),
                "zone_name": inventory_tracking.get("zone_name", ""),
                "access_duration": inventory_tracking.get("access_duration", 0),
                "change_percentage": inventory_tracking.get("change_percentage", 0),
                "change_type": inventory_tracking.get("change_type", "")
            }
            
            # Add detailed backend data if available
            if "backend_rules_data" in inventory_tracking:
                inventory_data["backend_rules_data"] = inventory_tracking["backend_rules_data"]
                
            # Add image paths if available
            if "before_image_path" in inventory_tracking:
                inventory_data["before_image_path"] = inventory_tracking["before_image_path"]
            if "after_image_path" in inventory_tracking:
                inventory_data["after_image_path"] = inventory_tracking["after_image_path"]
                
            metadata["inventory"] = inventory_data
        
        # Add event classification data if available
        if "classification" in detection_result:
            metadata["classification"] = detection_result["classification"]
        
        # Add event IDs if available for correlation
        if "motion_event_id" in detection_result:
            metadata["motion_event_id"] = detection_result["motion_event_id"]
        if "human_event_id" in detection_result:
            metadata["human_event_id"] = detection_result["human_event_id"]
        if "inventory_event_id" in detection_result:
            metadata["inventory_event_id"] = detection_result["inventory_event_id"]
        
        # Save metadata file
        metadata_filename = f"{filename_prefix}.json"
        metadata_path = str(self.clips_dir / metadata_filename)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4, cls=NumpyJSONEncoder)
        
        # Send event to backend using edge computing approach
        self._send_to_backend(event_type, metadata)
        
        # Log event based on type
        log_message = f"Saved {event_type} event from {camera_name}"
        
        if inventory_change:
            # Log inventory changes with higher visibility
            logger.warning(log_message)
        elif human_detected and human_confidence > 0.7:
            # Log definite human presence with higher visibility
            logger.warning(log_message)
        else:
            # Log other events at info level
            logger.info(log_message)
            
    def _send_to_backend(self, event_type: str, metadata: Dict[str, Any]):
        """
        Send event data to the backend using edge computing approach.
        
        Args:
            event_type: Type of event
            metadata: Event metadata
        """
        # Skip if backend client not initialized or API key not set
        if not hasattr(self, 'backend_client') or not self.backend_client.api_key:
            return
        
        # Determine priority based on event type
        priority = "normal"
        if event_type == "HUMAN_PRESENT" and metadata.get("human", {}).get("confidence", 0) > 0.8:
            priority = "high"  # High priority for confident human detection
        elif event_type == "INVENTORY_CHANGE":
            priority = "high"  # High priority for inventory changes
        
        # For edge computing, we don't send all data - just the important bits
        # Extract only the key information needed by the backend
        edge_data = {
            "van_id": metadata.get("van_id"),
            "camera": metadata.get("camera"),
            "timestamp": metadata.get("timestamp"),
            "datetime": metadata.get("datetime"),
            "event_type": event_type,
            "event_id": metadata.get("motion_event_id", "")
        }
        
        # Add human detection data if present
        human_data = metadata.get("human", {})
        if human_data.get("detected", False):
            edge_data["human_detection"] = {
                "confidence": human_data.get("confidence", 0),
                "count": len(human_data.get("bounding_boxes", []))
            }
        
        # Add inventory data if present
        inventory_data = metadata.get("inventory", {})
        if inventory_data:
            edge_data["inventory"] = {
                "zone_id": inventory_data.get("zone_id", ""),
                "access_detected": inventory_data.get("access_detected", False),
                "change_detected": inventory_data.get("change_detected", False),
                "change_type": inventory_data.get("change_type", "")
            }
        
        # Send to backend client
        self.backend_client.send_event(edge_data, priority=priority)
        
        # Also send performance metrics if available
        if hasattr(self, 'enhanced_monitor') and hasattr(self.enhanced_monitor, 'inventory_detector'):
            inventory_detector = self.enhanced_monitor.inventory_detector
            if hasattr(inventory_detector, 'performance_tracker'):
                performance_tracker = inventory_detector.performance_tracker
                if hasattr(performance_tracker, 'get_recent_metrics'):
                    # Get recent performance metrics
                    metrics = performance_tracker.get_recent_metrics()
                    if metrics:
                        self.backend_client.send_performance_metrics(metrics)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SmartVan Monitor System")
    parser.add_argument("--config", default="config.json", help="Path to configuration file")
    parser.add_argument("--no-display", action="store_true", help="Disable video display")
    parser.add_argument("--vibration-filter", type=int, default=40, 
                        help="Vibration filter threshold (0-100). Higher values filter more vibrations.")
    parser.add_argument("--van-id", type=str, default=None,
                        help="Unique identifier for this van. Overrides the config file setting.")
    parser.add_argument("--backend-url", type=str, default=None,
                        help="URL for the backend server. Overrides the config file setting.")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for backend authentication. Overrides the config file setting.")
    parser.add_argument("--heartbeat-interval", type=int, default=60,
                        help="Interval in seconds for sending heartbeats to the backend.")
    args = parser.parse_args()
    
    # Load configuration
    config_path = args.config
    
    try:
        # Create and start the monitor
        monitor = SmartVanMonitor(config_path=config_path)
        
        # Override display setting if specified
        if args.no_display:
            monitor.config["display"] = False
            
        # Apply custom vibration filter threshold if specified
        if args.vibration_filter != 40:
            logger.info(f"Using custom vibration filter threshold: {args.vibration_filter}")
            # This will be picked up by the EnhancedDetectionMonitor's process_frame method
            monitor.config["motion_detection"] = monitor.config.get("motion_detection", {})
            monitor.config["motion_detection"]["vibration_threshold"] = args.vibration_filter
            
        # Add van_id if specified on command line
        if args.van_id:
            logger.info(f"Setting van identifier to: {args.van_id}")
            monitor.config["van_id"] = args.van_id
            # Make sure the enhanced monitor gets the updated van_id
            monitor.enhanced_monitor.van_id = args.van_id
            # Update van_id in the backend client as well
            monitor.backend_client.van_id = args.van_id
            
        # Override backend URL if specified on command line
        if args.backend_url:
            logger.info(f"Setting backend server URL to: {args.backend_url}")
            if "backend" not in monitor.config:
                monitor.config["backend"] = {}
            monitor.config["backend"]["server_url"] = args.backend_url
            monitor.backend_client.server_url = args.backend_url
            
        # Override API key if specified on command line
        if args.api_key:
            logger.info("Setting backend API key from command line")
            if "backend" not in monitor.config:
                monitor.config["backend"] = {}
            monitor.config["backend"]["api_key"] = args.api_key
            monitor.backend_client.api_key = args.api_key
        
        # Start the monitor
        monitor.start()
    except Exception as e:
        logger.error(f"Error starting SmartVan Monitor: {e}")
        import traceback
        logger.error(traceback.format_exc())
