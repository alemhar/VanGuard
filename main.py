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

# Import local modules
from utils.camera import CameraManager
from detection.motion_detection import MotionDetector
from detection.event_classification import EventClassifier
from detection.object_detection.yolo_detector import YOLODetector
from detection.integrated_detector import IntegratedDetector

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
        
        # Initialize detection systems (one per camera)
        self.motion_detectors = {}
        self.integrated_detectors = {}
        self._setup_motion_detectors()
        self._setup_integrated_detectors()
        
        # Initialize event classifier with configurable parameters
        event_config = self.config.get("event_classification", {})
        self.event_classifier = EventClassifier(
            business_hours_start=event_config.get("business_hours_start", 7),
            business_hours_end=event_config.get("business_hours_end", 19),
            high_frequency_threshold=event_config.get("high_frequency_threshold", 5),
            high_frequency_window=event_config.get("high_frequency_window", 900),
            event_memory_window=event_config.get("event_memory_window", 3600)
        )
        
        # Set additional configurable parameters
        self.event_classifier.min_alert_interval = event_config.get("min_alert_interval", 120)
        
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
        logger.info("Starting main processing loop")
        
        # Settings from configuration
        display = self.config.get("display", True)
        object_detection_enabled = self.config.get("object_detection", {}).get("enabled", True)
        
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
                        continue
                    
                    # Use integrated detector if enabled, otherwise fall back to motion-only
                    if object_detection_enabled and camera_name in self.integrated_detectors:
                        # Perform integrated detection (motion + YOLO)
                        integrated_detector = self.integrated_detectors[camera_name]
                        detection_result = integrated_detector.detect(frame, timestamp)
                        
                        # Process detection results if motion or human detected
                        if detection_result["motion_detected"] or detection_result["human_detected"]:
                            self._process_detection(camera_name, frame, detection_result, timestamp)
                        
                        # Display the frame with visualizations if enabled
                        if display:
                            # Visualize integrated detection results
                            vis_frame = integrated_detector.visualize(frame, detection_result)
                            
                            # Display the frame
                            cv2.imshow(f"SmartVan Monitor - {camera_name}", vis_frame)
                    else:
                        # Fall back to motion-only detection
                        motion_detector = self.motion_detectors.get(camera_name)
                        if motion_detector is None:
                            continue
                        
                        # Detect motion
                        motion_result = motion_detector.detect(frame)
                        
                        # Process motion detection results if motion detected
                        if motion_result["motion_detected"]:
                            self._process_detection(camera_name, frame, motion_result, timestamp)
                        
                        # Display the frame with visualizations if enabled
                        if display:
                            # Visualize motion detection
                            vis_frame = motion_detector.visualize(frame, motion_result)
                            
                            # Display the frame
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
                if display and cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, stopping")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Clean up
        if display:
            cv2.destroyAllWindows()
    
    def _process_detection(self, camera_name: str, frame: any, 
                  detection_result: dict, timestamp: float) -> None:
        """
        Process detection results and classify events when motion or human is detected.
        
        Args:
            camera_name: Name of the camera
            frame: Current video frame
            detection_result: Result from motion detector or integrated detector
            timestamp: Frame timestamp
        """
        # Get recording and event classification configuration
        recording_config = self.config.get("recording", {})
        event_config = self.config.get("event_classification", {})
        
        # Use configurable minimum intensity thresholds
        min_intensity = recording_config.get("min_intensity", 20)  # For recording
        min_event_intensity = event_config.get("motion_intensity_threshold", 40)  # For classification
        min_recording_gap = recording_config.get("min_recording_gap", 10)
        
        # Check if motion is significant enough for processing
        # Determine if this is an integrated detection result
        is_integrated = "human_detected" in detection_result
        
        # Get values with defaults to prevent KeyError exceptions
        motion_detected = detection_result.get("motion_detected", False)
        intensity = detection_result.get("intensity", 0)
        confidence = detection_result.get("confidence", 0)
        is_vibration = detection_result.get("is_vibration", False)
        likely_human = detection_result.get("likely_human", False)
        
        # For integrated detection results, also check for human detection
        if is_integrated:
            human_detected = detection_result.get("human_detected", False)
            # If human is detected via YOLO, mark as likely human for processing
            if human_detected:
                likely_human = True
        
        # Skip processing entirely for vibrations with low intensity
        if is_vibration and intensity < min_event_intensity:
            return
            
        # Skip processing for non-human movement below higher threshold
        if not likely_human and intensity < min_event_intensity * 0.8:
            return
            
        # More sophisticated evaluation of motion significance that excludes vibration
        significant_motion = motion_detected and not is_vibration and (
            # Either high intensity
            intensity >= min_intensity or
            # or high confidence with moderate intensity
            (confidence >= 2 and intensity >= min_intensity * 0.7) or
            # or likely human with any confidence
            likely_human
        )
        
        # If this is an integrated result, adjust intensity based on human detection
        if is_integrated and detection_result.get("human_detected", False):
            # Boost intensity for human detections
            human_confidence = detection_result.get("human_confidence", 0)
            intensity = max(intensity, int(human_confidence * 100))
        
        # Check if intensity exceeds threshold or human is detected with high confidence
        min_intensity = self.config.get("recording", {}).get("min_intensity", 20)
        object_recording_threshold = self.config.get("object_detection", {}).get("object_recording_threshold", 0.6)
        
        should_record = False
        if intensity >= min_intensity:
            should_record = True
        elif is_integrated and detection_result.get("human_detected", False):
            human_confidence = detection_result.get("human_confidence", 0)
            if human_confidence >= object_recording_threshold:
                should_record = True
        
        if should_record:
            # Calculate time since last save for this camera
            current_time = time.time()
            last_save_time = self.last_save_time.get(camera_name, 0)
            time_since_last_save = current_time - last_save_time
            
            # Enhanced result with additional details
            enhanced_result = detection_result.copy()
            
            # Classify the event (with special handling for integrated detection)
            if is_integrated:
                # Use the motion_result part of the integrated detection for classification
                motion_part = detection_result.get("motion_result", {})
                
                # Enhance with object detection information
                motion_part["likely_human"] = detection_result.get("human_detected", False)
                motion_part["human_confidence"] = detection_result.get("human_confidence", 0)
                
                # If YOLO detected a person, mark as likely human with high confidence
                if detection_result.get("human_detected", False):
                    motion_part["likely_human"] = True
                    objects = detection_result.get("objects_detected", [])
                    # Add detected objects to the motion result
                    motion_part["detected_objects"] = objects
                
                classified_event = self.event_classifier.classify_event(
                    camera_name, motion_part, timestamp
                )
            else:
                # Regular motion-only classification
                classified_event = self.event_classifier.classify_event(
                    camera_name, detection_result, timestamp
                )
            
            # Add classification data to enhanced result
            enhanced_result["classification"] = {
                "event_type": classified_event["event_type"],
                "confidence": classified_event["confidence"],
                "reasons": classified_event["reasons"]
            }
            
            # Start a new recording
            self._save_event(camera_name, frame, enhanced_result, timestamp)
            self.recording_active[camera_name] = True
            self.last_save_time[camera_name] = current_time
            
            # Skip logging if event was rate-limited
            if classified_event.get('rate_limited', False):
                return
                
            # Log with appropriate severity based on confidence and event type
            if classified_event["confidence"] >= 3:
                objects_str = ""
                if is_integrated and detection_result.get("objects_detected", []):
                    objects = [f"{obj['class']} ({obj['confidence']:.2f})" for obj in detection_result.get("objects_detected", [])]
                    objects_str = f" - Objects: {', '.join(objects)}"
                
                logger.warning(f"Important event detected on {camera_name}: {classified_event['event_type']} - {', '.join(classified_event['reasons'])}{objects_str}")
            elif classified_event["confidence"] > 0:  # Skip logging for zero-confidence events
                logger.info(f"Event detected on {camera_name}: {classified_event['event_type']} - Intensity {intensity}")
        else:
            # No significant detection, mark as not recording
            self.recording_active[camera_name] = False
    
    def _save_event(self, camera_name: str, frame: any, 
                   detection_result: dict, timestamp: float) -> None:
        """
        Save a detection event (image and metadata).
        
        Args:
            camera_name: Name of the camera
            frame: Current video frame
            detection_result: Result from motion detector or integrated detector
            timestamp: Frame timestamp
        """
        # Determine if this is an integrated detection result
        is_integrated = "human_detected" in detection_result
        
        # Create timestamp string
        dt = datetime.fromtimestamp(timestamp)
        timestamp_str = dt.strftime("%Y%m%d_%H%M%S")
        
        # Add event type to filename if available
        event_type = "NORMAL"
        if "classification" in detection_result:
            event_type = detection_result["classification"]["event_type"]
        
        # Add human detection info to filename if available
        human_detected = detection_result.get("human_detected", False) if is_integrated else False
        human_indicator = "_HUMAN" if human_detected else ""
        
        # Add intensity to filename
        intensity = detection_result.get("intensity", 0)
            
        # Save image
        image_filename = f"{camera_name}_{timestamp_str}_{event_type}{human_indicator}_intensity{intensity}.jpg"
        image_path = os.path.join(self.output_dir, "clips", image_filename)
        cv2.imwrite(image_path, frame)
        
        # Save metadata with enhanced information
        metadata = {
            "camera": camera_name,
            "timestamp": timestamp,
            "datetime": dt.isoformat(),
            "motion_detected": detection_result.get("motion_detected", False),
            "intensity": intensity,
            "duration": detection_result.get("duration", 0),
            "confidence": detection_result.get("confidence", 0),
            "bounding_boxes": detection_result.get("bounding_boxes", []),
            "image_path": image_path
        }
        
        # Add YOLO detection data if available
        if is_integrated:
            metadata["human_detected"] = human_detected
            metadata["human_confidence"] = detection_result.get("human_confidence", 0)
            
            # Add object detection data
            if "objects_detected" in detection_result:
                objects = []
                for obj in detection_result["objects_detected"]:
                    objects.append({
                        "class": obj["class"],
                        "confidence": obj["confidence"],
                        "box": obj["box"]
                    })
                metadata["objects_detected"] = objects
            
            # Add YOLO-specific information
            if "yolo_result" in detection_result:
                yolo_result = detection_result["yolo_result"]
                metadata["yolo_inference_time"] = yolo_result.get("inference_time", 0)
                metadata["yolo_ran"] = not yolo_result.get("yolo_skipped", True)
        
        # Add classification data if available
        if "classification" in detection_result:
            metadata["classification"] = detection_result["classification"]
        
        # Save metadata file
        metadata_filename = f"{camera_name}_{timestamp_str}_{event_type}{human_indicator}_intensity{intensity}.json"
        metadata_path = os.path.join(self.output_dir, "clips", metadata_filename)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Saved event to {image_path}")
        
        # Log additional information for human detections
        if human_detected:
            human_confidence = detection_result.get("human_confidence", 0)
            logger.info(f"Human detected with confidence: {human_confidence:.2f}")


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
