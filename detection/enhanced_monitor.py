"""
SmartVan Monitor - Enhanced Detection Module
-------------------------------------------
This module integrates all detection components for the SmartVan Monitor system,
with a focus on robust false positive filtering and accurate inventory change detection.
"""

import cv2
import numpy as np
import time
import logging
import os
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from .motion_detection import MotionDetector
from .object_detection.yolo_detector import YOLODetector
from .integrated_detector import IntegratedDetector
from .inventory_detection import InventoryChangeDetector
from .inventory_tracker import InventoryTracker
from .event_framework import EnhancedEventClassifier
from .config_defaults import get_default_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("enhanced_monitor")

class EnhancedDetectionMonitor:
    """
    Enhanced detection monitor that integrates motion detection, object detection,
    and inventory change detection with robust false positive filtering.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the enhanced detection monitor.
        
        Args:
            config: Configuration dictionary (optional)
        """
        # Load configuration with defaults but preserve original config for reference
        self.original_config = config or {}
        self.config = get_default_config(config)
        
        # Handle legacy configuration format 
        # Create output directories
        if "output_dir" in self.original_config:
            self.output_dir = Path(self.original_config.get("output_dir", "output"))
        else:
            self.output_dir = Path(self.config.get("system", {}).get("output_dir", "output"))
            
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize core components
        self._setup_components()
        
        # Detection state
        self.last_save_time = {}
        self.recording_active = {}
        
        logger.info("Enhanced detection monitor initialized")
        
    def _setup_components(self):
        """Set up the detection components."""
        # Get configuration sections - handle both new and legacy format
        motion_config = self.config.get("motion_detection", {})
        human_config = self.config.get("human_detection", {})
        event_config = self.config.get("event_classification", {})
        inventory_config = self.config.get("inventory_detection", {})
        
        # Handle legacy format for event classification
        if "event_classification" in self.original_config:
            legacy_event_config = self.original_config.get("event_classification", {})
            event_config.update(legacy_event_config)
            
        # Handle legacy format for object detection
        if "object_detection" in self.original_config:
            legacy_object_config = self.original_config.get("object_detection", {})
            human_config.update(legacy_object_config)
        
        # Initialize enhanced event classifier
        self.event_classifier = EnhancedEventClassifier(
            output_dir=str(self.output_dir),
            business_hours_start=event_config.get("business_hours_start", 7),
            business_hours_end=event_config.get("business_hours_end", 19),
            high_frequency_threshold=event_config.get("high_frequency_threshold", 5),
            high_frequency_window=event_config.get("high_frequency_window", 900),
            min_alert_interval=event_config.get("min_alert_interval", 120)
        )
        
        # Initialize inventory detector
        self.inventory_detector = InventoryChangeDetector(
            output_dir=str(self.output_dir),
            change_threshold=inventory_config.get("change_threshold", 0.08),
            min_change_area=inventory_config.get("min_change_area", 200)
        )
        
        # Initialize inventory tracker
        self.inventory_tracker = InventoryTracker(
            inventory_detector=self.inventory_detector,
            event_classifier=self.event_classifier,
            config={
                "access_timeout": inventory_config.get("access_timeout", 5.0),
                "human_confidence_threshold": inventory_config.get("human_confidence_threshold", 0.5),
                "enabled": inventory_config.get("enabled", True)
            }
        )
        
        # Initialize detector collections - to be populated per camera
        self.motion_detectors = {}
        self.integrated_detectors = {}
        
        logger.info("Detection components initialized")
    
    def setup_camera(self, camera_name: str, roi_areas: List[List[int]] = None):
        """
        Set up detection components for a camera.
        
        Args:
            camera_name: Name of the camera
            roi_areas: Optional list of ROI areas as [x, y, width, height]
        """
        logger.info(f"Setting up camera {camera_name} with EnhancedDetectionMonitor")
        
        try:
            # Create a simple MotionDetector with standard parameters
            # We'll use our optimized parameters in the config for the actual detection process
            motion_detector = MotionDetector(
                algorithm=MotionDetector.ALGORITHM_MOG2,
                learning_rate=0.001,
                motion_threshold=25,
                min_area=300,
                blur_size=21,
                noise_filter=0.6,
                movement_stability=7  # Your optimized value - 7 consecutive frames for motion confirmation
            )
            
            # Apply our optimized flow detection config via the human_detection_config
            # This will affect how motion is classified during detection
            optimized_human_detection = {
                "enabled": True,
                "flow_magnitude_threshold": 3.5,    # Optimized value for better vibration detection
                "flow_uniformity_threshold": 0.5,   # Optimized value for better vibration detection
                "significant_motion_threshold": 6.0, # Optimized value for genuine motion detection
                "min_duration": 1.5                # Optimized value for motion duration threshold
            }
            
            # Update the detector with our optimized parameters
            motion_detector.human_detection_config = optimized_human_detection
            motion_detector.flow_magnitude_threshold = 3.5
            motion_detector.flow_uniformity_threshold = 0.5
            motion_detector.significant_motion_threshold = 6.0
            motion_detector.human_min_duration = 1.5
            
            logger.info("Applied optimized vibration detection parameters")
        except Exception as e:
            logger.error(f"Error setting up motion detector: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Create a fallback motion detector with minimal parameters
            motion_detector = MotionDetector()
        
        # Add ROI areas if provided
        if roi_areas:
            for roi in roi_areas:
                motion_detector.add_roi(*roi)
        
        # Get human detection configuration, checking in both possible locations for compatibility
        human_config = {}
        
        # Check the new config structure
        if "human_detection" in self.config:
            human_config.update(self.config["human_detection"])
            
        # Check the legacy config structure
        if "object_detection" in self.original_config:
            human_config.update(self.original_config["object_detection"])
        
        # Initialize YOLO detector with compatible configuration parameters
        yolo_config = {
            "model_size": human_config.get("model_size", "tiny"),
            "confidence_threshold": human_config.get("confidence_threshold", 0.5),
            "nms_threshold": human_config.get("nms_threshold", 0.4),
            # Handle parameter name differences between systems
            "only_on_motion": not human_config.get("always_run", False),
            "gpu_enabled": False
        }
        
        logger.info(f"YOLO configuration: {yolo_config}")
        
        # Initialize integrated detector with robust error handling
        try:
            # Get the parameters needed for integrated detector
            always_run_yolo = human_config.get("always_run", False)
            yolo_interval = human_config.get("run_interval", 5)
            confidence_threshold = human_config.get("confidence_threshold", 0.5)
            
            logger.info(f"Setting up IntegratedDetector for {camera_name} with YOLO interval: {yolo_interval}")
            
            # Create the integrated detector with your optimized settings
            integrated_detector = IntegratedDetector(
                motion_detector=motion_detector,
                yolo_config=yolo_config,
                always_run_yolo=always_run_yolo,
                yolo_interval=yolo_interval,
                human_confirmation_needed=True,  # Use YOLO confirmation for higher confidence
                object_recording_threshold=confidence_threshold
            )
            
            logger.info(f"Successfully created IntegratedDetector for camera {camera_name}")
        except Exception as e:
            logger.error(f"Error creating IntegratedDetector: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise  # This is a critical component, so we need to fail if it can't be created
        
        # Store detectors
        self.motion_detectors[camera_name] = motion_detector
        self.integrated_detectors[camera_name] = integrated_detector
        
        # Set up inventory zones (using full frame as single zone for simplicity)
        full_frame_zone = [{
            "id": "main_inventory",
            "name": f"{camera_name}_Inventory",
            "x": 0,
            "y": 0,
            "width": 640,  # Default width
            "height": 480  # Default height
        }]
        
        zones_config = {camera_name: full_frame_zone}
        self.inventory_tracker.setup_inventory_zones(zones_config)
        
        logger.info(f"Detection components set up for camera {camera_name}")
    
    def process_frame(self, camera_name: str, frame: np.ndarray, timestamp: float = None) -> Dict[str, Any]:
        """
        Process a frame from a camera.
        
        Args:
            camera_name: Name of the camera
            frame: Current frame from the camera
            timestamp: Current timestamp (default: current time)
            
        Returns:
            Enhanced detection result
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Ensure we have detectors for this camera
        if camera_name not in self.integrated_detectors:
            self.setup_camera(camera_name)
        
        # Always update inventory tracker with the latest frame
        self.inventory_tracker.update_frame(camera_name, frame, timestamp)
        
        # Get integrated detector for this camera
        integrated_detector = self.integrated_detectors[camera_name]
        
        # Run detection with integrated detector
        detection_result = integrated_detector.detect(frame, timestamp)
        
        # Apply optimized vibration filtering - completely ignore low intensity vibrations
        # based on the fine-tuned threshold of 40 from your testing
        motion_intensity_threshold = self.config.get("event_classification", {}).get("motion_intensity_threshold", 40)
        
        # If this is a vibration with intensity below threshold, filter it out
        if detection_result.get("is_vibration", False) and detection_result.get("intensity", 0) < motion_intensity_threshold:
            # Downgrade detection for low-intensity vibrations
            detection_result["motion_detected"] = False
            detection_result["confidence"] = 0
            logger.debug(f"Filtered out low intensity vibration: {detection_result.get('intensity', 0)} (threshold: {motion_intensity_threshold})")
            
        # For vibrations that are not completely filtered, still mark them for special handling
        elif detection_result.get("is_vibration", False):
            # Lower confidence for vibrations but keep detection
            original_confidence = detection_result.get("confidence", 0)
            detection_result["confidence"] = max(0, original_confidence - 1) 
            logger.debug(f"Reduced confidence for vibration: {detection_result.get('intensity', 0)}")
        
        # Override with vibration_threshold from command line if set
        if "vibration_threshold" in self.config.get("motion_detection", {}):
            custom_threshold = self.config["motion_detection"]["vibration_threshold"]
            if detection_result.get("is_vibration", False) and detection_result.get("intensity", 0) < custom_threshold:
                detection_result["motion_detected"] = False
                detection_result["confidence"] = 0
                logger.debug(f"Filtered with custom threshold: {detection_result.get('intensity', 0)} (threshold: {custom_threshold})")
                
        
        # Add timestamp to detection result
        detection_result["timestamp"] = timestamp
        
        # Process with inventory tracker for enhanced information
        enhanced_result = self.inventory_tracker.process_detection(
            camera_name, frame, detection_result, timestamp
        )
        
        # Create events based on detection results
        self._create_events_from_detection(camera_name, enhanced_result, timestamp)
        
        return enhanced_result
    
    def _create_events_from_detection(self, camera_name: str, detection_result: Dict[str, Any], timestamp: float):
        """
        Create events based on detection results.
        
        Args:
            camera_name: Name of the camera
            detection_result: Detection result from integrated detector
            timestamp: Current timestamp
        """
        # Get basic detection data
        motion_detected = detection_result.get("motion_detected", False)
        motion_intensity = detection_result.get("intensity", 0)
        motion_is_vibration = detection_result.get("is_vibration", False)
        human_detected = detection_result.get("human_detected", False)
        human_confidence = detection_result.get("human_confidence", 0)
        
        # Skip if no motion and no human detected
        if not motion_detected and not human_detected:
            return
        
        # Get inventory tracking data
        inventory_tracking = detection_result.get("inventory_tracking", {})
        inventory_access = inventory_tracking.get("inventory_access_detected", False)
        inventory_change = inventory_tracking.get("inventory_change_detected", False)
        
        # Create a motion event if motion is detected and it's not just vibration
        # or if vibration is significant enough
        if motion_detected and (not motion_is_vibration or motion_intensity > 40):
            motion_event = self.event_classifier.create_event(
                camera_name=camera_name,
                event_category=self.event_classifier.CATEGORY_MOTION,
                event_type=self.event_classifier.EVENT_MOTION,
                detection_data={
                    "intensity": motion_intensity,
                    "is_vibration": motion_is_vibration,
                    "bounding_boxes": detection_result.get("bounding_boxes", []),
                    "duration": detection_result.get("duration", 0)
                },
                timestamp=timestamp
            )
            
            # Save motion event ID for reference
            detection_result["motion_event_id"] = motion_event["event_id"]
        
        # Create a human event if human is detected with sufficient confidence
        if human_detected and human_confidence >= 0.5:
            human_event = self.event_classifier.create_event(
                camera_name=camera_name,
                event_category=self.event_classifier.CATEGORY_HUMAN,
                event_type=self.event_classifier.EVENT_HUMAN_PRESENT,
                detection_data={
                    "human_confidence": human_confidence,
                    "objects_detected": detection_result.get("objects_detected", []),
                    "motion_intensity": motion_intensity
                },
                timestamp=timestamp,
                # Link to motion event if it exists
                related_event_id=detection_result.get("motion_event_id")
            )
            
            # Save human event ID for reference
            detection_result["human_event_id"] = human_event["event_id"]
        
        # Inventory events are handled by the inventory tracker
        # but we can reference them here if needed
        if "inventory_event" in inventory_tracking:
            inventory_event = inventory_tracking["inventory_event"]
            detection_result["inventory_event_id"] = inventory_event["event_id"]
            
            # If this is a significant inventory event, flag it for attention
            if inventory_event["type"] in [
                self.event_classifier.EVENT_ITEM_REMOVAL,
                self.event_classifier.EVENT_ITEM_ADDITION
            ] and inventory_event["confidence"] >= self.event_classifier.CONFIDENCE_MEDIUM:
                detection_result["significant_inventory_event"] = True
    
    def should_record(self, detection_result: Dict[str, Any]) -> bool:
        """
        Determine if the current detection should trigger recording.
        
        Args:
            detection_result: Detection result from process_frame
            
        Returns:
            True if recording should be triggered
        """
        # Get recording configuration
        recording_config = self.config.get("recording", {})
        min_intensity = recording_config.get("min_intensity", 20)
        
        # Check if motion is significant enough
        motion_detected = detection_result.get("motion_detected", False)
        motion_intensity = detection_result.get("intensity", 0)
        is_vibration = detection_result.get("is_vibration", False)
        
        # Human detection has higher priority
        human_detected = detection_result.get("human_detected", False)
        human_confidence = detection_result.get("human_confidence", 0)
        
        # Inventory tracking
        inventory_tracking = detection_result.get("inventory_tracking", {})
        inventory_access = inventory_tracking.get("inventory_access_detected", False)
        inventory_change = inventory_tracking.get("inventory_change_detected", False)
        
        # Automatic recording triggers
        should_record = False
        
        # Strong human detection should always record
        if human_detected and human_confidence >= 0.7:
            should_record = True
        
        # Significant motion that's not vibration should record
        elif motion_detected and motion_intensity >= min_intensity and not is_vibration:
            should_record = True
        
        # Inventory access or change should record
        elif inventory_access or inventory_change:
            should_record = True
        
        # Significant inventory event should always record
        if detection_result.get("significant_inventory_event", False):
            should_record = True
        
        return should_record
    
    def visualize(self, camera_name: str, frame: np.ndarray, detection_result: Dict[str, Any]) -> np.ndarray:
        """
        Create a visualization of the detection result.
        
        Args:
            camera_name: Name of the camera
            frame: Current frame from the camera
            detection_result: Detection result from process_frame
            
        Returns:
            Frame with visualizations
        """
        # Get integrated detector for this camera
        if camera_name not in self.integrated_detectors:
            return frame
            
        integrated_detector = self.integrated_detectors[camera_name]
        
        # Start with integrated detector visualization
        vis_frame = integrated_detector.visualize(frame, detection_result)
        
        # Add inventory tracker visualization
        vis_frame = self.inventory_tracker.visualize(camera_name, vis_frame, detection_result)
        
        # Add recording status if needed
        if detection_result.get("should_record", False):
            # Add "RECORDING" text at the top
            cv2.putText(
                vis_frame,
                "RECORDING",
                (vis_frame.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),  # Red
                2
            )
        
        return vis_frame
