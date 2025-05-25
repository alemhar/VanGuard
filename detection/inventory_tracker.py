"""
SmartVan Monitor - Inventory Tracking Module
-------------------------------------------
This module integrates inventory change detection with motion and object detection.
It tracks inventory zones, monitors access events, and coordinates the detection process.

Key features:
- Manages inventory zones and zone definitions
- Tracks inventory access sessions
- Coordinates change detection with human presence detection
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any

from .inventory_detection import InventoryChangeDetector
from .event_framework import EnhancedEventClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("inventory_tracker")

class InventoryTracker:
    """
    Class that integrates inventory change detection with motion and object detection.
    Monitors inventory zones and tracks access events.
    """
    
    def __init__(self, 
                 config: Dict[str, Any], 
                 event_classifier: EnhancedEventClassifier,
                 van_id: str = "VAN001"):
        """
        Initialize the inventory tracker.
        
        Args:
            config: Inventory detection configuration
            event_classifier: Event classifier instance
            van_id: Unique identifier for this van
        """
        self.config = config
        self.event_classifier = event_classifier
        self.van_id = van_id
        
        # Default configuration
        self.config.setdefault("access_timeout", 10.0)  # Seconds after access event to check for changes
        self.config.setdefault("human_confidence_threshold", 0.5)  # Minimum confidence for human detection
        self.config.setdefault("change_confidence_threshold", 0.6)  # Minimum confidence for change detection
        self.config.setdefault("enabled", True)  # Whether inventory tracking is enabled
        
        # Track inventory access sessions per camera
        self.active_sessions = {}
        
        # Store previous frames for analysis
        self.prev_frames = {}
        
        # Initialize inventory change detector
        # Extract the output_dir and other parameters from config
        output_dir = self.config.get("output_dir", "output")
        change_threshold = self.config.get("change_threshold", 0.08)
        min_change_area = self.config.get("min_change_area", 200)
        
        # Initialize with the correct parameters
        self.change_detector = InventoryChangeDetector(
            output_dir=output_dir,
            change_threshold=change_threshold,
            min_change_area=min_change_area
        )
        
        logger.info("Inventory tracker initialized")
        if self.config["enabled"]:
            logger.info("Inventory tracking is enabled")
        else:
            logger.info("Inventory tracking is disabled")
    
    def setup_inventory_zones(self, zones_config: Dict[str, List[Dict[str, Any]]]):
        """
        Set up inventory zones from configuration.
        
        Args:
            zones_config: Dictionary mapping camera names to lists of zone definitions
        """
        for camera_name, zones in zones_config.items():
            self.change_detector.register_inventory_zones(camera_name, zones)
            logger.info(f"Registered {len(zones)} inventory zones for camera {camera_name}")
    
    def update_frame(self, camera_name: str, frame: np.ndarray, timestamp: float = None):
        """
        Update the tracker with a new frame from a camera.
        Should be called for each new frame regardless of detection results.
        
        Args:
            camera_name: Name of the camera
            frame: Current frame from the camera
            timestamp: Current timestamp (default: current time)
        """
        if not self.config["enabled"]:
            return
            
        if timestamp is None:
            timestamp = time.time()
        
        # Store the frame for this camera
        self.prev_frames[camera_name] = frame.copy()
        
        # Check for timed-out sessions that need verification
        self._check_pending_sessions(camera_name, timestamp)
    
    def process_detection(self, 
                         camera_name: str, 
                         frame: np.ndarray, 
                         detection_result: Dict[str, Any],
                         timestamp: float = None) -> Dict[str, Any]:
        """
        Process detection results to track inventory access.
        Should be called whenever there is a positive detection result.
        
        Args:
            camera_name: Name of the camera
            frame: Current frame from the camera
            detection_result: Detection result from integrated detector
            timestamp: Current timestamp (default: current time)
            
        Returns:
            Updated detection result with inventory tracking data
        """
        if not self.config["enabled"]:
            return detection_result
            
        if timestamp is None:
            timestamp = time.time()
        
        # Store the frame for reference
        self.prev_frames[camera_name] = frame.copy()
        
        # Check if this is a human detection that might be accessing inventory
        human_detected = detection_result.get("human_detected", False)
        human_confidence = detection_result.get("human_confidence", 0)
        
        # Enhanced result to return
        enhanced_result = detection_result.copy()
        enhanced_result["inventory_tracking"] = {
            "inventory_access_detected": False,
            "active_session": False
        }
        
        # If human detected with sufficient confidence, check for inventory access
        if human_detected and human_confidence >= self.config["human_confidence_threshold"]:
            # Determine if this human is accessing an inventory zone
            # In a real implementation, this would use bounding box overlap with inventory zones
            # For this prototype, we'll assume any human detection might be accessing inventory
            
            # Check if we already have an active session for this camera
            if camera_name in self.active_sessions:
                session = self.active_sessions[camera_name]
                
                # Update the session
                session["last_update"] = timestamp
                
                # If this is new detection after a verification period, verify the change
                if session["status"] == "pending_verification":
                    # Verify inventory change
                    change_result = self._verify_inventory_change(camera_name, frame, session, timestamp)
                    
                    # Add change result to detection result
                    enhanced_result["inventory_tracking"].update({
                        "inventory_change_detected": change_result["change_detected"],
                        "inventory_change_type": change_result["event_type"],
                        "inventory_change_confidence": change_result["confidence"],
                        "zone_id": session["zone_id"]
                    })
                    
                    # Create an inventory event
                    event = self.event_classifier.create_event(
                        camera_name=camera_name,
                        event_category=self.event_classifier.CATEGORY_INVENTORY,
                        event_type=change_result["event_type"],
                        detection_data={**change_result, "van_id": self.van_id},
                        timestamp=timestamp,
                        related_event_id=session.get("start_event_id")
                    )
                    
                    # Add event to result
                    enhanced_result["inventory_tracking"]["inventory_event"] = event
                    
                    # End the session
                    del self.active_sessions[camera_name]
                    
                else:
                    # Update the active session
                    enhanced_result["inventory_tracking"]["active_session"] = True
                    enhanced_result["inventory_tracking"]["zone_id"] = session["zone_id"]
                    enhanced_result["inventory_tracking"]["session_duration"] = timestamp - session["start_time"]
            
            else:
                # Start a new inventory access session
                # In a real implementation, we would determine which inventory zone is being accessed
                # For now, we'll use a placeholder zone ID "main_inventory"
                zone_id = "main_inventory"
                
                # Create an inventory access event
                event = self.event_classifier.create_event(
                    camera_name=camera_name,
                    event_category=self.event_classifier.CATEGORY_INVENTORY,
                    event_type=self.event_classifier.EVENT_INVENTORY_ACCESS,
                    detection_data={
                        "human_confidence": human_confidence,
                        "zone_id": zone_id,
                        "van_id": self.van_id
                    },
                    timestamp=timestamp
                )
                
                # Start tracking this access
                session = {
                    "zone_id": zone_id,
                    "start_time": timestamp,
                    "start_event_id": event["event_id"],
                    "last_update": timestamp,
                    "status": "active"
                }
                
                self.active_sessions[camera_name] = session
                
                # Start inventory access in the detector
                access_start = self.change_detector.start_inventory_access(
                    camera_name, frame, zone_id, timestamp
                )
                
                # Update result
                enhanced_result["inventory_tracking"].update({
                    "inventory_access_detected": True,
                    "zone_id": zone_id,
                    "active_session": True,
                    "inventory_event": event
                })
                
                logger.info(f"Started inventory access tracking for {camera_name}, zone {zone_id}")
        
        # Handle case where we have an active session but no human detected
        elif camera_name in self.active_sessions:
            session = self.active_sessions[camera_name]
            
            # If we haven't seen human for a while, mark session for verification
            time_since_update = timestamp - session["last_update"]
            
            if time_since_update > self.config["access_timeout"] and session["status"] == "active":
                # Mark session for verification
                session["status"] = "pending_verification"
                logger.info(f"Marking inventory session for verification: {camera_name}, zone {session['zone_id']}")
                
                # Update result
                enhanced_result["inventory_tracking"].update({
                    "active_session": True,
                    "session_status": "pending_verification",
                    "zone_id": session["zone_id"],
                    "session_duration": timestamp - session["start_time"]
                })
        
        return enhanced_result
    
    def _check_pending_sessions(self, camera_name: str, timestamp: float):
        """
        Check for sessions that are pending verification and verify if needed.
        
        Args:
            camera_name: Name of the camera
            timestamp: Current timestamp
        """
        if camera_name not in self.active_sessions:
            return
            
        session = self.active_sessions[camera_name]
        
        # Skip if not pending verification
        if session["status"] != "pending_verification":
            return
            
        # Check if we should verify now (give a bit more time after marking)
        time_since_update = timestamp - session["last_update"]
        if time_since_update > 3.0:  # 3 seconds grace period
            # Get the current frame
            frame = self.prev_frames.get(camera_name)
            if frame is not None:
                # Verify inventory change
                self._verify_inventory_change(camera_name, frame, session, timestamp)
                
                # End the session
                del self.active_sessions[camera_name]
    
    def _verify_inventory_change(self, 
                              camera_name: str, 
                              frame: np.ndarray, 
                              session: Dict[str, Any],
                              timestamp: float) -> Dict[str, Any]:
        """
        Verify if there was a change to inventory during a session.
        
        Args:
            camera_name: Name of the camera
            frame: Current frame
            session: Session data
            timestamp: Current timestamp
            
        Returns:
            Change detection result
        """
        zone_id = session["zone_id"]
        
        # End inventory access in the detector
        change_result = self.change_detector.end_inventory_access(
            camera_name, frame, zone_id, timestamp
        )
        
        # Log the result
        if change_result["change_detected"]:
            logger.info(
                f"Inventory change detected: {change_result['event_type']} in {camera_name}, " +
                f"zone {zone_id} with {change_result['confidence']} confidence"
            )
        else:
            logger.info(f"No inventory change detected in {camera_name}, zone {zone_id}")
        
        return change_result
    
    def visualize(self, 
                 camera_name: str, 
                 frame: np.ndarray, 
                 detection_result: Dict[str, Any] = None) -> np.ndarray:
        """
        Visualize inventory zones and access status.
        
        Args:
            camera_name: Name of the camera
            frame: Current frame
            detection_result: Optional detection result with inventory data
            
        Returns:
            Frame with visualizations
        """
        if not self.config["enabled"]:
            return frame
            
        # Start with inventory detector visualization
        vis_frame = self.change_detector.visualize(frame, camera_name)
        
        # If we have detection results with inventory data, add that
        if detection_result and "inventory_tracking" in detection_result:
            tracking_data = detection_result["inventory_tracking"]
            
            # If there's an active session, highlight it
            if tracking_data.get("active_session", False):
                zone_id = tracking_data.get("zone_id", "unknown")
                session_duration = tracking_data.get("session_duration", 0)
                
                # Add inventory access text at the bottom of the frame
                text = f"Inventory Access: Zone {zone_id} ({session_duration:.1f}s)"
                cv2.putText(
                    vis_frame,
                    text,
                    (10, vis_frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),  # Red
                    2
                )
            
            # If there was a change detected, highlight it
            if tracking_data.get("inventory_change_detected", False):
                change_type = tracking_data.get("inventory_change_type", "unknown")
                confidence = tracking_data.get("inventory_change_confidence", 0)
                
                # Add change text at the bottom of the frame
                text = f"Inventory Change: {change_type} (Conf: {confidence})"
                cv2.putText(
                    vis_frame,
                    text,
                    (10, vis_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),  # Red
                    2
                )
        
        return vis_frame
