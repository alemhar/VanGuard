"""
SmartVan Monitor - Inventory Change Detection Module
---------------------------------------------------
This module implements inventory change detection for the SmartVan Monitor system.

Key features:
- Capture and compare before/after inventory states
- Basic image differencing for change detection
- Item removal/addition detection
- Classification of inventory interaction events
"""

import cv2
import numpy as np
import time
import logging
import os
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("inventory_detection")

class InventoryChangeDetector:
    """Class for detecting changes in inventory areas and classifying inventory events."""
    
    # Event types
    EVENT_INVENTORY_ACCESS = "INVENTORY_ACCESS"  # General access to inventory area
    EVENT_ITEM_REMOVAL = "ITEM_REMOVAL"          # Item likely removed from inventory
    EVENT_ITEM_ADDITION = "ITEM_ADDITION"        # Item likely added to inventory
    EVENT_NO_CHANGE = "NO_CHANGE"                # Access without apparent inventory change
    
    # Change confidence levels
    CONFIDENCE_HIGH = 3
    CONFIDENCE_MEDIUM = 2
    CONFIDENCE_LOW = 1
    CONFIDENCE_NONE = 0
    
    def __init__(self, 
                output_dir: str = "output",
                change_threshold: float = 0.15,   # Percentage change required for detection
                min_change_area: int = 200,       # Minimum size of change area in pixels
                state_memory_window: int = 3600): # How long to remember inventory states (seconds)
        """
        Initialize the inventory change detector.
        
        Args:
            output_dir: Directory to store inventory state images
            change_threshold: Threshold for significant pixel changes (0.0-1.0)
            min_change_area: Minimum contour area to consider as meaningful change
            state_memory_window: How long to keep inventory states in memory (seconds)
        """
        self.output_dir = Path(output_dir)
        self.inventory_states_dir = self.output_dir / "inventory_states"
        self.inventory_states_dir.mkdir(exist_ok=True)
        
        self.change_threshold = change_threshold
        self.min_change_area = min_change_area
        self.state_memory_window = state_memory_window
        
        # Inventory state storage
        # camera_name -> zone_id -> {"image": image, "timestamp": timestamp}
        self.inventory_states = {}
        
        # Track ongoing inventory access events
        # camera_name -> {"start_time": time, "zone_id": zone_id, "start_image": image}
        self.active_access_events = {}
        
        logger.info("Inventory change detector initialized")
        logger.info(f"Change threshold: {self.change_threshold}, Min area: {self.min_change_area}")
    
    def register_inventory_zones(self, camera_name: str, zones: List[Dict[str, Any]]) -> None:
        """
        Register inventory zones for a camera.
        
        Args:
            camera_name: Name of the camera
            zones: List of zones, each with {id, name, x, y, width, height}
        """
        if camera_name not in self.inventory_states:
            self.inventory_states[camera_name] = {}
        
        for zone in zones:
            zone_id = zone["id"]
            self.inventory_states[camera_name][zone_id] = {
                "image": None,
                "timestamp": 0,
                "metadata": {
                    "name": zone.get("name", f"Zone-{zone_id}"),
                    "x": zone["x"],
                    "y": zone["y"],
                    "width": zone["width"],
                    "height": zone["height"]
                }
            }
        
        logger.info(f"Registered {len(zones)} inventory zones for camera {camera_name}")
    
    def _extract_zone_image(self, frame: np.ndarray, zone: Dict[str, Any]) -> np.ndarray:
        """
        Extract the portion of the frame corresponding to the specified zone.
        
        Args:
            frame: Full camera frame
            zone: Zone metadata with x, y, width, height
            
        Returns:
            Image of the zone area
        """
        x = zone["x"]
        y = zone["y"]
        width = zone["width"]
        height = zone["height"]
        
        # Ensure coordinates are within frame bounds
        frame_height, frame_width = frame.shape[:2]
        x = max(0, min(x, frame_width - 1))
        y = max(0, min(y, frame_height - 1))
        width = min(width, frame_width - x)
        height = min(height, frame_height - y)
        
        return frame[y:y+height, x:x+width].copy()
    
    def update_inventory_state(self, camera_name: str, frame: np.ndarray, 
                              zone_id: str, timestamp: float = None) -> None:
        """
        Update the stored state of an inventory zone.
        
        Args:
            camera_name: Name of the camera
            frame: Current camera frame
            zone_id: ID of the inventory zone
            timestamp: Current timestamp (default: current time)
        """
        if timestamp is None:
            timestamp = time.time()
        
        if camera_name not in self.inventory_states:
            logger.warning(f"Camera {camera_name} not registered for inventory tracking")
            return
        
        if zone_id not in self.inventory_states[camera_name]:
            logger.warning(f"Zone {zone_id} not registered for camera {camera_name}")
            return
        
        # Extract zone image
        zone_metadata = self.inventory_states[camera_name][zone_id]["metadata"]
        zone_image = self._extract_zone_image(frame, zone_metadata)
        
        # Update state
        self.inventory_states[camera_name][zone_id]["image"] = zone_image
        self.inventory_states[camera_name][zone_id]["timestamp"] = timestamp
        
        # Save image for reference
        zone_name = zone_metadata["name"]
        dt = datetime.fromtimestamp(timestamp)
        timestamp_str = dt.strftime("%Y%m%d_%H%M%S")
        image_filename = f"{camera_name}_{zone_name}_{timestamp_str}_state.jpg"
        image_path = os.path.join(self.inventory_states_dir, image_filename)
        cv2.imwrite(image_path, zone_image)
        
        logger.debug(f"Updated inventory state for {camera_name}, zone {zone_id}")
    
    def start_inventory_access(self, camera_name: str, frame: np.ndarray, 
                               zone_id: str, timestamp: float = None) -> Dict[str, Any]:
        """
        Start tracking an inventory access event.
        
        Args:
            camera_name: Name of the camera
            frame: Current camera frame
            zone_id: ID of the inventory zone being accessed
            timestamp: Current timestamp (default: current time)
            
        Returns:
            Event metadata
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Extract zone image
        if camera_name in self.inventory_states and zone_id in self.inventory_states[camera_name]:
            zone_metadata = self.inventory_states[camera_name][zone_id]["metadata"]
            zone_image = self._extract_zone_image(frame, zone_metadata)
            
            # Create access event
            access_key = f"{camera_name}_{zone_id}"
            self.active_access_events[access_key] = {
                "start_time": timestamp,
                "zone_id": zone_id,
                "start_image": zone_image.copy(),
                "zone_metadata": zone_metadata
            }
            
            logger.info(f"Started inventory access tracking: {camera_name}, zone {zone_id}")
            
            return {
                "event_type": self.EVENT_INVENTORY_ACCESS,
                "camera": camera_name,
                "zone_id": zone_id,
                "zone_name": zone_metadata["name"],
                "timestamp": timestamp,
                "status": "started"
            }
        else:
            logger.warning(f"Cannot start inventory access - invalid camera/zone: {camera_name}/{zone_id}")
            return {
                "event_type": self.EVENT_INVENTORY_ACCESS,
                "status": "error",
                "error": "Invalid camera or zone"
            }
    
    def end_inventory_access(self, camera_name: str, frame: np.ndarray, 
                            zone_id: str, timestamp: float = None) -> Dict[str, Any]:
        """
        End tracking an inventory access event and analyze changes.
        
        Args:
            camera_name: Name of the camera
            frame: Current camera frame
            zone_id: ID of the inventory zone being accessed
            timestamp: Current timestamp (default: current time)
            
        Returns:
            Dictionary with change detection results
        """
        if timestamp is None:
            timestamp = time.time()
        
        access_key = f"{camera_name}_{zone_id}"
        if access_key not in self.active_access_events:
            logger.warning(f"No active inventory access for {camera_name}, zone {zone_id}")
            return {
                "event_type": self.EVENT_NO_CHANGE,
                "camera": camera_name,
                "zone_id": zone_id,
                "change_detected": False,
                "confidence": self.CONFIDENCE_NONE,
                "error": "No active access event"
            }
        
        # Get access event data
        access_event = self.active_access_events[access_key]
        start_time = access_event["start_time"]
        zone_metadata = access_event["zone_metadata"]
        start_image = access_event["start_image"]
        
        # Get current zone image
        end_image = self._extract_zone_image(frame, zone_metadata)
        
        # Calculate duration
        duration = timestamp - start_time
        
        # Analyze change
        change_result = self._analyze_inventory_change(
            start_image, end_image, duration, camera_name, zone_id
        )
        
        # Remove from active events
        del self.active_access_events[access_key]
        
        # Save the before/after images
        dt = datetime.fromtimestamp(timestamp)
        timestamp_str = dt.strftime("%Y%m%d_%H%M%S")
        zone_name = zone_metadata["name"]
        
        # Save before image
        before_filename = f"{camera_name}_{zone_name}_{timestamp_str}_before.jpg"
        before_path = os.path.join(self.inventory_states_dir, before_filename)
        cv2.imwrite(before_path, start_image)
        
        # Save after image
        after_filename = f"{camera_name}_{zone_name}_{timestamp_str}_after.jpg"
        after_path = os.path.join(self.inventory_states_dir, after_filename)
        cv2.imwrite(after_path, end_image)
        
        # Save difference image if significant change
        if change_result["change_detected"]:
            diff_image = change_result.get("diff_image")
            if diff_image is not None:
                diff_filename = f"{camera_name}_{zone_name}_{timestamp_str}_diff.jpg"
                diff_path = os.path.join(self.inventory_states_dir, diff_filename)
                cv2.imwrite(diff_path, diff_image)
        
        # Add file paths to result
        change_result.update({
            "before_image_path": before_path,
            "after_image_path": after_path,
            "camera": camera_name,
            "zone_id": zone_id,
            "zone_name": zone_name,
            "start_time": start_time,
            "end_time": timestamp,
            "duration": duration,
            # Additional fields for backend rule processing
            "sync_status": "PENDING",
            "van_id": detection_result.get("van_id", "VAN001"),  # Include van_id for fleet identification
            "backend_rules_data": {
                "access_duration": duration,
                "change_percentage": change_result.get("change_percentage", 0),
                "change_pattern": self._categorize_change_pattern(before_image, after_image),
                "item_count_estimate": self._estimate_item_count_change(change_result),
                "concealment_indicators": self._detect_concealment_indicators(change_result)
            }
        })
        
        # Update inventory state with the new image
        self.update_inventory_state(camera_name, frame, zone_id, timestamp)
        
        logger.info(f"Ended inventory access: {camera_name}, zone {zone_id}, " +
                   f"{'change detected' if change_result['change_detected'] else 'no change'}")
        
        return change_result
    
    def _analyze_inventory_change(self, before_image: np.ndarray, after_image: np.ndarray, 
                                 duration: float, camera_name: str, zone_id: str) -> Dict[str, Any]:
        """
        Analyze changes between before and after images of inventory zone.
        
        Args:
            before_image: Image of zone before access
            after_image: Image of zone after access
            duration: Duration of access in seconds
            camera_name: Camera name
            zone_id: Zone ID
            
        Returns:
            Dictionary with change detection results
        """
        # Ensure images are the same size
        if before_image.shape != after_image.shape:
            logger.warning("Before/after images have different dimensions - resizing")
            after_image = cv2.resize(after_image, (before_image.shape[1], before_image.shape[0]))
        
        # Convert to grayscale for analysis
        before_gray = cv2.cvtColor(before_image, cv2.COLOR_BGR2GRAY)
        after_gray = cv2.cvtColor(after_image, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(before_gray, after_gray)
        
        # Apply threshold to highlight significant changes
        _, thresholded = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
        thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
        
        # Find contours of changed regions
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        significant_contours = [c for c in contours if cv2.contourArea(c) > self.min_change_area]
        
        # Calculate percentage of changed pixels
        total_pixels = before_gray.size
        changed_pixels = np.count_nonzero(thresholded)
        change_percentage = changed_pixels / total_pixels
        
        # Create visualization of differences
        diff_visualization = after_image.copy()
        
        # Identify if change is significant
        change_detected = change_percentage > self.change_threshold and len(significant_contours) > 0
        
        # Determine event type and confidence
        if not change_detected:
            event_type = self.EVENT_NO_CHANGE
            confidence = self.CONFIDENCE_LOW
        else:
            # Calculate mean brightness in before and after images
            before_brightness = np.mean(before_gray)
            after_brightness = np.mean(after_gray)
            
            # If after is darker than before, likely an item was removed
            if after_brightness > before_brightness:
                event_type = self.EVENT_ITEM_REMOVAL
            else:
                event_type = self.EVENT_ITEM_ADDITION
                
            # Draw contours on visualization
            cv2.drawContours(diff_visualization, significant_contours, -1, (0, 0, 255), 2)
            
            # Set confidence based on change percentage and duration
            if change_percentage > 0.3:
                confidence = self.CONFIDENCE_HIGH
            elif change_percentage > 0.2:
                confidence = self.CONFIDENCE_MEDIUM
            else:
                confidence = self.CONFIDENCE_LOW
                
            # Higher confidence for longer interactions
            if duration > 5.0 and confidence < self.CONFIDENCE_HIGH:
                confidence += 1
        
        # Prepare result
        result = {
            "event_type": event_type,
            "change_detected": change_detected,
            "confidence": confidence,
            "change_percentage": change_percentage,
            "changed_pixels": changed_pixels,
            "total_pixels": total_pixels,
            "contour_count": len(significant_contours),
            "diff_image": diff_visualization if change_detected else None,
            "analysis": {
                "significant_contours": len(significant_contours),
                "change_threshold": self.change_threshold,
                "min_change_area": self.min_change_area
            }
        }
        
        return result
    
    def _categorize_change_pattern(self, before_image: np.ndarray, after_image: np.ndarray) -> str:
        """
        Categorize the pattern of change between before and after images.
        
        Args:
            before_image: Image before inventory access
            after_image: Image after inventory access
            
        Returns:
            String describing the change pattern
        """
        # Convert to grayscale for analysis
        before_gray = cv2.cvtColor(before_image, cv2.COLOR_BGR2GRAY) if len(before_image.shape) > 2 else before_image
        after_gray = cv2.cvtColor(after_image, cv2.COLOR_BGR2GRAY) if len(after_image.shape) > 2 else after_image
        
        # Calculate absolute difference
        diff = cv2.absdiff(before_gray, after_gray)
        
        # Apply threshold
        _, thresholded = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        significant_contours = [c for c in contours if cv2.contourArea(c) > self.min_change_area]
        
        # Analyze contour locations and patterns
        if len(significant_contours) == 0:
            return "NO_CHANGE"
            
        # Determine if changes are concentrated in one area or distributed
        height, width = before_gray.shape
        quadrants_with_changes = set()
        
        for c in significant_contours:
            # Get contour center
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Determine quadrant (1, 2, 3, or 4)
                quadrant = 1
                if cx <= width/2 and cy <= height/2:
                    quadrant = 1
                elif cx > width/2 and cy <= height/2:
                    quadrant = 2
                elif cx <= width/2 and cy > height/2:
                    quadrant = 3
                else:
                    quadrant = 4
                    
                quadrants_with_changes.add(quadrant)
        
        # Determine pattern based on quadrant distribution
        if len(quadrants_with_changes) == 1:
            return "SINGLE_AREA_CHANGE"
        elif len(quadrants_with_changes) == 2:
            return "DUAL_AREA_CHANGE"
        else:
            return "DISTRIBUTED_CHANGE"
    
    def _estimate_item_count_change(self, change_result: Dict[str, Any]) -> int:
        """
        Provide a rough estimate of how many items were added/removed.
        This is a simplistic approximation based on contour count and change percentage.
        
        Args:
            change_result: Result from change detection
            
        Returns:
            Estimated number of items changed (positive for additions, negative for removals)
        """
        if not change_result.get("change_detected", False):
            return 0
            
        contour_count = change_result.get("contour_count", 0)
        change_percentage = change_result.get("change_percentage", 0)
        event_type = change_result.get("event_type", self.EVENT_NO_CHANGE)
        
        # Basic heuristic: divide contour count by 2 for a conservative estimate
        # but ensure at least 1 item if change is detected
        estimated_count = max(1, contour_count // 2)
        
        # For very small changes, reduce estimate
        if change_percentage < 0.1:
            estimated_count = 1
        
        # Negative count for removals, positive for additions
        if event_type == self.EVENT_ITEM_REMOVAL:
            return -estimated_count
        elif event_type == self.EVENT_ITEM_ADDITION:
            return estimated_count
        else:
            return 0
    
    def _detect_concealment_indicators(self, change_result: Dict[str, Any]) -> List[str]:
        """
        Detect potential indicators of concealment behavior.
        
        Args:
            change_result: Result from change detection
            
        Returns:
            List of concealment indicators detected
        """
        indicators = []
        
        # Check for specific patterns that might indicate concealment
        if not change_result.get("change_detected", False):
            # No change detected but significant access duration could indicate concealment
            duration = change_result.get("duration", 0)
            if duration > 10.0:  # More than 10 seconds with no visible change
                indicators.append("LONG_ACCESS_NO_CHANGE")
        else:
            # Check for minimal change percentage that might indicate careful concealment
            change_percentage = change_result.get("change_percentage", 0)
            if change_percentage < 0.05 and change_result.get("event_type") == self.EVENT_ITEM_REMOVAL:
                indicators.append("MINIMAL_VISIBLE_CHANGE")
        
        return indicators
        
    def visualize(self, frame: np.ndarray, camera_name: str, results: Dict[str, Any] = None) -> np.ndarray:
        """
        Visualize inventory zones and change detection results.
        
        Args:
            frame: Current camera frame
            camera_name: Name of the camera
            results: Optional change detection results to visualize
            
        Returns:
            Frame with visualizations
        """
        vis_frame = frame.copy()
        
        # Draw all inventory zones for this camera
        if camera_name in self.inventory_states:
            for zone_id, zone_data in self.inventory_states[camera_name].items():
                zone = zone_data["metadata"]
                
                # Default zone color is green
                color = (0, 255, 0)
                
                # If we have active access, make zone red
                access_key = f"{camera_name}_{zone_id}"
                if access_key in self.active_access_events:
                    color = (0, 0, 255)  # Red for active access
                
                # Draw zone rectangle
                cv2.rectangle(
                    vis_frame, 
                    (zone["x"], zone["y"]), 
                    (zone["x"] + zone["width"], zone["y"] + zone["height"]), 
                    color, 
                    2
                )
                
                # Add zone name
                cv2.putText(
                    vis_frame,
                    zone["name"],
                    (zone["x"], zone["y"] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )
        
        # If we have change detection results, visualize them
        if results and results.get("change_detected", False):
            # Add change information text
            event_type = results.get("event_type", "UNKNOWN")
            confidence = results.get("confidence", 0)
            change_pct = results.get("change_percentage", 0) * 100
            
            # Position text at top of frame
            text_lines = [
                f"Change: {event_type}",
                f"Confidence: {confidence}",
                f"Changed: {change_pct:.1f}%"
            ]
            
            for i, text in enumerate(text_lines):
                cv2.putText(
                    vis_frame,
                    text,
                    (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),  # Red
                    2
                )
        
        return vis_frame
