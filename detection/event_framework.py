"""
SmartVan Monitor - Enhanced Event Framework
------------------------------------------
This module implements an enhanced event classification framework that combines
motion detection, object detection, and inventory change detection.

Key features:
- Unified event data model for all detection types
- Standardized confidence scoring across detection methods
- Event relationship tracking and correlation
- Support for complex multi-step events
"""

import time
import datetime
import logging
import json
import os
import uuid
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("event_framework")

class EnhancedEventClassifier:
    """Enhanced event classification framework for SmartVan Monitor."""
    
    # Event categories
    CATEGORY_MOTION = "MOTION"
    CATEGORY_HUMAN = "HUMAN"
    CATEGORY_INVENTORY = "INVENTORY"
    CATEGORY_SUSPICIOUS = "SUSPICIOUS"
    CATEGORY_SYSTEM = "SYSTEM"
    
    # Event types - Motion
    EVENT_MOTION = "MOTION"
    EVENT_VIBRATION = "VIBRATION"
    
    # Event types - Human
    EVENT_HUMAN_PRESENT = "HUMAN_PRESENT"
    EVENT_HUMAN_ENTERING = "HUMAN_ENTERING"
    EVENT_HUMAN_EXITING = "HUMAN_EXITING"
    
    # Event types - Inventory
    EVENT_INVENTORY_ACCESS = "INVENTORY_ACCESS"
    EVENT_ITEM_REMOVAL = "ITEM_REMOVAL"
    EVENT_ITEM_ADDITION = "ITEM_ADDITION"
    EVENT_NO_INVENTORY_CHANGE = "NO_INVENTORY_CHANGE"
    
    # Event types - Suspicious
    EVENT_HIGH_FREQUENCY = "HIGH_FREQUENCY"
    EVENT_AFTER_HOURS = "AFTER_HOURS"
    EVENT_SUSPICIOUS_PATTERN = "SUSPICIOUS_PATTERN"
    
    # Confidence levels
    CONFIDENCE_HIGH = 3
    CONFIDENCE_MEDIUM = 2
    CONFIDENCE_LOW = 1
    CONFIDENCE_NONE = 0
    
    def __init__(self, 
                output_dir: str = "output",
                business_hours_start: int = 7,  # 7 AM
                business_hours_end: int = 19,   # 7 PM
                high_frequency_threshold: int = 3,  # 3 accesses
                high_frequency_window: int = 900,  # 15 minutes (in seconds)
                event_memory_window: int = 3600,   # 1 hour (in seconds)
                min_alert_interval: float = 30.0,  # Min seconds between alerts for same camera
                van_id: str = "VAN001"):  # Van identifier for fleet tracking
        """
        Initialize the enhanced event classifier.
        
        Args:
            output_dir: Directory to store event data
            business_hours_start: Start of business hours (24-hour format)
            business_hours_end: End of business hours (24-hour format)
            high_frequency_threshold: Number of events that trigger high frequency alert
            high_frequency_window: Time window for high frequency events (seconds)
            event_memory_window: How long to remember events for pattern analysis (seconds)
            min_alert_interval: Minimum seconds between alerts for same camera
            van_id: Unique identifier for this van (for fleet tracking)
        """
        self.output_dir = Path(output_dir)
        self.events_dir = self.output_dir / "events"
        self.events_dir.mkdir(exist_ok=True)
        
        self.business_hours_start = business_hours_start
        self.business_hours_end = business_hours_end
        self.high_frequency_threshold = high_frequency_threshold
        self.high_frequency_window = high_frequency_window
        self.event_memory_window = event_memory_window
        self.min_alert_interval = min_alert_interval
        self.van_id = van_id  # Store van_id for fleet tracking
        
        # Event storage
        self.events = {}  # camera_name -> list of events
        self.current_sessions = {}  # camera_name -> session data
        self.last_alert_time = {}  # camera_name -> time of last alert
        
        # Related event tracking
        self.related_events = {}  # event_id -> list of related event IDs
        
        logger.info(f"Enhanced event framework initialized for van: {self.van_id}")
    
    def _is_business_hours(self, timestamp: float = None) -> bool:
        """
        Check if the given timestamp is within business hours.
        
        Args:
            timestamp: Unix timestamp to check (default: current time)
            
        Returns:
            True if within business hours, False otherwise
        """
        if timestamp is None:
            timestamp = time.time()
            
        dt = datetime.datetime.fromtimestamp(timestamp)
        hour = dt.hour
        
        return self.business_hours_start <= hour < self.business_hours_end
    
    def _clean_old_events(self, camera_name: str, current_time: float) -> None:
        """
        Remove events older than the memory window.
        
        Args:
            camera_name: Name of the camera to clean events for
            current_time: Current timestamp
        """
        if camera_name not in self.events:
            return
            
        # Filter out old events
        self.events[camera_name] = [
            e for e in self.events[camera_name] 
            if current_time - e["timestamp"] < self.event_memory_window
        ]
    
    def _check_high_frequency(self, camera_name: str, current_time: float) -> Tuple[bool, List[str]]:
        """
        Check if there are too many events in a short time period.
        
        Args:
            camera_name: Name of the camera to check
            current_time: Current timestamp
            
        Returns:
            Tuple of (is_high_frequency, reasons)
        """
        if camera_name not in self.events:
            return False, []
            
        # Count events within the high frequency window
        recent_events = [
            e for e in self.events[camera_name] 
            if current_time - e["timestamp"] < self.high_frequency_window
        ]
        
        event_count = len(recent_events)
        
        # Check if above threshold
        is_high_frequency = event_count >= self.high_frequency_threshold
        reasons = []
        
        if is_high_frequency:
            window_minutes = self.high_frequency_window / 60
            reasons.append(f"High frequency access: {event_count} events in {window_minutes:.1f} minutes")
            
        return is_high_frequency, reasons
    
    def _generate_event_id(self) -> str:
        """
        Generate a unique event ID.
        
        Returns:
            Unique event ID string
        """
        return str(uuid.uuid4())
    
    def _get_session_id(self, camera_name: str, current_time: float) -> str:
        """
        Get or create a session ID for continuous activity.
        
        Args:
            camera_name: Camera name
            current_time: Current timestamp
            
        Returns:
            Session ID string
        """
        # Check if we have an active session
        if camera_name in self.current_sessions:
            session = self.current_sessions[camera_name]
            # If the session is still active (less than 30 seconds since last update)
            if current_time - session["last_update"] < 30:
                # Update the last update time
                session["last_update"] = current_time
                return session["id"]
                
        # Create a new session
        session_id = f"s_{self._generate_event_id()}"
        self.current_sessions[camera_name] = {
            "id": session_id,
            "start_time": current_time,
            "last_update": current_time
        }
        
        return session_id
    
    def _get_previous_access_time(self, camera_name: str, current_time: float) -> float:
        """
        Get the timestamp of the previous access event for this camera.
        
        Args:
            camera_name: Camera name
            current_time: Current timestamp
            
        Returns:
            Timestamp of previous access or 0 if none
        """
        if camera_name not in self.events:
            return 0
            
        # Filter for relevant events (within memory window)
        relevant_events = [
            e for e in self.events[camera_name] 
            if current_time - e["timestamp"] < self.event_memory_window
        ]
        
        if not relevant_events:
            return 0
            
        # Sort by timestamp (newest first)
        relevant_events.sort(key=lambda e: e["timestamp"], reverse=True)
        
        # Return the timestamp of the most recent event
        return relevant_events[0]["timestamp"]
    
    def _get_time_since_last_access(self, camera_name: str, current_time: float) -> float:
        """
        Calculate time since the last access event for this camera.
        
        Args:
            camera_name: Camera name
            current_time: Current timestamp
            
        Returns:
            Seconds since last access or -1 if no previous access
        """
        previous_time = self._get_previous_access_time(camera_name, current_time)
        
        if previous_time == 0:
            return -1
            
        return current_time - previous_time
    
    def _get_camera_sequence(self, camera_name: str, current_time: float, count: int) -> List[str]:
        """
        Get the sequence of cameras accessed most recently across all cameras.
        Useful for detecting patterns like moving between cameras in a suspicious way.
        
        Args:
            camera_name: Current camera name
            current_time: Current timestamp
            count: Number of cameras in the sequence
            
        Returns:
            List of camera names in order of access (most recent first)
        """
        # Collect all events within memory window
        all_events = []
        for cam, events in self.events.items():
            for event in events:
                if current_time - event["timestamp"] < self.event_memory_window:
                    all_events.append(event)
        
        # Sort by timestamp (newest first)
        all_events.sort(key=lambda e: e["timestamp"], reverse=True)
        
        # Extract camera sequence
        sequence = []
        for event in all_events:
            if event["camera"] not in sequence:
                sequence.append(event["camera"])
                if len(sequence) >= count:
                    break
        
        return sequence
    
    def create_event(self, 
                    camera_name: str, 
                    event_category: str, 
                    event_type: str, 
                    detection_data: Dict[str, Any], 
                    confidence: int = None,
                    timestamp: float = None,
                    related_event_id: str = None,
                    pos_transaction_id: str = None,
                    metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a new event with the enhanced event framework.
        
        Args:
            camera_name: Name of the camera that detected the event
            event_category: Category of the event (MOTION, HUMAN, INVENTORY, etc.)
            event_type: Type of the event within the category
            detection_data: Raw detection data from the detector
            confidence: Optional confidence level (if None, calculated from detection data)
            timestamp: Event timestamp (default: current time)
            related_event_id: Optional ID of a related event
            pos_transaction_id: Optional ID of a related POS transaction
            metadata: Optional additional metadata to include in the event
            
        Returns:
            Dictionary with event data
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Clean up old events
        self._clean_old_events(camera_name, timestamp)
        
        # Check for high frequency events
        is_high_frequency, frequency_reasons = self._check_high_frequency(camera_name, timestamp)
        
        # Generate event ID
        event_id = self._generate_event_id()
        
        # Get session ID
        session_id = self._get_session_id(camera_name, timestamp)
        
        # Determine if this is after hours
        is_after_hours = not self._is_business_hours(timestamp)
        
        # Calculate confidence if not provided
        if confidence is None:
            confidence = self._calculate_confidence(event_category, event_type, detection_data)
        
        # Apply rate limiting for alerts
        allow_new_alert = True
        if camera_name in self.last_alert_time:
            time_since_last_alert = timestamp - self.last_alert_time[camera_name]
            allow_new_alert = time_since_last_alert >= self.min_alert_interval
        
        # Add van_id to detection data if not already present
        if "van_id" not in detection_data:
            detection_data["van_id"] = self.van_id
            
        # Create event structure
        event = {
            "event_id": event_id,
            "camera": camera_name,
            "timestamp": timestamp,
            "datetime": datetime.datetime.fromtimestamp(timestamp).isoformat(),
            "session_id": session_id,
            "van_id": detection_data["van_id"],  # Always include van_id in the event
            "category": event_category,
            "type": event_type,
            "confidence": confidence,
            "is_high_frequency": is_high_frequency,
            "is_after_hours": is_after_hours,
            "rate_limited": not allow_new_alert,
            "detection_data": detection_data,
            "sync_status": "PENDING",
            "sync_retry_count": 0,
            "pos_transaction_id": pos_transaction_id,
            "backend_processed": False,
            "pattern_data": {
                "session_duration": 0,  # Will be updated when session ends
                "access_count": len([e for e in self.events.get(camera_name, []) if e.get("session_id") == session_id]),
                "previous_access_time": self._get_previous_access_time(camera_name, timestamp),
                "time_since_last_access": self._get_time_since_last_access(camera_name, timestamp),
                "camera_activity_sequence": self._get_camera_sequence(camera_name, timestamp, 5)  # Last 5 cameras accessed
            }
        }
        
        # Add custom metadata if provided
        if metadata:
            event.update(metadata)
        
        # Add alert reasons
        reasons = []
        
        # Add high frequency reasons
        if is_high_frequency:
            reasons.extend(frequency_reasons)
        
        # Add after hours reason
        if is_after_hours:
            reasons.append("Activity outside business hours")
        
        # Add human detection reason if applicable
        if event_category == self.CATEGORY_HUMAN:
            human_confidence = detection_data.get("human_confidence", 0)
            reasons.append(f"Human detected with {human_confidence:.2f} confidence")
        
        # Add inventory reasons if applicable
        if event_category == self.CATEGORY_INVENTORY:
            if event_type == self.EVENT_ITEM_REMOVAL:
                change_pct = detection_data.get("change_percentage", 0) * 100
                reasons.append(f"Item removal detected ({change_pct:.1f}% change)")
            elif event_type == self.EVENT_ITEM_ADDITION:
                change_pct = detection_data.get("change_percentage", 0) * 100
                reasons.append(f"Item addition detected ({change_pct:.1f}% change)")
        
        event["reasons"] = reasons
        
        # If this is a significant event, update the last alert time
        if confidence >= self.CONFIDENCE_MEDIUM and allow_new_alert:
            self.last_alert_time[camera_name] = timestamp
        
        # Add to event list for this camera
        if camera_name not in self.events:
            self.events[camera_name] = []
        self.events[camera_name].append(event)
        
        # Handle related events
        if related_event_id:
            if related_event_id not in self.related_events:
                self.related_events[related_event_id] = []
            self.related_events[related_event_id].append(event_id)
            
            # Add to this event for reference
            event["related_to"] = related_event_id
        
        # Save event to disk for persistence
        self._save_event(event)
        
        # Log the event
        log_message = f"Event: {event_category}/{event_type} from {camera_name} (confidence: {confidence})"
        if reasons:
            log_message += f" ({', '.join(reasons)})"
            
        if confidence == self.CONFIDENCE_HIGH:
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        return event
    
    def _process_motion_event(self, camera_name: str, motion_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process motion detection data to determine if it qualifies as an event.
        Enhanced with improved vibration filtering.
        
        Args:
            camera_name: Name of the camera that captured the motion
            motion_data: Motion detection data
            
        Returns:
            Event data dictionary or None if no event should be created
        """
        if not motion_data.get("motion_detected", False):
            return None
        
        # Enhanced vibration analysis data
        intensity = motion_data.get("intensity", 0)
        is_vibration = motion_data.get("is_vibration", False)
        vibration_confidence = motion_data.get("vibration_confidence", 0.0)
        vibration_type = motion_data.get("vibration_type", "UNKNOWN")
        likely_human = motion_data.get("likely_human", False)
        human_confidence = motion_data.get("human_confidence", 0.0)
        motion_confidence = motion_data.get("confidence", 0)
        
        # Early rejection of very low intensity events
        if intensity < self.motion_intensity_threshold * 0.5:
            self.logger.debug(f"Rejecting low intensity motion: {intensity}")
            return None
            
        # Enhanced vibration filtering: reject high-confidence vibrations
        if is_vibration and vibration_confidence > 0.7 and intensity < self.motion_intensity_threshold * 1.5:
            self.logger.debug(f"Rejecting high-confidence vibration: {vibration_type}, conf={vibration_confidence:.2f}")
            return None
        
        # Get the current timestamp
        timestamp = time.time()
        
        # Check if we should throttle this event
        if not self._check_rate_limit(camera_name, "motion", timestamp):
            return None
            
        # Start with the confidence from the motion detector
        # These are directly mapped from CONFIDENCE_NONE (0) to CONFIDENCE_HIGH (3)
        confidence = motion_confidence
        
        # Further adjust confidence based on our analysis
        # Upgrade confidence for high human confidence detections
        if likely_human and human_confidence > 0.8:
            confidence = max(confidence, self.CONFIDENCE_HIGH)
        elif likely_human and human_confidence > 0.6:
            confidence = max(confidence, self.CONFIDENCE_MEDIUM)
        
        # Downgrade confidence for vibrations that weren't rejected
        if is_vibration and vibration_confidence > 0.4:
            confidence = max(self.CONFIDENCE_NONE, confidence - 1)
        
        # Determine event type based on intensity and patterns
        if intensity > self.motion_intensity_threshold * 1.5 and likely_human:
            event_type = "human_activity"
        elif intensity > self.motion_intensity_threshold:
            event_type = "significant_motion"
        else:
            event_type = "minor_motion"
            
        # Prepare the enhanced event data with all vibration analysis information
        event_data = {
            "camera_name": camera_name,
            "timestamp": timestamp,
            "event_type": event_type,
            "confidence": confidence,
            "intensity": intensity,
            "duration": motion_data.get("duration", 0),
            "is_vibration": is_vibration,
            "vibration_confidence": vibration_confidence,
            "vibration_type": vibration_type,
            "likely_human": likely_human,
            "human_confidence": human_confidence,
            "bounding_boxes": motion_data.get("bounding_boxes", []),
            "van_id": self.van_id  # Include van_id in event data
        }
        
        return event_data
    
    def _calculate_confidence(self, category: str, event_type: str, 
                            detection_data: Dict[str, Any]) -> int:
        """
        Calculate confidence level based on event data.
        
        Args:
            category: Event category
            event_type: Event type
            detection_data: Detection data
            
        Returns:
            Confidence level (0-3)
        """
        # Default to low confidence
        confidence = self.CONFIDENCE_LOW
        
        if category == self.CATEGORY_MOTION:
            # Get motion intensity
            intensity = detection_data.get("intensity", 0)
            is_vibration = detection_data.get("is_vibration", False)
            vibration_confidence = detection_data.get("vibration_confidence", 0.0)
            vibration_type = detection_data.get("vibration_type", "UNKNOWN")
            likely_human = detection_data.get("likely_human", False)
            human_confidence = detection_data.get("human_confidence", 0.0)
            motion_confidence = detection_data.get("confidence", 0)
            
            # Vibrations are lower confidence
            if is_vibration and vibration_confidence > 0.4:
                confidence = self.CONFIDENCE_NONE
            else:
                # Adjust confidence based on intensity and human confidence
                if intensity > 60:
                    confidence = self.CONFIDENCE_MEDIUM
                if intensity > 80:
                    confidence = self.CONFIDENCE_HIGH
                    
                if likely_human and human_confidence > 0.8:
                    confidence = max(confidence, self.CONFIDENCE_HIGH)
                elif likely_human and human_confidence > 0.6:
                    confidence = max(confidence, self.CONFIDENCE_MEDIUM)
        
        elif category == self.CATEGORY_HUMAN:
            # Get human detection confidence
            human_confidence = detection_data.get("human_confidence", 0)
            
            # Map human confidence to our confidence scale
            if human_confidence > 0.7:
                confidence = self.CONFIDENCE_HIGH
            elif human_confidence > 0.5:
                confidence = self.CONFIDENCE_MEDIUM
            else:
                confidence = self.CONFIDENCE_LOW
                
        elif category == self.CATEGORY_INVENTORY:
            # Use the inventory detector's confidence
            inventory_confidence = detection_data.get("confidence", self.CONFIDENCE_LOW)
            change_percentage = detection_data.get("change_percentage", 0)
            
            # Trust the inventory detector's confidence level
            confidence = inventory_confidence
            
            # But upgrade if change percentage is very high
            if change_percentage > 0.3 and confidence < self.CONFIDENCE_HIGH:
                confidence = self.CONFIDENCE_HIGH
        
        elif category == self.CATEGORY_SUSPICIOUS:
            # For suspicious events, default to medium confidence
            confidence = self.CONFIDENCE_MEDIUM
            
            # Higher confidence for after hours events
            if event_type == self.EVENT_AFTER_HOURS:
                confidence = self.CONFIDENCE_HIGH
        
        return confidence
    
    def _ensure_json_serializable(self, obj):
        """
        Ensure all values in dict/list are JSON serializable (convert numpy types to Python types)
        """
        if isinstance(obj, dict):
            return {k: self._ensure_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._ensure_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._ensure_json_serializable(item) for item in obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
            # Explicitly handle numpy float types
            return float(obj)
        elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
            # Explicitly handle numpy int types
            return int(obj)
        elif isinstance(obj, np.bool_):
            # Handle numpy boolean type
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            # For image data in results, convert small arrays to lists
            return obj.tolist() if obj.size < 1000 else f"<ndarray shape={obj.shape}>"
        elif str(type(obj)).startswith("<class 'numpy"):
            # Catch any other numpy types
            try:
                return obj.item()
            except:
                return str(obj)
        else:
            return obj
    
    def _save_event(self, event: Dict[str, Any]) -> None:
        """
        Save event to disk for persistence.
        
        Args:
            event: Event data dictionary
        """
        # Create a filename based on event data
        timestamp_str = datetime.datetime.fromtimestamp(event["timestamp"]).strftime("%Y%m%d_%H%M%S")
        camera_name = event["camera"]
        event_type = event["type"]
        event_id = event["event_id"]
        
        filename = f"{camera_name}_{timestamp_str}_{event_type}_{event_id}.json"
        filepath = self.events_dir / filename
        
        # Ensure all values are JSON serializable (convert numpy types)
        serializable_event = self._ensure_json_serializable(event)
        
        # Save to disk
        with open(filepath, 'w') as f:
            json.dump(serializable_event, f, indent=4)
    
    def get_recent_events(self, 
                         camera_name: str = None,
                         category: str = None, 
                         event_type: str = None,
                         max_events: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent events, optionally filtered by camera, category, and type.
        
        Args:
            camera_name: Optional camera name to filter by
            category: Optional event category to filter by
            event_type: Optional event type to filter by
            max_events: Maximum number of events to return
            
        Returns:
            List of events
        """
        all_events = []
        
        # Get events from specified camera or all cameras
        if camera_name:
            cameras_to_check = [camera_name] if camera_name in self.events else []
        else:
            cameras_to_check = list(self.events.keys())
            
        # Collect events from each camera
        for cam in cameras_to_check:
            all_events.extend(self.events[cam])
        
        # Filter by category if specified
        if category:
            all_events = [e for e in all_events if e["category"] == category]
            
        # Filter by event type if specified
        if event_type:
            all_events = [e for e in all_events if e["type"] == event_type]
            
        # Sort by timestamp (newest first)
        all_events.sort(key=lambda e: e["timestamp"], reverse=True)
        
        # Return limited number of events
        return all_events[:max_events]
    
    def get_related_events(self, event_id: str) -> List[Dict[str, Any]]:
        """
        Get events related to the specified event.
        
        Args:
            event_id: ID of the event to get related events for
            
        Returns:
            List of related events
        """
        if event_id not in self.related_events:
            return []
        
        related_ids = self.related_events[event_id]
        related_events = []
        
        # Search for related events in all cameras
        for camera_name, events in self.events.items():
            for event in events:
                if event["event_id"] in related_ids:
                    related_events.append(event)
        
        # Sort by timestamp
        related_events.sort(key=lambda e: e["timestamp"])
        
        return related_events
