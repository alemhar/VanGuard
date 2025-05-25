"""
SmartVan Monitor - Event Classification Module
--------------------------------------------
This module implements event classification for the SmartVan Monitor system.

Key features:
- Classification of motion events into different categories
- Time-based access patterns detection
- High-frequency access detection
- After-hours detection
"""

import time
import datetime
import logging
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("event_classification")

class EventClassifier:
    """Class for classifying motion events into different categories."""
    
    # Event classification types
    EVENT_NORMAL = "NORMAL"           # Normal access during business hours
    EVENT_HIGH_FREQUENCY = "HIGH_FREQUENCY"  # Multiple accesses in short period
    EVENT_AFTER_HOURS = "AFTER_HOURS"  # Access outside business hours
    EVENT_SUSPICIOUS = "SUSPICIOUS"    # Unusual pattern that may indicate theft
    EVENT_INVENTORY_ACCESS = "INVENTORY_ACCESS"  # Definite human inventory access detected
    EVENT_AFTER_HOURS_ACCESS = "AFTER_HOURS_ACCESS"  # Human inventory access outside business hours
    
    # Confidence levels
    CONFIDENCE_HIGH = 3
    CONFIDENCE_MEDIUM = 2
    CONFIDENCE_LOW = 1
    CONFIDENCE_NONE = 0
    
    def __init__(self, 
                business_hours_start: int = 7,  # 7 AM
                business_hours_end: int = 19,   # 7 PM
                high_frequency_threshold: int = 3,  # 3 accesses
                high_frequency_window: int = 900,  # 15 minutes (in seconds)
                event_memory_window: int = 3600):  # 1 hour (in seconds)
        """
        Initialize the event classifier with specified parameters.
        
        Args:
            business_hours_start: Start of business hours (24-hour format)
            business_hours_end: End of business hours (24-hour format)
            high_frequency_threshold: Number of events that trigger high frequency alert
            high_frequency_window: Time window for high frequency events (seconds)
            event_memory_window: How long to remember events for pattern analysis (seconds)
        """
        self.business_hours_start = business_hours_start
        self.business_hours_end = business_hours_end
        self.high_frequency_threshold = high_frequency_threshold
        self.high_frequency_window = high_frequency_window
        self.event_memory_window = event_memory_window
        
        # Event history for pattern detection
        self.event_history = {}  # camera_name -> list of event timestamps
        self.classified_events = {}  # camera_name -> list of classified events
        
        # Continuous motion tracking to prevent alert storms
        self.current_motion_start = {}  # camera_name -> start time of current continuous motion
        self.last_alert_time = {}  # camera_name -> time of last alert
        self.min_alert_interval = 30.0  # minimum seconds between alerts for same camera
        
        logger.info("Event classifier initialized")
    
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
            
        current_time = datetime.datetime.fromtimestamp(timestamp)
        current_hour = current_time.hour
        
        return self.business_hours_start <= current_hour < self.business_hours_end
    
    def _clean_old_events(self, camera_name: str, current_time: float) -> None:
        """
        Remove events older than the memory window.
        
        Args:
            camera_name: Name of the camera to clean events for
            current_time: Current timestamp
        """
        if camera_name in self.event_history:
            # Keep only events within the memory window
            self.event_history[camera_name] = [
                ts for ts in self.event_history[camera_name] 
                if current_time - ts <= self.event_memory_window
            ]
            
            # If list is too long, keep only the most recent 20 events to prevent memory issues
            if len(self.event_history[camera_name]) > 20:
                self.event_history[camera_name] = sorted(self.event_history[camera_name], reverse=True)[:20]
        
        if camera_name in self.classified_events:
            # Keep only classified events within the memory window
            self.classified_events[camera_name] = [
                event for event in self.classified_events[camera_name]
                if current_time - event['timestamp'] <= self.event_memory_window
            ]
            
            # If list is too long, keep only the most recent 10 events
            if len(self.classified_events[camera_name]) > 10:
                self.classified_events[camera_name] = sorted(self.classified_events[camera_name], 
                                                   key=lambda e: e['timestamp'], 
                                                   reverse=True)[:10]
    
    def _check_high_frequency(self, camera_name: str, current_time: float, 
                           detection_result: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check if there are too many events in a short time period.
        
        Args:
            camera_name: Name of the camera to check
            current_time: Current timestamp
            detection_result: Motion detection result data
            
        Returns:
            Tuple of (is_high_frequency, reasons)
        """
        if camera_name not in self.event_history:
            return False, []
        
        # Check if this is a vibration event
        is_vibration = detection_result.get("is_vibration", False)
        if is_vibration:
            return False, []  # Don't count vibrations as high-frequency events
            
        # Get flow uniformity if available (low uniformity = more likely human action)
        flow_uniformity = detection_result.get("flow_uniformity", 1.0)
        
        # Count only distinct event periods (not continuous detection)
        # Two events are considered distinct if they're separated by at least 2 seconds
        distinct_events = []
        sorted_events = sorted(self.event_history[camera_name])
        
        if sorted_events:
            # Add the first event
            distinct_events.append(sorted_events[0])
            
            # Add subsequent events only if they're separated enough
            for ts in sorted_events[1:]:
                if ts - distinct_events[-1] >= 2.0:  # 2-second gap means distinct event
                    distinct_events.append(ts)
        
        # Count recent distinct events
        recent_distinct_events = [
            ts for ts in distinct_events
            if current_time - ts <= self.high_frequency_window
        ]
        
        # More sophisticated high-frequency detection
        # 1. Need enough distinct events
        # 2. For uniform flow events (vehicle movement), need more events to trigger
        threshold_modifier = 1.0 if flow_uniformity > 0.5 else 2.0
        adjusted_threshold = self.high_frequency_threshold * threshold_modifier
        
        is_high_frequency = len(recent_distinct_events) >= adjusted_threshold
        reasons = []
        
        if is_high_frequency:
            reasons.append(f"Multiple distinct accesses ({len(recent_distinct_events)}) in short period")
            
        return is_high_frequency, reasons
    
    def classify_event(self, camera_name: str, detection_result: Dict[str, Any], 
                     timestamp: float = None) -> Dict[str, Any]:
        """
        Classify a motion detection event.
        
        Args:
            camera_name: Name of the camera that detected the event
            detection_result: Result from motion detector
            timestamp: Event timestamp (default: current time)
            
        Returns:
            Dictionary with classification results
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Initialize camera tracking structures if not exist
        if camera_name not in self.event_history:
            self.event_history[camera_name] = []
        if camera_name not in self.classified_events:
            self.classified_events[camera_name] = []
        if camera_name not in self.current_motion_start:
            self.current_motion_start[camera_name] = None
        if camera_name not in self.last_alert_time:
            self.last_alert_time[camera_name] = 0
        
        # Get motion detection info
        motion_detected = detection_result.get('motion_detected', False)
        motion_intensity = detection_result.get('intensity', 0)
        motion_confidence = detection_result.get('confidence', 0)
        motion_duration = detection_result.get('duration', 0)
        is_vibration = detection_result.get('is_vibration', False)
        likely_human = detection_result.get('likely_human', False)  # New human detection feature
        
        # Track continuous motion to avoid duplicate event recording
        # Only consider it a new event if motion stopped and started again
        if motion_detected and motion_confidence > 0:
            if self.current_motion_start[camera_name] is None:
                # New motion started
                self.current_motion_start[camera_name] = timestamp
                # This is a new distinct event, add to history
                self.event_history[camera_name].append(timestamp)
                
                # Clean old events after adding new one
                self._clean_old_events(camera_name, timestamp)
        elif not motion_detected and self.current_motion_start[camera_name] is not None:
            # Motion has stopped
            self.current_motion_start[camera_name] = None
        
        # Skip remaining classification if this is definitely just vibration and not human
        if is_vibration and motion_intensity < 30 and not likely_human:
            return {
                'camera': camera_name,
                'timestamp': timestamp,
                'datetime': datetime.datetime.fromtimestamp(timestamp).isoformat(),
                'event_type': self.EVENT_NORMAL,
                'confidence': self.CONFIDENCE_NONE,  # Zero confidence 
                'reasons': ["Filtered out vibration or small movement"],
                'motion_data': {
                    'intensity': motion_intensity,
                    'duration': motion_duration,
                    'motion_confidence': motion_confidence,
                    'is_vibration': True,
                    'likely_human': False
                }
            }
            
        # Check if we should rate-limit alerts
        time_since_last_alert = timestamp - self.last_alert_time.get(camera_name, 0)
        allow_new_alert = time_since_last_alert >= self.min_alert_interval
        
        # Default classification
        event_type = self.EVENT_NORMAL
        event_confidence = self.CONFIDENCE_LOW
        reasons = []
        
        # Apply rate-limiting for alerts
        # Only proceed with full classification if we're allowing a new alert
        # or if this is not a potentially high-alert situation
        # OR if this is likely human movement (which should always be classified)
        low_priority_event = (motion_intensity < 40 and motion_duration < 5.0 and not likely_human)
        
        # Always process likely human events regardless of rate limiting
        if allow_new_alert or low_priority_event or likely_human:
            # Check for human presence first - highest priority classification
            if likely_human:
                # This is likely a human moving in the frame - classify as inventory access
                event_type = self.EVENT_INVENTORY_ACCESS
                reasons.append("Human movement detected")
                
                # Increase confidence level based on motion characteristics
                # For entering/exiting frame, we want higher confidence even with moderate intensity
                if motion_intensity > 30 or motion_duration > 1.5:
                    event_confidence = max(event_confidence, self.CONFIDENCE_MEDIUM)
                    
                # Higher confidence for more significant movement
                if motion_intensity > 40 or motion_duration > 2.5:
                    event_confidence = self.CONFIDENCE_HIGH
                    reasons.append(f"Significant human movement (intensity: {motion_intensity})")
                    
                # Force inventory access detection for humans
                # Even at low intensity, we want to detect humans entering/exiting frame
                if motion_confidence > 0:
                    event_confidence = max(event_confidence, self.CONFIDENCE_MEDIUM)
            
            # Check for after-hours access
            if not self._is_business_hours(timestamp):
                event_type = self.EVENT_AFTER_HOURS if not likely_human else self.EVENT_AFTER_HOURS_ACCESS
                reasons.append("Access outside business hours")
                if not is_vibration or likely_human:  # Increase confidence for non-vibration or human detection
                    event_confidence = max(event_confidence, self.CONFIDENCE_MEDIUM)
            
            # Check for high frequency access (now returns tuple of (bool, reasons))
            is_high_frequency, high_freq_reasons = self._check_high_frequency(camera_name, timestamp, detection_result)
            if is_high_frequency:
                event_type = self.EVENT_HIGH_FREQUENCY
                reasons.extend(high_freq_reasons)
                if not is_vibration:  # Only increase confidence if not vibration
                    event_confidence = max(event_confidence, self.CONFIDENCE_MEDIUM)
                
            # Increase confidence for longer or higher intensity motion
            if motion_duration > 5.0:
                reasons.append(f"Extended duration: {motion_duration:.1f}s")
                event_confidence = max(event_confidence, self.CONFIDENCE_MEDIUM)
                
            if motion_intensity > 70 and not is_vibration:
                reasons.append(f"High intensity motion: {motion_intensity}")
                event_confidence = max(event_confidence, self.CONFIDENCE_MEDIUM)
                
            # Highest confidence if multiple factors are present AND not vibration or IS human
            if len(reasons) >= 2 and (not is_vibration or likely_human):
                event_confidence = self.CONFIDENCE_HIGH
                
            # If this is a significant event, update the last alert time
            if event_type != self.EVENT_NORMAL and event_confidence >= self.CONFIDENCE_MEDIUM:
                self.last_alert_time[camera_name] = timestamp
        else:
            # We're rate-limiting this alert
            reasons.append(f"Similar alert reported recently (within {self.min_alert_interval}s)")
            # Still classify but reduce confidence
            event_confidence = min(event_confidence, self.CONFIDENCE_LOW)
        
        # Create classified event with enhanced information
        classified_event = {
            'camera': camera_name,
            'timestamp': timestamp,
            'datetime': datetime.datetime.fromtimestamp(timestamp).isoformat(),
            'event_type': event_type,
            'confidence': event_confidence,
            'reasons': reasons,
            'rate_limited': not allow_new_alert and not low_priority_event,
            'motion_data': {
                'intensity': motion_intensity,
                'duration': motion_duration,
                'motion_confidence': motion_confidence,
                'is_vibration': is_vibration
            }
        }
        
        # Add to classified events history
        self.classified_events[camera_name].append(classified_event)
        
        # Log significant events
        if event_type != self.EVENT_NORMAL or event_confidence >= self.CONFIDENCE_MEDIUM:
            log_message = f"Event: {event_type} on {camera_name} "
            log_message += f"({', '.join(reasons)})"
            
            if event_confidence == self.CONFIDENCE_HIGH:
                logger.warning(log_message)
            else:
                logger.info(log_message)
        
        return classified_event
    
    def get_recent_events(self, camera_name: str = None, 
                         event_type: str = None,
                         max_events: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent classified events, optionally filtered by camera and type.
        
        Args:
            camera_name: Optional camera name to filter by
            event_type: Optional event type to filter by
            max_events: Maximum number of events to return
            
        Returns:
            List of classified events
        """
        all_events = []
        
        # Get events from specified camera or all cameras
        if camera_name:
            cameras_to_check = [camera_name] if camera_name in self.classified_events else []
        else:
            cameras_to_check = list(self.classified_events.keys())
            
        # Collect events from each camera
        for cam in cameras_to_check:
            all_events.extend(self.classified_events[cam])
            
        # Filter by event type if specified
        if event_type:
            all_events = [e for e in all_events if e['event_type'] == event_type]
            
        # Sort by timestamp (newest first)
        all_events.sort(key=lambda e: e['timestamp'], reverse=True)
        
        # Return limited number of events
        return all_events[:max_events]
