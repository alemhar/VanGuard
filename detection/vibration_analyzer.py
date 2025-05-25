"""
SmartVan Monitor - Enhanced Vibration Analysis
----------------------------------------------
This module provides enhanced vibration analysis capabilities for the SmartVan Monitor
to better distinguish between vehicle vibrations and genuine human presence.
"""

import numpy as np
import cv2
import time
import logging
from typing import Dict, List, Any, Tuple, Optional
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vibration_analyzer")

class VibrationAnalyzer:
    """
    Advanced vibration analysis for SmartVan Monitor.
    This class implements enhanced detection and filtering of vehicle vibrations
    to reduce false positives while maintaining sensitivity to human movements.
    """
    
    def __init__(self, 
                window_size: int = 20,        # Number of frames to analyze
                pattern_threshold: float = 0.7,  # Threshold for pattern detection
                freq_threshold: float = 0.5,   # Threshold for frequency analysis
                adaptive_learning: bool = True, # Enable adaptive learning
                pattern_memory: int = 10):     # Number of patterns to remember
        """
        Initialize the vibration analyzer.
        
        Args:
            window_size: Number of frames to keep in history for pattern analysis
            pattern_threshold: Threshold for pattern recognition (0.0-1.0)
            freq_threshold: Threshold for frequency analysis (0.0-1.0)
            adaptive_learning: Whether to enable adaptive threshold learning
            pattern_memory: Number of distinct patterns to remember
        """
        self.window_size = window_size
        self.pattern_threshold = pattern_threshold
        self.freq_threshold = freq_threshold
        self.adaptive_learning = adaptive_learning
        
        # History of flow data
        self.flow_history = deque(maxlen=window_size)
        
        # Detected vibration patterns
        self.vibration_patterns = []
        self.pattern_memory = pattern_memory
        
        # Adaptive thresholds
        self.magnitude_baseline = 1.0
        self.uniformity_baseline = 0.3
        
        # Frequency analysis
        self.frequency_counts = {}
        self.frequency_timestamps = {}
        
        # Repetitive movement detection
        self.rep_movement_threshold = 3  # Minimum repetitions to consider repetitive
        self.last_flow_direction = None
        self.direction_changes = 0
        self.direction_change_times = deque(maxlen=10)
        
        logger.info("VibrationAnalyzer initialized")
        
    def update(self, 
              flow_magnitude: float, 
              flow_uniformity: float, 
              flow_data: Optional[Dict[str, Any]] = None,
              timestamp: float = None) -> None:
        """
        Update the analyzer with new flow data.
        
        Args:
            flow_magnitude: Magnitude of optical flow
            flow_uniformity: Uniformity of optical flow
            flow_data: Additional flow data (if available)
            timestamp: Current timestamp (default: current time)
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Append to history
        self.flow_history.append({
            "magnitude": flow_magnitude,
            "uniformity": flow_uniformity,
            "timestamp": timestamp,
            "additional_data": flow_data
        })
        
        # Detect direction changes for repetitive movement analysis
        if flow_data and "dx" in flow_data and "dy" in flow_data:
            self._analyze_flow_direction(flow_data["dx"], flow_data["dy"], timestamp)
            
        # Update adaptive thresholds if enabled
        if self.adaptive_learning and len(self.flow_history) >= self.window_size // 2:
            self._update_adaptive_thresholds()
    
    def _analyze_flow_direction(self, dx: float, dy: float, timestamp: float) -> None:
        """
        Analyze flow direction changes to detect repetitive movements.
        
        Args:
            dx: X component of flow
            dy: Y component of flow
            timestamp: Current timestamp
        """
        # Skip tiny movements
        if abs(dx) < 0.1 and abs(dy) < 0.1:
            return
            
        # Determine primary direction (simplified to 4 directions)
        if abs(dx) > abs(dy):
            current_direction = "right" if dx > 0 else "left"
        else:
            current_direction = "down" if dy > 0 else "up"
            
        # Check for direction change
        if self.last_flow_direction is not None and current_direction != self.last_flow_direction:
            self.direction_changes += 1
            self.direction_change_times.append(timestamp)
            
            # Analyze frequency of direction changes
            if len(self.direction_change_times) >= 2:
                intervals = []
                for i in range(1, len(self.direction_change_times)):
                    intervals.append(self.direction_change_times[i] - self.direction_change_times[i-1])
                
                # Check if intervals are consistent (repetitive movement)
                if len(intervals) >= 2:
                    mean_interval = np.mean(intervals)
                    std_interval = np.std(intervals)
                    cv = std_interval / mean_interval if mean_interval > 0 else float('inf')
                    
                    # Low coefficient of variation indicates regular repetitive movement
                    # which is more likely to be vibration than human activity
                    if cv < 0.3 and mean_interval < 0.5:  # Fast repetitive movement
                        logger.debug(f"Detected repetitive movement: {mean_interval:.2f}s intervals, CV={cv:.2f}")
        
        self.last_flow_direction = current_direction
    
    def _update_adaptive_thresholds(self) -> None:
        """Update the adaptive thresholds based on recent flow history."""
        recent_magnitudes = [entry["magnitude"] for entry in self.flow_history]
        recent_uniformities = [entry["uniformity"] for entry in self.flow_history]
        
        # Calculate statistics
        mag_mean = np.mean(recent_magnitudes)
        mag_std = np.std(recent_magnitudes)
        uni_mean = np.mean(recent_uniformities)
        
        # Update baselines slowly (exponential moving average)
        self.magnitude_baseline = 0.9 * self.magnitude_baseline + 0.1 * mag_mean
        self.uniformity_baseline = 0.9 * self.uniformity_baseline + 0.1 * uni_mean
        
        # Log significant changes
        if abs(self.magnitude_baseline - mag_mean) > 1.0:
            logger.debug(f"Significant change in flow magnitude baseline: {self.magnitude_baseline:.2f}")
    
    def analyze_vibration_pattern(self) -> Dict[str, Any]:
        """
        Analyze the current flow history to detect vibration patterns.
        
        Returns:
            Dictionary with vibration analysis results
        """
        if len(self.flow_history) < self.window_size // 2:
            return {
                "is_vibration": False,
                "confidence": 0.0,
                "type": "UNKNOWN",
                "frequency": 0.0
            }
        
        # Extract features for analysis
        magnitudes = [entry["magnitude"] for entry in self.flow_history]
        uniformities = [entry["uniformity"] for entry in self.flow_history]
        timestamps = [entry["timestamp"] for entry in self.flow_history]
        
        # Calculate time-based features
        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        if duration <= 0:
            freq = 0
        else:
            # Estimate frequency of oscillation using zero crossings
            # Normalize magnitudes to detect oscillations around mean
            norm_mags = np.array(magnitudes) - np.mean(magnitudes)
            zero_crossings = np.where(np.diff(np.signbit(norm_mags)))[0]
            freq = len(zero_crossings) / (2 * duration) if duration > 0 else 0
        
        # Advanced pattern detection
        # 1. Check for consistent small magnitude
        small_consistent_magnitude = (
            np.mean(magnitudes) < 2.0 and 
            np.std(magnitudes) / np.mean(magnitudes) < 0.5 if np.mean(magnitudes) > 0 else False
        )
        
        # 2. Check for low uniformity (consistent direction)
        consistent_direction = np.mean(uniformities) < 0.3
        
        # 3. Check for repetitive high-frequency pattern
        high_frequency = freq > 1.0  # More than 1 oscillation per second
        
        # 4. Check for repetitive movement pattern
        repetitive_movement = (
            self.direction_changes >= self.rep_movement_threshold and
            len(self.direction_change_times) >= 2
        )
        
        # Calculate overall vibration confidence
        vibration_confidence = 0.0
        if small_consistent_magnitude:
            vibration_confidence += 0.3
        if consistent_direction:
            vibration_confidence += 0.3
        if high_frequency:
            vibration_confidence += 0.2
        if repetitive_movement:
            vibration_confidence += 0.2
            
        # Determine vibration type
        vibration_type = "UNKNOWN"
        if vibration_confidence > 0.6:
            if high_frequency and small_consistent_magnitude:
                vibration_type = "HIGH_FREQ_VEHICLE"
            elif repetitive_movement and consistent_direction:
                vibration_type = "REPETITIVE_MOVEMENT"
            else:
                vibration_type = "GENERAL_VIBRATION"
        
        # Determine if this is a vibration based on confidence
        is_vibration = vibration_confidence > self.pattern_threshold
        
        # Log significant vibrations
        if is_vibration and vibration_confidence > 0.8:
            logger.debug(
                f"High confidence vibration detected: {vibration_type}, " +
                f"Freq={freq:.1f}Hz, Conf={vibration_confidence:.2f}"
            )
        
        return {
            "is_vibration": is_vibration,
            "confidence": vibration_confidence,
            "type": vibration_type,
            "frequency": freq,
            "repetitive_movement": repetitive_movement,
            "small_magnitude": small_consistent_magnitude,
            "consistent_direction": consistent_direction,
            "direction_changes": self.direction_changes
        }
    
    def extract_flow_components(self, flow: np.ndarray) -> Dict[str, Any]:
        """
        Extract useful components from optical flow data.
        
        Args:
            flow: Optical flow data from cv2.calcOpticalFlowFarneback
            
        Returns:
            Dictionary with extracted components
        """
        if flow is None or flow.size == 0:
            return {
                "dx": 0.0,
                "dy": 0.0,
                "magnitude": 0.0,
                "angle": 0.0
            }
        
        # Extract x and y components
        dx = np.mean(flow[..., 0])
        dy = np.mean(flow[..., 1])
        
        # Calculate magnitude and angle
        magnitude = np.sqrt(dx*dx + dy*dy)
        angle = np.arctan2(dy, dx)
        
        # Calculate edge flow (flow near image edges)
        h, w = flow.shape[:2]
        edge_margin = max(int(min(h, w) * 0.1), 10)  # 10% of image dimension
        
        # Extract edge regions
        top_edge = flow[:edge_margin, :, :]
        bottom_edge = flow[-edge_margin:, :, :]
        left_edge = flow[:, :edge_margin, :]
        right_edge = flow[:, -edge_margin:, :]
        
        # Calculate mean flow in edge regions
        edge_regions = [top_edge, bottom_edge, left_edge, right_edge]
        edge_flow = 0.0
        for region in edge_regions:
            if region.size > 0:
                region_mag = np.mean(np.sqrt(region[..., 0]**2 + region[..., 1]**2))
                edge_flow = max(edge_flow, region_mag)
        
        return {
            "dx": dx,
            "dy": dy,
            "magnitude": magnitude,
            "angle": angle,
            "edge_flow": edge_flow
        }
    
    def detect_human_vs_vibration(self, 
                                 flow_magnitude: float, 
                                 flow_uniformity: float,
                                 flow_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Advanced detection to distinguish between human movement and vibration.
        
        Args:
            flow_magnitude: Magnitude of optical flow
            flow_uniformity: Uniformity of optical flow
            flow_data: Additional flow data (if available)
            
        Returns:
            Dictionary with detection results
        """
        # Update analyzer with current flow data
        self.update(flow_magnitude, flow_uniformity, flow_data)
        
        # Get vibration pattern analysis
        vibration_analysis = self.analyze_vibration_pattern()
        is_vibration = vibration_analysis["is_vibration"]
        vibration_confidence = vibration_analysis["confidence"]
        
        # Human movement typically has:
        # - Higher magnitude than vehicle vibration
        # - Less repetitive pattern
        # - More non-uniform direction (flow_uniformity > 0.4)
        # - Often concentrated on edges (something entering/exiting frame)
        
        # Edge flow is a good indicator of human movement (entering/exiting frame)
        edge_flow_strength = flow_data.get("edge_flow", 0.0) if flow_data else 0.0
        
        # Indicators of human presence
        significant_magnitude = flow_magnitude > 3.0
        non_uniform_direction = flow_uniformity > 0.4
        strong_edge_flow = edge_flow_strength > 1.5
        non_repetitive = not vibration_analysis.get("repetitive_movement", False)
        
        # Calculate human movement confidence
        human_confidence = 0.0
        if significant_magnitude:
            human_confidence += 0.3
        if non_uniform_direction:
            human_confidence += 0.2
        if strong_edge_flow:
            human_confidence += 0.3
        if non_repetitive:
            human_confidence += 0.2
            
        # Determine if this is likely human movement
        # If vibration confidence is high, require stronger human indicators
        likely_human = human_confidence > 0.5
        if vibration_confidence > 0.7:
            # With high vibration confidence, require more human evidence
            likely_human = human_confidence > 0.7
            
        # Special case: very strong edge flow almost always indicates human
        if strong_edge_flow and edge_flow_strength > 3.0:
            likely_human = True
            
        # Special case: very high magnitude with non-uniform direction
        if flow_magnitude > 5.0 and flow_uniformity > 0.6:
            likely_human = True
            
        return {
            "is_vibration": is_vibration,
            "vibration_confidence": vibration_confidence,
            "vibration_type": vibration_analysis["type"],
            "likely_human": likely_human,
            "human_confidence": human_confidence,
            "flow_magnitude": flow_magnitude,
            "flow_uniformity": flow_uniformity
        }
    
    def reset(self) -> None:
        """Reset the analyzer state."""
        self.flow_history.clear()
        self.direction_changes = 0
        self.direction_change_times.clear()
        self.last_flow_direction = None
