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
        # Not enough data for reliable analysis
        if len(self.flow_history) < self.window_size // 2:
            return {
                "is_vibration": False,
                "confidence": 0.0,
                "type": "unknown",
                "repetitive_movement": False,
                "frequency": 0.0,
                "intensity": 0.0
            }
        
        # Extract magnitudes and timestamps
        magnitudes = [item["magnitude"] for item in self.flow_history]
        timestamps = [item["timestamp"] for item in self.flow_history]
        uniformities = [item["uniformity"] for item in self.flow_history]
        
        # Calculate intensity measures
        max_magnitude = max(magnitudes)
        avg_magnitude = sum(magnitudes) / len(magnitudes)
        magnitude_variance = sum((m - avg_magnitude) ** 2 for m in magnitudes) / len(magnitudes)
        
        # Calculate temporal features
        total_time = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.001
        if total_time <= 0:
            total_time = 0.001  # Avoid division by zero
            
        # Calculate frequency if we have direction changes
        frequency = 0.0
        if len(self.direction_change_times) >= 2:
            # Calculate average time between direction changes
            change_times = list(self.direction_change_times)
            time_diffs = [change_times[i+1] - change_times[i] for i in range(len(change_times)-1)]
            if time_diffs:
                avg_time_diff = sum(time_diffs) / len(time_diffs)
                if avg_time_diff > 0:
                    frequency = 1.0 / avg_time_diff  # frequency in Hz
        
        # Calculate regularity of movement
        regularity = 0.0
        if len(magnitudes) >= 3:
            # Check for alternating patterns in magnitude
            diffs = [abs(magnitudes[i+1] - magnitudes[i]) for i in range(len(magnitudes)-1)]
            avg_diff = sum(diffs) / len(diffs)
            diff_variance = sum((d - avg_diff) ** 2 for d in diffs) / len(diffs)
            
            # Low variance in differences indicates regular patterns
            if avg_diff > 0:
                regularity = 1.0 - min(1.0, diff_variance / (avg_diff * avg_diff))
                
            # Enhanced pattern detection: check for repeating sequence patterns
            if len(magnitudes) >= 6:
                # Look for repeating sequences of 2-3 values
                pattern_detected = False
                for pattern_length in [2, 3]:
                    if len(magnitudes) >= pattern_length * 2:
                        # Compare adjacent segments
                        for start in range(0, len(magnitudes) - pattern_length * 2 + 1):
                            segment1 = magnitudes[start:start+pattern_length]
                            segment2 = magnitudes[start+pattern_length:start+pattern_length*2]
                            
                            # Calculate similarity between segments
                            similarity = 1.0
                            for i in range(pattern_length):
                                if max(segment1[i], segment2[i]) > 0:
                                    similarity *= 1.0 - min(1.0, abs(segment1[i] - segment2[i]) / max(segment1[i], segment2[i]))
                            
                            if similarity > 0.85:  # High similarity threshold
                                pattern_detected = True
                                regularity = max(regularity, similarity)
                                break
                        
                        if pattern_detected:
                            break
        
        # Uniformity analysis: consistently high uniformity suggests vibration
        avg_uniformity = sum(uniformities) / len(uniformities)
        uniform_movement = avg_uniformity < 0.3  # Lower values indicate more uniform movement
        
        # Detect repetitive movements with improved thresholds
        repetitive_movement = False
        if (self.direction_changes >= self.rep_movement_threshold and frequency > 0.3) or regularity > 0.7:
            repetitive_movement = True
            
        # Calculate vibration signature based on magnitude pattern
        signature_strength = 0.0
        if repetitive_movement and uniform_movement and avg_magnitude < 4.0:
            signature_strength = min(1.0, (regularity * 0.7 + (1.0 - avg_uniformity) * 0.3))
            
        # Detect if this movement pattern matches known vibration patterns
        vibration_confidence = 0.0
        
        # Confidence based on frequency (typical van vibrations: 0.5-5 Hz)
        # Refined based on motion detection settings from memory
        if 0.5 <= frequency <= 5.0:
            # Higher confidence for frequencies in the 1.5-3.5 Hz range (common vehicle vibrations)
            if 1.5 <= frequency <= 3.5:
                freq_factor = 1.0 - abs(frequency - 2.5) / 2.0  # Peak confidence at 2.5Hz
                vibration_confidence += freq_factor * 0.3
            else:
                freq_factor = 0.7 - abs(frequency - 2.5) / 5.0  # Lower confidence for edge frequencies
                vibration_confidence += max(0, freq_factor * 0.2)
        
        # Confidence based on regularity
        vibration_confidence += regularity * 0.25
        
        # Confidence based on magnitude (typical vibration magnitude is low to moderate)
        # Improved threshold based on memory settings (flow_magnitude_threshold: 3.5)
        if max_magnitude < 3.5:
            mag_factor = 1.0 - (max_magnitude / 3.5)
            vibration_confidence += mag_factor * 0.2
        
        # Confidence based on movement uniformity
        if uniform_movement:
            # More uniform movement (low uniformity value) increases vibration confidence
            vibration_confidence += (1.0 - avg_uniformity) * 0.15
        
        # Confidence based on repetitive movement
        if repetitive_movement:
            vibration_confidence += 0.2
            
        # Signature strength provides additional confidence
        vibration_confidence += signature_strength * 0.2
            
        # Determine vibration type with improved frequency classification
        vibration_type = "unknown"
        if vibration_confidence > 0.5:
            if frequency < 1.0:
                vibration_type = "low_frequency"
            elif frequency < 3.0:
                vibration_type = "medium_frequency"
            else:
                vibration_type = "high_frequency"
        
        # Calculate motion intensity score - useful for filtering out minor vibrations
        # Use the significant_motion_threshold from memory: 6.0
        motion_intensity = avg_magnitude * 10  # Scale for readability
        
        # Determine if this is likely a vibration
        is_vibration = vibration_confidence > self.pattern_threshold
        
        # Filter out very low intensity vibrations (threshold: 40 from memory)
        if is_vibration and motion_intensity < 40:
            # Keep the vibration flag but with reduced confidence
            vibration_confidence *= 0.8
        
        return {
            "is_vibration": is_vibration,
            "confidence": vibration_confidence,
            "type": vibration_type,
            "repetitive_movement": repetitive_movement,
            "frequency": frequency,
            "regularity": regularity,
            "intensity": motion_intensity,
            "max_intensity": max_magnitude,
            "signature_strength": signature_strength,
            "uniformity": avg_uniformity,
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
        vibration_type = vibration_analysis["type"]
        repetitive_movement = vibration_analysis.get("repetitive_movement", False)
        
        # Human movement typically has:
        # - Higher magnitude than vehicle vibration
        # - Less repetitive pattern
        # - More non-uniform direction (flow_uniformity > 0.4)
        # - Often concentrated on edges (something entering/exiting frame)
        
        # Edge flow is a good indicator of human movement (entering/exiting frame)
        edge_flow_strength = flow_data.get("edge_flow", 0.0) if flow_data else 0.0
        
        # Extract flow components to analyze movement patterns
        dx = flow_data.get("dx", 0.0) if flow_data else 0.0
        dy = flow_data.get("dy", 0.0) if flow_data else 0.0
        
        # New: Calculate trajectory features to detect human movement patterns
        trajectory_score = 0.0
        if len(self.flow_history) >= 3:
            # Get last few flow points to analyze trajectory
            recent_flows = list(self.flow_history)[-3:]
            
            # Human movement tends to have consistent direction over short periods
            dx_values = [item.get("additional_data", {}).get("dx", 0) for item in recent_flows if item.get("additional_data")]
            dy_values = [item.get("additional_data", {}).get("dy", 0) for item in recent_flows if item.get("additional_data")]
            
            if dx_values and dy_values:
                # Calculate direction consistency (dot product of consecutive vectors)
                direction_consistency = 0.0
                for i in range(len(dx_values)-1):
                    # Normalize vectors
                    mag1 = max(0.0001, np.sqrt(dx_values[i]**2 + dy_values[i]**2))
                    mag2 = max(0.0001, np.sqrt(dx_values[i+1]**2 + dy_values[i+1]**2))
                    
                    # Calculate normalized dot product (cosine similarity)
                    dot_product = (dx_values[i] * dx_values[i+1] + dy_values[i] * dy_values[i+1]) / (mag1 * mag2)
                    direction_consistency += max(0, dot_product)  # Only count positive consistency
                
                # Average consistency
                if len(dx_values) > 1:
                    direction_consistency /= (len(dx_values) - 1)
                    trajectory_score = direction_consistency
        
        # New: Enhanced edge detection for partial human presence (e.g., hand reaching in)
        partial_presence_score = 0.0
        if edge_flow_strength > 1.0:
            # Calculate the ratio of edge flow to overall flow
            edge_ratio = edge_flow_strength / (flow_magnitude + 0.0001)  # Avoid division by zero
            
            # When edge flow is concentrated and overall flow is low, likely partial presence
            if edge_ratio > 2.0 and edge_flow_strength > 2.0:
                partial_presence_score = min(1.0, edge_ratio / 3.0)  # Cap at 1.0
        
        # Indicators of human presence with improved thresholds based on memory
        significant_magnitude = flow_magnitude > 3.5  # Based on memory settings
        non_uniform_direction = flow_uniformity > 0.5  # Based on memory settings
        strong_edge_flow = edge_flow_strength > 1.5
        non_repetitive = not repetitive_movement
        
        # Calculate human movement confidence with new factors
        human_confidence = 0.0
        if significant_magnitude:
            human_confidence += 0.25
        if non_uniform_direction:
            human_confidence += 0.15
        if strong_edge_flow:
            human_confidence += 0.25
        if non_repetitive:
            human_confidence += 0.15
        
        # Add trajectory and partial presence contributions
        human_confidence += trajectory_score * 0.1  # Max contribution: 0.1
        human_confidence += partial_presence_score * 0.1  # Max contribution: 0.1
            
        # Determine if this is likely human movement
        # Vibration intensity filtering: completely ignore low intensity vibrations (< 40)
        if flow_magnitude < 1.5 and vibration_confidence > 0.5:
            likely_human = False
            human_confidence *= 0.5  # Reduce confidence for very low magnitudes with vibration
        else:
            # If vibration confidence is high, require stronger human indicators
            likely_human = human_confidence > 0.5
            if vibration_confidence > 0.7:
                # With high vibration confidence, require more human evidence
                likely_human = human_confidence > 0.7
        
        # New: Detection refinements based on motion patterns
        # Repetitive, high-frequency movements with low magnitude are likely vibrations
        if repetitive_movement and vibration_analysis.get("frequency", 0) > 2.0 and flow_magnitude < 4.0:
            likely_human = False
            human_confidence *= 0.7  # Reduce confidence but don't eliminate completely
            
        # Special case: very strong edge flow almost always indicates human
        if strong_edge_flow and edge_flow_strength > 3.0:
            likely_human = True
            human_confidence = max(human_confidence, 0.8)  # Ensure high confidence
            
        # Special case: very high magnitude with non-uniform direction
        if flow_magnitude > 5.0 and flow_uniformity > 0.6:
            likely_human = True
            human_confidence = max(human_confidence, 0.8)  # Ensure high confidence
            
        # New: Strong partial presence detection
        if partial_presence_score > 0.7:
            likely_human = True
            human_confidence = max(human_confidence, 0.75)  # High but not maximum confidence
            
        return {
            "is_vibration": is_vibration,
            "vibration_confidence": vibration_confidence,
            "vibration_type": vibration_type,
            "likely_human": likely_human,
            "human_confidence": human_confidence,
            "flow_magnitude": flow_magnitude,
            "flow_uniformity": flow_uniformity,
            "edge_flow": edge_flow_strength,
            "trajectory_score": trajectory_score,
            "partial_presence_score": partial_presence_score,
            "repetitive_movement": repetitive_movement
        }
    
    def reset(self) -> None:
        """Reset the analyzer state."""
        self.flow_history.clear()
        self.direction_changes = 0
        self.direction_change_times.clear()
        self.last_flow_direction = None
