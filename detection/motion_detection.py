"""
SmartVan Monitor - Motion Detection Module
-----------------------------------------
This module implements basic motion detection for the SmartVan Monitor system.

Key features:
- Background subtraction using MOG2 or KNN
- Region of Interest (ROI) definition
- Motion intensity scoring
- Configurable thresholds
- Minimum size filtering
"""

import time
import logging
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# Import the enhanced vibration analyzer
from .vibration_analyzer import VibrationAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("motion_detection")

class MotionDetector:
    """Class for detecting motion in video frames using background subtraction."""
    
    ALGORITHM_MOG2 = "MOG2"
    ALGORITHM_KNN = "KNN"
    MOVEMENT_STABILITY = 7 # The higher the value, the more stable the motion detection (5-8 frames is recommended for van motion detection)
    
    # Detection confidence levels
    CONFIDENCE_HIGH = 3    # High confidence motion detection
    CONFIDENCE_MEDIUM = 2  # Medium confidence detection
    CONFIDENCE_LOW = 1     # Low confidence/possible false positive
    CONFIDENCE_NONE = 0    # No motion detected
    
    def __init__(self, 
                algorithm: str = ALGORITHM_MOG2,
                learning_rate: float = 0.001,
                motion_threshold: int = 25,
                min_area: int = 300,
                blur_size: int = 21,
                roi_areas: Optional[List[Tuple[int, int, int, int]]] = None,
                noise_filter: float = 0.6,
                movement_stability: int = MOVEMENT_STABILITY,
                human_detection_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the motion detector with specified parameters.
        
        Args:
            algorithm: Background subtraction algorithm to use ("MOG2" or "KNN")
            learning_rate: How quickly the background model adapts to changes
            motion_threshold: Threshold for distinguishing foreground from background
            min_area: Minimum contour area to be considered motion
            blur_size: Size of Gaussian blur kernel for noise reduction
            roi_areas: List of ROI areas as (x, y, width, height) tuples
            noise_filter: Threshold (0-1) for filtering sporadic noise detections
            movement_stability: Number of consecutive frames required to confirm motion
            human_detection_config: Dictionary of human detection parameters
        """
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.motion_threshold = motion_threshold
        self.min_area = min_area
        self.blur_size = blur_size
        self.roi_areas = roi_areas or []
        self.noise_filter = noise_filter
        self.movement_stability = movement_stability
        
        # Motion stability tracking
        self.consecutive_motion_frames = 0
        self.stable_motion_detected = False
        self.motion_history = []  # Track recent motion detections
        
        # Create background subtractor based on algorithm choice
        if algorithm == self.ALGORITHM_MOG2:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500,
                varThreshold=16,
                detectShadows=True
            )
        elif algorithm == self.ALGORITHM_KNN:
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
                history=500,
                dist2Threshold=400.0,
                detectShadows=True
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Use MOG2 or KNN.")
        
        # Optical flow for motion verification
        self.flow_history = []
        self.prev_gray = None
        self.use_optical_flow = True
        
        # Human detection parameters (configurable)
        self.human_detection_config = human_detection_config or {}
        self.human_detection_enabled = self.human_detection_config.get("enabled", True)
        
        # Get the optimized flow thresholds from memory
        self.flow_magnitude_threshold = self.human_detection_config.get("flow_magnitude_threshold", 3.5)
        self.flow_uniformity_threshold = self.human_detection_config.get("flow_uniformity_threshold", 0.5)
        self.significant_motion_threshold = self.human_detection_config.get("significant_motion_threshold", 6.0)
        
        # Initialize enhanced vibration analyzer for improved filtering
        self.vibration_analyzer = VibrationAnalyzer(
            window_size=20,             # 20 frames history (10 seconds at 2 FPS)
            pattern_threshold=0.7,      # Threshold for vibration pattern detection
            adaptive_learning=True      # Enable adaptive thresholds
        )
        
        self.human_min_duration = self.human_detection_config.get("min_duration", 2.0)
        
        # Track motion state
        self.last_motion_time = None
        self.motion_duration = 0
        self.is_motion_active = False
        
        # Frame dimensions (will be set on first detection)
        self.frame_width = None
        self.frame_height = None
        
        # Time tracking for day/night awareness
        self.last_time_check = time.time()
        self.is_night_time = False  # Will be updated based on time
        
        # Initialize time-aware thresholds with default values
        self.effective_threshold = self.motion_threshold
        self.effective_min_area = self.min_area
        
        logger.info(f"Enhanced motion detector initialized with {algorithm} algorithm")
    
    def add_roi(self, x: int, y: int, width: int, height: int) -> None:
        """
        Add a Region of Interest (ROI) to monitor for motion.
        
        Args:
            x: X-coordinate of top-left corner
            y: Y-coordinate of top-left corner
            width: Width of ROI
            height: Height of ROI
        """
        self.roi_areas.append((x, y, width, height))
        logger.info(f"Added ROI at ({x}, {y}) with size {width}x{height}")
    
    def clear_roi(self) -> None:
        """Clear all defined ROIs."""
        self.roi_areas = []
        logger.info("Cleared all ROIs")
    
    def apply_roi_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply ROI mask to the frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with areas outside ROIs masked out
        """
        if not self.roi_areas:
            return frame  # No ROIs defined, use full frame
        
        # Store frame dimensions on first run    
        if self.frame_width is None or self.frame_height is None:
            self.frame_height, self.frame_width = frame.shape[:2]
            
        # Create mask of same size as frame, initialize to zeros (all black)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # Fill ROI areas with white (255)
        for x, y, w, h in self.roi_areas:
            # Ensure ROI coordinates are within frame boundaries
            x = max(0, min(x, self.frame_width - 1))
            y = max(0, min(y, self.frame_height - 1))
            w = min(w, self.frame_width - x)
            h = min(h, self.frame_height - y)
            
            if w > 0 and h > 0:  # Ensure valid dimensions
                mask[y:y+h, x:x+w] = 255
            
        # Apply mask to frame
        return cv2.bitwise_and(frame, frame, mask=mask)
    
    def _check_time_of_day(self) -> None:
        """
        Check if current time is day or night to adjust detection sensitivity.
        """
        current_time = time.time()
        # Only check once per minute to avoid excessive calculations
        if current_time - self.last_time_check > 60:
            self.last_time_check = current_time
            
            # Get current hour (0-23)
            current_hour = time.localtime(current_time).tm_hour
            
            # Define night time as 7PM to 7AM (19-23, 0-7)
            self.is_night_time = current_hour >= 19 or current_hour < 7
            
            # Adjust detection parameters based on time of day
            if self.is_night_time:
                # More conservative at night to prevent false positives
                self.effective_threshold = self.motion_threshold + 5
                self.effective_min_area = self.min_area * 1.2
            else:
                # Standard parameters during day
                self.effective_threshold = self.motion_threshold
                self.effective_min_area = self.min_area
    
    def _calculate_optical_flow(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Calculate optical flow to verify motion and filter camera shake/vibration.
        Enhanced to detect human movement patterns vs. vibration using advanced pattern analysis.
        
        Args:
            frame: Current frame in grayscale
            
        Returns:
            Dictionary containing flow magnitude, uniformity, vibration and human detection metrics
        """
        if not self.use_optical_flow or self.prev_gray is None:
            self.prev_gray = frame.copy()
            return {"magnitude": 0.0, "uniformity": 1.0, "is_vibration": False, "likely_human": False}
        
        # Calculate optical flow with optimized parameters for better motion detection
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Calculate the magnitude and direction of the flow
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Get average flow magnitude
        flow_magnitude = np.mean(mag)
        
        # Enhanced flow uniformity calculation with weighted regions
        # This helps better distinguish between uniform vehicle vibrations and non-uniform human movements
        mask = mag > 0.5
        if np.any(mask):
            angles_of_interest = ang[mask]
            # Use magnitudes as weights to emphasize stronger movements
            weights = mag[mask] / np.sum(mag[mask])
            # Calculate weighted std dev of angles
            mean_angle = np.average(angles_of_interest, weights=weights)
            angle_variance = np.average((angles_of_interest - mean_angle)**2, weights=weights)
            angle_std = np.sqrt(angle_variance)
            
            # Normalize to 0-1 range (higher value means less uniform, more likely human activity)
            # Using the optimized flow_uniformity_threshold from config (0.5)
            flow_uniformity = min(1.0, angle_std / np.pi)
        else:
            flow_uniformity = 0.0
        
        # Extract additional flow components for enhanced analysis
        flow_components = self.vibration_analyzer.extract_flow_components(flow)
        
        # Enhance edge detection for partial human presence (like hands reaching into frame)
        # Calculate additional edge metrics for the flow components
        h, w = flow.shape[:2]
        edge_margin = max(int(min(h, w) * 0.15), 15)  # Increased edge margin (15% of frame)
        
        # Extract flow in edge regions with finer detail
        edge_regions = {
            "top": flow[:edge_margin, :, :],
            "bottom": flow[-edge_margin:, :, :],
            "left": flow[:, :edge_margin, :],
            "right": flow[:, -edge_margin:, :]
        }
        
        # Calculate directional edge flows (useful for detecting entry/exit)
        edge_directional_flows = {}
        for region_name, region in edge_regions.items():
            if region.size > 0:
                if region_name == "top":
                    # Flow coming from top should have positive y component
                    direction_component = np.mean(region[..., 1])
                    entry_flow = max(0, direction_component)  # Only count positive flow (entry)
                elif region_name == "bottom":
                    # Flow coming from bottom should have negative y component
                    direction_component = np.mean(region[..., 1])
                    entry_flow = max(0, -direction_component)  # Only count negative flow (entry)
                elif region_name == "left":
                    # Flow coming from left should have positive x component
                    direction_component = np.mean(region[..., 0])
                    entry_flow = max(0, direction_component)  # Only count positive flow (entry)
                elif region_name == "right":
                    # Flow coming from right should have negative x component
                    direction_component = np.mean(region[..., 0])
                    entry_flow = max(0, -direction_component)  # Only count negative flow (entry)
                
                edge_directional_flows[region_name] = entry_flow
        
        # Add directional edge flows to flow components
        flow_components.update({
            "edge_directional_flows": edge_directional_flows,
            "max_entry_flow": max(edge_directional_flows.values()) if edge_directional_flows else 0.0
        })
        
        # Use enhanced vibration analyzer for better pattern detection
        # This analyzes temporal patterns across frames to detect repetitive movements
        enhanced_detection = self.vibration_analyzer.detect_human_vs_vibration(
            flow_magnitude, 
            flow_uniformity,
            flow_components
        )
        
        # Get enhanced vibration detection results
        is_vibration = enhanced_detection["is_vibration"]
        vibration_confidence = enhanced_detection["vibration_confidence"]
        vibration_type = enhanced_detection["vibration_type"]
        likely_human = enhanced_detection["likely_human"]
        human_confidence = enhanced_detection["human_confidence"]
        
        # Extract additional metrics for more nuanced detection
        trajectory_score = enhanced_detection.get("trajectory_score", 0.0)
        partial_presence_score = enhanced_detection.get("partial_presence_score", 0.0)
        motion_intensity = enhanced_detection.get("intensity", 0.0)
        
        # Use motion intensity to filter out low-intensity vibrations
        # Based on the memory config value of motion_intensity_threshold: 40
        if is_vibration and motion_intensity < 40:
            vibration_confidence *= 0.7  # Reduce confidence for low intensity vibrations
        
        # Enhanced human detection based on trajectory and edge entry flow
        if not likely_human and flow_components.get("max_entry_flow", 0) > 2.0:
            # Strong entry flow suggests something entering the frame
            likely_human = True
            human_confidence = max(human_confidence, 0.65)  # Boost confidence but not to maximum
        
        # If enhanced vibration detection is very confident, override the default detection
        # but make sure human detection still takes precedence when strong evidence exists
        if not self.human_detection_enabled:
            likely_human = False
        elif likely_human and human_confidence > 0.7:
            # Strong human confidence overrides vibration detection
            is_vibration = False
        
        # Log significant detections for debugging
        if is_vibration and vibration_confidence > 0.8:
            logger.debug(f"High confidence vibration: {vibration_type}, confidence={vibration_confidence:.2f}")
        elif likely_human and human_confidence > 0.8:
            logger.debug(f"High confidence human movement, confidence={human_confidence:.2f}")
        
        # Update previous frame
        self.prev_gray = frame.copy()
        
        return {
            "magnitude": flow_magnitude,
            "uniformity": flow_uniformity,
            "is_vibration": is_vibration,
            "likely_human": likely_human,
            "vibration_confidence": vibration_confidence,
            "vibration_type": vibration_type,
            "human_confidence": human_confidence,
            "flow_components": flow_components,
            "trajectory_score": trajectory_score,
            "partial_presence_score": partial_presence_score,
            "motion_intensity": motion_intensity,
            "max_entry_flow": flow_components.get("max_entry_flow", 0.0)
        }
    
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect motion in the current frame.
        
        Args:
            frame: Current video frame
            
        Returns:
            Dictionary containing motion information:
                - motion_detected: Boolean indicating if motion was detected
                - bounding_boxes: List of (x, y, width, height) for motion areas
                - intensity: Motion intensity score (0-100)
                - duration: Duration of continuous motion in seconds
                - mask: Binary mask showing motion areas
                - confidence: Detection confidence level (0-3)
                - is_night_time: Whether detection occurred during night hours
        """
        # Check time of day for sensitivity adjustment
        self._check_time_of_day()
        
        # Apply ROI mask if defined
        masked_frame = self.apply_roi_mask(frame)
        
        # Convert to grayscale for optical flow
        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow for motion verification with enhanced vibration detection
        flow_data = self._calculate_optical_flow(gray)
        flow_magnitude = flow_data["magnitude"]
        flow_uniformity = flow_data["uniformity"]
        is_vibration = flow_data["is_vibration"]
        vibration_confidence = flow_data.get("vibration_confidence", 0.0)
        vibration_type = flow_data.get("vibration_type", "UNKNOWN")
        likely_human = flow_data.get("likely_human", False)
        human_confidence = flow_data.get("human_confidence", 0.0)
        
        # Apply Gaussian blur to reduce noise
        if self.blur_size > 0:
            blurred = cv2.GaussianBlur(masked_frame, 
                                      (self.blur_size, self.blur_size), 0)
        else:
            blurred = masked_frame
            
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(
            blurred, 
            learningRate=self.learning_rate
        )
        
        # Threshold to binary image to remove shadows (value 127)
        _, thresh = cv2.threshold(
            fg_mask, 
            self.effective_threshold,  # Use time-aware threshold 
            255, 
            cv2.THRESH_BINARY
        )
        
        # Additional morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(
            thresh, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by minimum area and check if they're within ROIs
        significant_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.effective_min_area:  # Use time-aware min area
                # Additional check for ROI if defined
                if self.roi_areas:
                    x, y, w, h = cv2.boundingRect(contour)
                    contour_center = (x + w//2, y + h//2)
                    
                    # Check if contour center is within any ROI
                    in_roi = False
                    for roi_x, roi_y, roi_w, roi_h in self.roi_areas:
                        if (roi_x <= contour_center[0] <= roi_x + roi_w and 
                            roi_y <= contour_center[1] <= roi_y + roi_h):
                            in_roi = True
                            break
                            
                    if in_roi:
                        significant_contours.append(contour)
                else:
                    significant_contours.append(contour)
        
        # Calculate motion intensity (0-100 scale)
        # Based on percentage of frame covered by motion, contour count, and flow magnitude
        total_area = sum(cv2.contourArea(c) for c in significant_contours)
        frame_area = frame.shape[0] * frame.shape[1]
        area_percentage = (total_area / frame_area) * 100 if frame_area > 0 else 0
        contour_factor = min(len(significant_contours), 10) / 10  # Scale to 0-1
        
        # Factor in optical flow and uniformity
        # - High uniformity with motion suggests vehicle movement/vibration (reduce score)
        # - Low uniformity suggests localized motion like human activity (increase score)
        flow_factor = min(1.0, flow_magnitude / 2.0) * flow_uniformity  # Weight by non-uniformity
        
        # Weight factors differently
        # Area percentage is most important, then contours, then flow
        raw_intensity = (area_percentage * 0.5) + (contour_factor * 30) + (flow_factor * 20)
        
        # Enhanced vibration filtering based on pattern analysis
        if is_vibration:
            # More aggressive reduction for repetitive movements
            if vibration_type == "REPETITIVE_MOVEMENT":
                raw_intensity *= 0.1  # 90% reduction for repetitive movements
            # Significant reduction for high-frequency vehicle vibrations
            elif vibration_type == "HIGH_FREQ_VEHICLE":
                raw_intensity *= 0.2  # 80% reduction for vehicle vibrations
            # Standard reduction for general vibrations
            else:
                raw_intensity *= 0.3  # 70% reduction for general vibrations
                
            # Apply confidence-based scaling
            # Higher vibration confidence = more reduction
            confidence_factor = max(0.1, 1.0 - vibration_confidence)
            raw_intensity *= confidence_factor
            
        intensity = int(min(100, raw_intensity))
        
        # Get bounding boxes for contours
        boxes = []
        for contour in significant_contours:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))
        
        # Determine if motion is detected based on stability
        preliminary_motion = len(significant_contours) > 0
        
        # Update motion stability tracking
        if preliminary_motion:
            self.consecutive_motion_frames += 1
        else:
            self.consecutive_motion_frames = 0
            
        # Motion is only considered stable if detected for several consecutive frames
        # If likely human is detected, we can be more lenient with stability requirements
        if likely_human and intensity > 30:
            # Human movement gets a boost to stability - require fewer frames
            stable_motion = self.consecutive_motion_frames >= max(2, self.movement_stability // 2)
        else:
            # Regular stability check
            stable_motion = self.consecutive_motion_frames >= self.movement_stability
        
        # Update motion history for trend analysis
        self.motion_history.append(preliminary_motion)
        if len(self.motion_history) > 10:  # Keep last 10 frames
            self.motion_history.pop(0)
            
        # Determine confidence level with enhanced human vs. vibration detection
        confidence = self.CONFIDENCE_NONE
        
        if stable_motion:
            # Start with low confidence
            confidence = self.CONFIDENCE_LOW
            
            # Enhanced vibration filtering
            if is_vibration and not likely_human:
                # Complete rejection of vibrations with high confidence
                if vibration_confidence > 0.7:
                    confidence = self.CONFIDENCE_NONE
                # Downgrade confidence for medium confidence vibrations
                elif vibration_confidence > 0.4:
                    confidence = max(self.CONFIDENCE_NONE, confidence - 1)
            
            # Enhanced human detection
            if likely_human:
                # Boost confidence based on human detection confidence
                if human_confidence > 0.7:
                    confidence = max(confidence + 1, self.CONFIDENCE_MEDIUM)
                if human_confidence > 0.9:
                    confidence = self.CONFIDENCE_HIGH
            
            # Increase confidence if motion has high intensity and duration
            if intensity > 40 and self.motion_duration >= 1.0:
                confidence = max(confidence, self.CONFIDENCE_MEDIUM)
                
            # Highest confidence if sustained high intensity motion
            if intensity > 70 and self.motion_duration >= 2.0:
                confidence = self.CONFIDENCE_HIGH
        # Update duration tracking
        current_time = time.time()
        if stable_motion:
            if not self.is_motion_active:
                # Motion just started
                self.is_motion_active = True
                self.last_motion_time = current_time
            # Update duration if motion continues
            self.motion_duration = current_time - self.last_motion_time
        else:
            if self.is_motion_active:
                # Motion just ended
                self.is_motion_active = False
            # Reset duration if no motion
            self.motion_duration = 0
        
        # Return enhanced detection results with vibration information
        return {
            # Reject motion if it's high-confidence vibration and not likely human
            "motion_detected": stable_motion and (not (is_vibration and vibration_confidence > 0.7) or likely_human),
            "bounding_boxes": boxes,
            "intensity": intensity,
            "duration": round(self.motion_duration, 1),
            "mask": thresh,
            "confidence": confidence,
            "is_night_time": self.is_night_time,
            "frame_count": self.consecutive_motion_frames,
            "flow_magnitude": flow_magnitude,
            "flow_uniformity": flow_uniformity,
            "is_vibration": is_vibration and not likely_human,  # Override vibration if human movement is detected
            "vibration_confidence": vibration_confidence,
            "vibration_type": vibration_type,
            "likely_human": likely_human,
            "human_confidence": human_confidence
        }
    
    def visualize(self, frame: np.ndarray, detection_result: Dict[str, Any], show_text: bool = True) -> np.ndarray:
        """
        Visualize the detection result by drawing bounding boxes and text information.
        Enhanced with vibration analysis visualization.
        
        Args:
            frame: Original frame
            detection_result: Result from detect method
            show_text: Whether to show text information
            
        Returns:
            Visualized frame with detection information
        """
        output_frame = frame.copy()
        
        # Draw bounding boxes
        if detection_result["motion_detected"]:
            for box in detection_result["bounding_boxes"]:
                x, y, w, h = box
                cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add text information
        if show_text:
            motion_text = "Motion: Yes" if detection_result["motion_detected"] else "Motion: No"
            intensity_text = f"Intensity: {detection_result['intensity']}"
            
            # Enhanced vibration display
            if detection_result.get("is_vibration", False):
                vib_conf = detection_result.get("vibration_confidence", 0.0)
                vib_type = detection_result.get("vibration_type", "UNKNOWN")
                vibration_text = f"Type: {vib_type} ({vib_conf:.2f})"
                vibration_color = (0, 0, 255)  # Red for vibration
            else:
                vibration_text = "Type: Normal"
                vibration_color = (0, 255, 0)  # Green for normal
            
            # Enhanced human detection display
            if detection_result.get("likely_human", False):
                human_conf = detection_result.get("human_confidence", 0.0)
                human_text = f"Human: Likely ({human_conf:.2f})"
                human_color = (255, 128, 0)  # Orange for human
            else:
                human_text = "Human: No"
                human_color = (0, 255, 0)  # Green
            
            # Add confidence level
            confidence_level = detection_result.get("confidence", 0)
            confidence_names = ["None", "Low", "Medium", "High"]
            confidence_text = f"Confidence: {confidence_names[confidence_level]}"
            
            y_pos = 30
            cv2.putText(output_frame, motion_text, (10, y_pos), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_pos += 30
            cv2.putText(output_frame, intensity_text, (10, y_pos), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_pos += 30
            cv2.putText(output_frame, vibration_text, (10, y_pos), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, vibration_color, 2)
            y_pos += 30
            cv2.putText(output_frame, human_text, (10, y_pos), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, human_color, 2)
            y_pos += 30
            cv2.putText(output_frame, confidence_text, (10, y_pos), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return output_frame

# Example usage
if __name__ == "__main__":
    # This code will run when the module is executed directly
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test motion detection on a video file or camera")
    parser.add_argument("--input", default=0, help="Path to video file or camera index (default: 0)")
    parser.add_argument("--algorithm", default="MOG2", choices=["MOG2", "KNN"], help="Background subtraction algorithm")
    parser.add_argument("--threshold", type=int, default=25, help="Motion threshold")
    parser.add_argument("--min-area", type=int, default=200, help="Minimum contour area")
    args = parser.parse_args()
    
    # Create motion detector
    detector = MotionDetector(
        algorithm=args.algorithm,
        motion_threshold=args.threshold,
        min_area=args.min_area
    )
    
    # Try to interpret input as camera index if it's a number
    video_source = args.input
    try:
        video_source = int(args.input)
    except ValueError:
        pass  # Not an integer, use as file path
    
    # Open video capture
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {video_source}")
        exit(1)
    
    logger.info(f"Press 'q' to quit, 'r' to add ROI, 'c' to clear ROIs")
    
    # Process frames
    drawing_roi = False
    roi_start = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        result = detector.detect(frame)
        
        # Visualize results
        output = detector.visualize(frame, result)
        
        # Display frame
        cv2.imshow("Motion Detection", output)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            logger.info("Click and drag to define ROI")
            drawing_roi = True
            roi_start = None
        elif key == ord("c"):
            detector.clear_roi()
        
        # Handle ROI drawing
        if drawing_roi:
            cv2.setMouseCallback("Motion Detection", None)  # Reset previous callback
            
            # Define callback function with proper scope access
            def mouse_callback(event, x, y, flags, param):
                # Use global variables instead of nonlocal for top-level scope
                global drawing_roi, roi_start
                
                if event == cv2.EVENT_LBUTTONDOWN:
                    roi_start = (x, y)
                elif event == cv2.EVENT_LBUTTONUP and roi_start:
                    roi_end = (x, y)
                    x_start, y_start = roi_start
                    width = abs(x - x_start)
                    height = abs(y - y_start)
                    x = min(x_start, x)
                    y = min(y_start, y)
                    detector.add_roi(x, y, width, height)
                    drawing_roi = False
                    
            cv2.setMouseCallback("Motion Detection", mouse_callback)
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
