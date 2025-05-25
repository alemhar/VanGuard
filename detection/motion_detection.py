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

import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
import logging

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
        self.flow_magnitude_threshold = self.human_detection_config.get("flow_magnitude_threshold", 5.0)
        self.flow_uniformity_threshold = self.human_detection_config.get("flow_uniformity_threshold", 0.6)
        self.significant_motion_threshold = self.human_detection_config.get("significant_motion_threshold", 8.0)
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
        Enhanced to detect human movement patterns vs. vibration.
        
        Args:
            frame: Current frame in grayscale
            
        Returns:
            Dictionary containing flow magnitude, uniformity, vibration and human detection metrics
        """
        if not self.use_optical_flow or self.prev_gray is None:
            self.prev_gray = frame.copy()
            return {"magnitude": 0.0, "uniformity": 1.0, "is_vibration": False, "likely_human": False}
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Calculate the magnitude and direction of the flow
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Get average flow magnitude
        flow_magnitude = np.mean(mag)
        
        # Calculate flow uniformity (standard deviation of angles)
        # High uniformity (low std dev) suggests whole-frame movement like vehicle motion
        # Low uniformity (high std dev) suggests localized movement like human access
        # Mask out areas with very low magnitude
        mask = mag > 0.5
        if np.any(mask):
            angles_of_interest = ang[mask]
            angle_std = np.std(angles_of_interest)
            # Normalize to 0-1 range (higher value means less uniform, more likely human activity)
            flow_uniformity = min(1.0, angle_std / np.pi)
        else:
            flow_uniformity = 0.0
            
        # Detect vibration pattern (high frequency small magnitude movements)
        is_vibration = (flow_magnitude > 0.2 and flow_magnitude < 2.0 and flow_uniformity < 0.3)
        
        # Detect human movement patterns using configurable thresholds
        # Human movement typically has:  
        # 1. Moderate to high magnitude (significant movement)
        # 2. Higher flow uniformity value (less uniform direction across frame)
        # 3. Distinct patterns different from vibrations (entering/exiting frame)
        if not self.human_detection_enabled:
            likely_human = False
        else:
            # Entering/exiting the frame usually creates significant edge changes
            # with moderate uniformity but high magnitude
            likely_human = (
                # Significant edge flow with moderate uniformity (entering/exiting frame)
                (flow_magnitude > self.flow_magnitude_threshold * 0.7 and flow_uniformity > 0.4) or 
                # Very significant motion, definitely human
                (flow_magnitude > self.significant_motion_threshold * 0.8) or
                # Unusual flow pattern different from vibration
                (flow_uniformity > 0.7 and flow_magnitude > self.flow_magnitude_threshold * 0.5)
            )
        
        # Update previous frame
        self.prev_gray = frame.copy()
        
        return {
            "magnitude": flow_magnitude,
            "uniformity": flow_uniformity,
            "is_vibration": is_vibration,
            "likely_human": likely_human
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
        
        # Calculate optical flow for motion verification
        flow_data = self._calculate_optical_flow(gray)
        flow_magnitude = flow_data["magnitude"]
        flow_uniformity = flow_data["uniformity"]
        is_vibration = flow_data["is_vibration"]
        likely_human = flow_data.get("likely_human", False)  # New human detection feature
        
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
        
        # Reduce intensity for vibration patterns
        if is_vibration:
            raw_intensity *= 0.3  # Significantly reduce intensity for vibration
            
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
            
        # Calculate detection confidence
        confidence = self.CONFIDENCE_NONE
        if stable_motion:
            # Calculate confidence level (0-3)
            if is_vibration and not likely_human:
                confidence = self.CONFIDENCE_NONE  # No confidence if this is just vibration
            elif likely_human and stable_motion:
                # Boost confidence for likely human movement
                if intensity > 50:
                    confidence = self.CONFIDENCE_HIGH
                else:
                    confidence = self.CONFIDENCE_MEDIUM
            elif stable_motion and intensity > 60:
                confidence = self.CONFIDENCE_HIGH
            elif stable_motion and intensity > 40:
                confidence = self.CONFIDENCE_MEDIUM
            elif stable_motion:
                confidence = self.CONFIDENCE_LOW
            else:
                confidence = self.CONFIDENCE_NONE
        
        # Determine final motion detection status
        motion_detected = stable_motion
        
        # Update duration tracking
        current_time = time.time()
        if motion_detected:
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
            "motion_detected": motion_detected and (not is_vibration or likely_human),
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
            "likely_human": likely_human
        }
    
    def visualize(self, frame: np.ndarray, detection_result: Dict[str, Any], show_text: bool = True) -> np.ndarray:
        """
        Visualize motion detection results on the frame.
        
        Args:
            frame: Original video frame
            detection_result: Result from detect() method
            show_text: Whether to show text overlays (set to False when used in integrated detector)
            
        Returns:
            Frame with visualization overlays
        """
        output_frame = frame.copy()
        
        # Draw ROI areas with thinner lines when used in integrated detector
        line_thickness = 1 if not show_text else 2
        
        for x, y, w, h in self.roi_areas:
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), line_thickness)
        
        # Draw motion bounding boxes with confidence-based colors
        confidence = detection_result.get("confidence", 0)
        for x, y, w, h in detection_result.get("bounding_boxes", []):
            # Color based on confidence level (red=high, orange=medium, yellow=low)
            if confidence == self.CONFIDENCE_HIGH:
                color = (0, 0, 255)  # Red
            elif confidence == self.CONFIDENCE_MEDIUM:
                color = (0, 140, 255)  # Orange
            else:
                color = (0, 255, 255)  # Yellow
                
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), color, line_thickness)
        
        # Add text information only if show_text is True
        if show_text:
            info_text = [
                f"Motion: {'Yes' if detection_result.get('motion_detected', False) else 'No'}",
                f"Intensity: {detection_result.get('intensity', 0)}",
                f"Duration: {detection_result.get('duration', 0)}s",
                f"Confidence: {detection_result.get('confidence', 0)}",
                f"{'Night' if detection_result.get('is_night_time', False) else 'Day'} Mode"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(
                    output_frame, text, (10, 30 + i*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                )
        
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
