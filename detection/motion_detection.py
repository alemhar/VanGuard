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
    
    def __init__(self, 
                algorithm: str = ALGORITHM_MOG2,
                learning_rate: float = 0.001,
                motion_threshold: int = 25,
                min_area: int = 200,
                blur_size: int = 21,
                roi_areas: Optional[List[Tuple[int, int, int, int]]] = None):
        """
        Initialize the motion detector with specified parameters.
        
        Args:
            algorithm: Background subtraction algorithm to use ("MOG2" or "KNN")
            learning_rate: How quickly the background model adapts to changes
            motion_threshold: Threshold for distinguishing foreground from background
            min_area: Minimum contour area to be considered motion
            blur_size: Size of Gaussian blur kernel for noise reduction
            roi_areas: List of ROI areas as (x, y, width, height) tuples
        """
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.motion_threshold = motion_threshold
        self.min_area = min_area
        self.blur_size = blur_size
        self.roi_areas = roi_areas or []
        
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
        
        self.last_motion_time = None
        self.motion_duration = 0
        self.is_motion_active = False
        
        logger.info(f"Motion detector initialized with {algorithm} algorithm")
    
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
            
        # Create mask of same size as frame, initialize to zeros (all black)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # Fill ROI areas with white (255)
        for x, y, w, h in self.roi_areas:
            mask[y:y+h, x:x+w] = 255
            
        # Apply mask to frame
        return cv2.bitwise_and(frame, frame, mask=mask)
    
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect motion in the current frame.
        
        Args:
            frame: Current video frame
            
        Returns:
            Dictionary containing motion information:
                - motion_detected: Boolean indicating if motion was detected
                - bounding_boxes: List of (x,y,w,h) for motion regions
                - intensity: Motion intensity score (0-100)
                - duration: Duration of continuous motion in seconds
                - mask: Binary mask showing motion areas
        """
        # Apply ROI mask if defined
        roi_frame = self.apply_roi_mask(frame)
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(
            blurred, 
            learningRate=self.learning_rate
        )
        
        # Threshold the mask to remove shadows (usually values of 127)
        _, thresh = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(
            thresh.copy(), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by minimum area
        significant_contours = [c for c in contours if cv2.contourArea(c) > self.min_area]
        
        # Get bounding boxes for significant contours
        bounding_boxes = [cv2.boundingRect(c) for c in significant_contours]
        
        # Calculate motion intensity (0-100)
        white_pixel_count = np.count_nonzero(thresh)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        intensity = min(100, int((white_pixel_count / total_pixels) * 1000))
        
        # Update motion duration
        current_time = time.time()
        motion_detected = len(significant_contours) > 0
        
        if motion_detected:
            if self.is_motion_active:
                # Continue existing motion
                self.motion_duration = current_time - self.last_motion_time
            else:
                # Start new motion
                self.is_motion_active = True
                self.last_motion_time = current_time
                self.motion_duration = 0
        else:
            if self.is_motion_active and current_time - self.last_motion_time > 1.0:
                # End motion after 1 second of no detection
                self.is_motion_active = False
                self.motion_duration = 0
        
        return {
            "motion_detected": motion_detected,
            "bounding_boxes": bounding_boxes,
            "intensity": intensity,
            "duration": round(self.motion_duration, 2) if self.is_motion_active else 0,
            "mask": thresh,
        }
    
    def visualize(self, frame: np.ndarray, detection_result: Dict[str, Any]) -> np.ndarray:
        """
        Visualize motion detection results on the frame.
        
        Args:
            frame: Original video frame
            detection_result: Result from detect() method
            
        Returns:
            Frame with visualization overlays
        """
        output_frame = frame.copy()
        
        # Draw ROI areas
        for x, y, w, h in self.roi_areas:
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw motion bounding boxes
        for x, y, w, h in detection_result["bounding_boxes"]:
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # Add text information
        info_text = [
            f"Motion: {'Yes' if detection_result['motion_detected'] else 'No'}",
            f"Intensity: {detection_result['intensity']}",
            f"Duration: {detection_result['duration']}s",
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
