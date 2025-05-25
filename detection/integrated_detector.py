"""
SmartVan Monitor - Integrated Detection Module
---------------------------------------------
This module combines motion detection and YOLO object detection
to provide a comprehensive detection system that is both efficient
and accurate for the SmartVan Monitor.
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any

from .motion_detection import MotionDetector
from .object_detection.yolo_detector import YOLODetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("integrated_detection")

class IntegratedDetector:
    """
    Integrated detection system that combines motion detection with YOLO object detection.
    Uses motion detection as a pre-filter for the more computationally intensive YOLO detection.
    """
    
    def __init__(self, 
                 motion_detector: MotionDetector,
                 yolo_config: Optional[Dict[str, Any]] = None,
                 always_run_yolo: bool = False,
                 yolo_interval: int = 5,  # Run YOLO every X frames
                 human_confirmation_needed: bool = True,
                 object_recording_threshold: float = 0.6):
        """
        Initialize the integrated detector.
        
        Args:
            motion_detector: Existing motion detector instance
            yolo_config: Configuration for YOLO detector
            always_run_yolo: Whether to run YOLO on every frame regardless of motion
            yolo_interval: Run YOLO every X frames when no motion is detected
            human_confirmation_needed: Whether YOLO confirmation is needed for human detection
            object_recording_threshold: Confidence threshold for recording object detection events
        """
        self.motion_detector = motion_detector
        self.always_run_yolo = always_run_yolo
        self.yolo_interval = yolo_interval
        self.human_confirmation_needed = human_confirmation_needed
        self.object_recording_threshold = object_recording_threshold
        
        # Frame counter for YOLO interval
        self.frame_counter = 0
        
        # Default YOLO configuration if none provided
        if yolo_config is None:
            yolo_config = {
                "model_size": "tiny",
                "confidence_threshold": 0.5,
                "nms_threshold": 0.4,
                "only_on_motion": not always_run_yolo,
                "gpu_enabled": False
            }
        
        # Initialize YOLO detector
        self.yolo_detector = YOLODetector(**yolo_config)
        
        # Performance tracking
        self.last_detection_time = 0
        self.avg_detection_time = 0
        self.detection_count = 0
        
        logger.info("Integrated detector initialized")
        logger.info(f"YOLO will run on {'all frames' if always_run_yolo else 'motion detection'}")
        if not always_run_yolo:
            logger.info(f"YOLO will also run every {yolo_interval} frames regardless of motion")
    
    def detect(self, frame: np.ndarray, timestamp: float = None) -> Dict[str, Any]:
        """
        Perform integrated detection on the current frame.
        
        Args:
            frame: Current video frame
            timestamp: Current timestamp (if None, current time will be used)
            
        Returns:
            Dictionary containing integrated detection results
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Start timing detection
        start_time = time.time()
        
        # Increment frame counter
        self.frame_counter += 1
        
        # Step 1: Run motion detection
        motion_result = self.motion_detector.detect(frame)
        
        # Default YOLO result (for when we skip YOLO)
        yolo_result = {
            "objects_detected": [],
            "person_detected": False,
            "boxes": [],
            "classes": [],
            "confidences": [],
            "inference_time": 0,
            "yolo_skipped": True
        }
        
        # Step 2: Determine if we should run YOLO
        run_yolo = False
        
        # Run YOLO if:
        # 1. Always run YOLO is enabled
        # 2. Motion is detected
        # 3. Motion detector thinks it might be a human
        # 4. We've reached the YOLO interval frame count
        if (self.always_run_yolo or 
            motion_result.get("motion_detected", False) or 
            motion_result.get("likely_human", False) or 
            self.frame_counter % self.yolo_interval == 0):
            run_yolo = True
            
        # Step 3: Run YOLO if needed
        if run_yolo:
            yolo_result = self.yolo_detector.detect(frame, motion_result)
            
            # Reset frame counter if we ran YOLO
            if self.frame_counter % self.yolo_interval == 0:
                self.frame_counter = 0
        
        # Step 4: Integrate results
        integrated_result = self._integrate_results(motion_result, yolo_result, timestamp)
        
        # Calculate total detection time
        detection_time = time.time() - start_time
        
        # Update performance metrics
        self.last_detection_time = detection_time
        self.detection_count += 1
        self.avg_detection_time = ((self.avg_detection_time * (self.detection_count - 1)) + 
                                  detection_time) / self.detection_count
        
        # Add detection time to result
        integrated_result["detection_time"] = detection_time
        
        return integrated_result
    
    def _integrate_results(self, 
                          motion_result: Dict[str, Any], 
                          yolo_result: Dict[str, Any],
                          timestamp: float) -> Dict[str, Any]:
        """
        Integrate motion detection and YOLO results.
        
        Args:
            motion_result: Result from motion detector
            yolo_result: Result from YOLO detector
            timestamp: Current timestamp
            
        Returns:
            Dictionary containing integrated detection results
        """
        # Extract key information
        motion_detected = motion_result.get("motion_detected", False)
        likely_human_motion = motion_result.get("likely_human", False)
        person_detected_yolo = yolo_result.get("person_detected", False)
        yolo_skipped = yolo_result.get("yolo_skipped", True)
        objects_detected = yolo_result.get("objects_detected", [])
        motion_confidence = motion_result.get("confidence", 0)
        
        # Determine human presence
        human_detected = False
        human_confidence = 0.0
        
        if self.human_confirmation_needed:
            # Require both motion detection and YOLO to confirm human
            if likely_human_motion and person_detected_yolo:
                human_detected = True
                # Average confidences
                motion_human_conf = motion_result.get("human_confidence", 0.5)
                
                # Find highest person confidence from YOLO
                yolo_human_conf = 0
                for obj in objects_detected:
                    if obj["class"] == "person" and obj["confidence"] > yolo_human_conf:
                        yolo_human_conf = obj["confidence"]
                
                # Weighted average (give more weight to YOLO)
                human_confidence = (motion_human_conf * 0.3) + (yolo_human_conf * 0.7)
            elif likely_human_motion and yolo_skipped:
                # If YOLO was skipped, use motion detection result with lower confidence
                human_detected = True
                human_confidence = motion_result.get("human_confidence", 0.5) * 0.6
        else:
            # Either detection method can confirm human presence
            if likely_human_motion or person_detected_yolo:
                human_detected = True
                
                # Calculate confidence based on available information
                if likely_human_motion and person_detected_yolo:
                    # Both methods detected human, higher confidence
                    motion_human_conf = motion_result.get("human_confidence", 0.5)
                    
                    # Find highest person confidence from YOLO
                    yolo_human_conf = 0
                    for obj in objects_detected:
                        if obj["class"] == "person" and obj["confidence"] > yolo_human_conf:
                            yolo_human_conf = obj["confidence"]
                    
                    # Weighted average
                    human_confidence = (motion_human_conf * 0.3) + (yolo_human_conf * 0.7)
                elif likely_human_motion:
                    # Only motion detection found human
                    human_confidence = motion_result.get("human_confidence", 0.5) * 0.6
                else:
                    # Only YOLO found human
                    for obj in objects_detected:
                        if obj["class"] == "person" and obj["confidence"] > human_confidence:
                            human_confidence = obj["confidence"]
        
        # Determine if this should be recorded as an event
        should_record = False
        if human_detected and human_confidence >= self.object_recording_threshold:
            should_record = True
        elif motion_detected and motion_confidence >= 0.7:
            should_record = True
        
        # Build integrated result
        result = {
            # Core detection results
            "motion_detected": motion_detected,
            "human_detected": human_detected,
            "objects_detected": objects_detected,
            "timestamp": timestamp,
            
            # Confidence scores
            "motion_confidence": motion_confidence,
            "human_confidence": human_confidence,
            
            # YOLO specific data
            "person_detected_yolo": person_detected_yolo,
            "yolo_ran": not yolo_skipped,
            
            # Motion specific data
            "likely_human_motion": likely_human_motion,
            "motion_vector": motion_result.get("motion_vector", None),
            "contours": motion_result.get("contours", None),
            
            # Recording decision
            "should_record": should_record,
            
            # Raw results for reference
            "motion_result": motion_result,
            "yolo_result": yolo_result
        }
        
        return result
    
    def visualize(self, frame: np.ndarray, detection_result: Dict[str, Any]) -> np.ndarray:
        """
        Visualize integrated detection results on the frame.
        
        Args:
            frame: Current video frame
            detection_result: Result from detect method
            
        Returns:
            Frame with detection visualizations
        """
        # Start with motion visualization
        motion_result = detection_result.get("motion_result", {})
        vis_frame = self.motion_detector.visualize(frame, motion_result)
        
        # Add YOLO visualization if YOLO was run
        yolo_result = detection_result.get("yolo_result", {})
        yolo_skipped = yolo_result.get("yolo_skipped", True)
        
        if not yolo_skipped:
            vis_frame = self.yolo_detector.visualize(vis_frame, yolo_result)
        
        # Get frame dimensions for better text positioning
        height, width = vis_frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        line_height = 30  # Space between text lines
        
        # Draw semi-transparent background for status text area
        status_bg_height = 150
        status_overlay = vis_frame.copy()
        cv2.rectangle(status_overlay, (0, 0), (width, status_bg_height), (0, 0, 0), -1)
        cv2.addWeighted(status_overlay, 0.3, vis_frame, 0.7, 0, vis_frame)
        
        # Starting position for text
        text_y = 30
        
        # Add motion information
        motion_detected = detection_result.get("motion_detected", False)
        intensity = detection_result.get("intensity", 0)
        motion_text = f"Motion: {'YES' if motion_detected else 'NO'} | Intensity: {intensity:.1f}"
        cv2.putText(vis_frame, motion_text, (10, text_y), 
                  font, font_scale, (255, 255, 255), font_thickness)
        text_y += line_height
        
        # Add human detection information
        human_detected = detection_result.get("human_detected", False)
        human_confidence = detection_result.get("human_confidence", 0)
        
        # Color based on confidence (red if high confidence)
        human_color = (0, 0, 255) if human_detected and human_confidence > 0.7 else (0, 255, 0)
        human_text = f"Human: {'YES' if human_detected else 'NO'} | Confidence: {human_confidence:.2f}"
        cv2.putText(vis_frame, human_text, (10, text_y), 
                  font, font_scale, human_color, font_thickness)
        text_y += line_height
        
        # Add recording status
        should_record = detection_result.get("should_record", False)
        if should_record:
            cv2.putText(vis_frame, "RECORDING", (10, text_y), 
                      font, font_scale, (0, 0, 255), font_thickness)
            text_y += line_height
        
        # Add YOLO information
        yolo_ran = not yolo_result.get("yolo_skipped", True)
        yolo_text = f"YOLO: {'Active' if yolo_ran else 'Inactive'}"
        if yolo_ran:
            inference_time = yolo_result.get("inference_time", 0)
            yolo_text += f" | Time: {inference_time:.3f}s"
        cv2.putText(vis_frame, yolo_text, (10, text_y), 
                  font, font_scale, (255, 255, 0), font_thickness)
        text_y += line_height
        
        # Add objects detected if any
        objects = detection_result.get("objects_detected", [])
        if objects:
            obj_classes = [obj["class"] for obj in objects]
            obj_count = len(objects)
            obj_text = f"Objects: {obj_count} ({', '.join(obj_classes[:3])}{'...' if len(obj_classes) > 3 else ''})"
            cv2.putText(vis_frame, obj_text, (10, text_y), 
                      font, font_scale, (0, 255, 255), font_thickness)
        
        # Add total detection time at the bottom of the frame
        detection_time = detection_result.get("detection_time", 0)
        cv2.putText(vis_frame, f"Detection: {detection_time:.3f}s", (10, height - 20), 
                  font, font_scale, (0, 255, 0), font_thickness)
        
        return vis_frame
