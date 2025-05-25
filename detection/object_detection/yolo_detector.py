"""
SmartVan Monitor - YOLO Object Detection Module
----------------------------------------------
This module implements YOLO-based object detection for the SmartVan Monitor system.
It works alongside the motion detection to provide specific object recognition
while maintaining efficiency on embedded hardware.
"""

import cv2
import numpy as np
import os
import time
import logging
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("yolo_detection")

class YOLODetector:
    """Class for YOLO-based object detection that integrates with motion detection."""
    
    # Default confidence thresholds
    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4
    
    # Class labels we're interested in (COCO dataset)
    PERSON_CLASS_ID = 0  # Person in COCO dataset
    CLASSES_OF_INTEREST = {
        0: "person",
        24: "backpack",
        26: "handbag",
        28: "suitcase",
        39: "bottle",
        41: "cup",
        67: "cell phone",
        73: "book",
        # Add more classes as needed
    }
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 weights_path: Optional[str] = None,
                 model_size: str = "tiny",  # "tiny", "medium", or "large"
                 confidence_threshold: float = CONFIDENCE_THRESHOLD,
                 nms_threshold: float = NMS_THRESHOLD,
                 only_on_motion: bool = True,
                 gpu_enabled: bool = False,
                 ):
        """
        Initialize YOLO detector with specified parameters.
        
        Args:
            config_path: Path to YOLO config file (darknet cfg)
            weights_path: Path to YOLO weights file
            model_size: Size of YOLO model ("tiny", "medium", "large")
            confidence_threshold: Minimum confidence for detection
            nms_threshold: Non-maximum suppression threshold
            only_on_motion: Only run YOLO when motion is detected
            gpu_enabled: Whether to use GPU acceleration if available
        """
        self.config_path = config_path
        self.weights_path = weights_path
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.only_on_motion = only_on_motion
        self.gpu_enabled = gpu_enabled
        
        # Initialize YOLO model
        self.net = None
        self.output_layers = None
        self.input_height = 416  # Default input size
        self.input_width = 416   # Default input size
        
        # Get model paths if not provided
        if not self.config_path or not self.weights_path:
            self._get_model_paths()
            
        # Initialize model
        self._initialize_model()
        
        # Performance tracking
        self.last_inference_time = 0
        self.avg_inference_time = 0
        self.inference_count = 0
        
        logger.info(f"YOLO Detector initialized with {model_size} model")
        if self.only_on_motion:
            logger.info("YOLO will only run when motion is detected (energy efficient mode)")
        
    def _get_model_paths(self) -> None:
        """
        Set paths to YOLO model files based on selected model size.
        Automatically downloads models if they don't exist.
        """
        # Create models directory if it doesn't exist
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Set paths based on model size
        if self.model_size == "tiny":
            self.config_path = os.path.join(models_dir, "yolov4-tiny.cfg")
            self.weights_path = os.path.join(models_dir, "yolov4-tiny.weights")
            self.input_height = 416
            self.input_width = 416
        elif self.model_size == "medium":
            self.config_path = os.path.join(models_dir, "yolov4.cfg")
            self.weights_path = os.path.join(models_dir, "yolov4.weights")
            self.input_height = 608
            self.input_width = 608
        elif self.model_size == "large":
            self.config_path = os.path.join(models_dir, "yolov4-p6.cfg")
            self.weights_path = os.path.join(models_dir, "yolov4-p6.weights")
            self.input_height = 896
            self.input_width = 896
        else:
            # Default to tiny if invalid model size
            logger.warning(f"Invalid model size: {self.model_size}. Using 'tiny' instead.")
            self.model_size = "tiny"
            self.config_path = os.path.join(models_dir, "yolov4-tiny.cfg")
            self.weights_path = os.path.join(models_dir, "yolov4-tiny.weights")
            self.input_height = 416
            self.input_width = 416
            
        # Check if model files exist and download if needed
        self._download_models_if_needed()
        
    def _download_models_if_needed(self) -> None:
        """
        Download YOLO model files if they don't exist locally.
        """
        # Define model URLs based on model size
        model_urls = {
            "tiny": {
                "cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
                "weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
            },
            "medium": {
                "cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg",
                "weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4.weights"
            },
            "large": {
                "cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-p6.cfg",
                "weights": "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-p6.weights"
            }
        }
        
        # Download config file if it doesn't exist
        if not os.path.exists(self.config_path):
            logger.info(f"Downloading {self.model_size} YOLO config file...")
            import urllib.request
            urllib.request.urlretrieve(model_urls[self.model_size]["cfg"], self.config_path)
            logger.info(f"Downloaded config file to {self.config_path}")
            
        # Download weights file if it doesn't exist (this may take a while)
        if not os.path.exists(self.weights_path):
            logger.info(f"Downloading {self.model_size} YOLO weights file (this may take a while)...")
            import urllib.request
            urllib.request.urlretrieve(model_urls[self.model_size]["weights"], self.weights_path)
            logger.info(f"Downloaded weights file to {self.weights_path}")
    
    def _initialize_model(self) -> None:
        """
        Initialize the YOLO neural network model.
        """
        # Load YOLO network
        self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
        
        # Configure backend
        if self.gpu_enabled:
            # Try to use GPU if available
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            logger.info("Using GPU acceleration for YOLO inference")
        else:
            # Use CPU
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            logger.info("Using CPU for YOLO inference")
        
        # Get output layers
        layer_names = self.net.getLayerNames()
        try:
            # OpenCV 4.5.4+
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        except:
            # Older OpenCV versions
            self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
    
    def detect(self, 
              frame: np.ndarray, 
              motion_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect objects in the current frame.
        
        Args:
            frame: Current video frame
            motion_result: Result from motion detector (optional)
            
        Returns:
            Dictionary containing detection information:
                - objects_detected: List of detected objects
                - person_detected: Boolean indicating if a person was detected
                - boxes: List of bounding boxes for detected objects
                - classes: List of class IDs for detected objects
                - confidences: List of confidence scores for detected objects
                - inference_time: Time taken for YOLO inference
        """
        # Check if we should run YOLO based on motion detection
        if self.only_on_motion and motion_result:
            motion_detected = motion_result.get("motion_detected", False)
            likely_human = motion_result.get("likely_human", False)
            
            # Skip YOLO if no motion detected and not likely human
            if not motion_detected and not likely_human:
                return {
                    "objects_detected": [],
                    "person_detected": False,
                    "boxes": [],
                    "classes": [],
                    "confidences": [],
                    "inference_time": 0,
                    "yolo_skipped": True,
                    "motion_detected": motion_detected,
                    "likely_human": likely_human
                }
        
        # Start timing inference
        start_time = time.time()
        
        # Prepare image for YOLO
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (self.input_width, self.input_height), 
                                     swapRB=True, crop=False)
        
        # Set input and run forward pass
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        # Process outputs
        boxes = []
        confidences = []
        class_ids = []
        
        # Process each output layer
        for output in outputs:
            # Process each detection
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter by confidence and classes of interest
                if confidence > self.confidence_threshold and class_id in self.CLASSES_OF_INTEREST:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        # Prepare results
        final_boxes = []
        final_confidences = []
        final_class_ids = []
        objects_detected = []
        person_detected = False
        
        # Process NMS results
        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append(boxes[i])
                final_confidences.append(confidences[i])
                final_class_ids.append(class_ids[i])
                
                # Add to objects list
                obj_class = self.CLASSES_OF_INTEREST[class_ids[i]]
                objects_detected.append({
                    "class": obj_class,
                    "confidence": confidences[i],
                    "box": boxes[i]
                })
                
                # Check if person detected
                if class_ids[i] == self.PERSON_CLASS_ID:
                    person_detected = True
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        # Update performance metrics
        self.last_inference_time = inference_time
        self.inference_count += 1
        self.avg_inference_time = ((self.avg_inference_time * (self.inference_count - 1)) + 
                                 inference_time) / self.inference_count
        
        # Return detection results
        return {
            "objects_detected": objects_detected,
            "person_detected": person_detected,
            "boxes": final_boxes,
            "classes": final_class_ids,
            "confidences": final_confidences,
            "inference_time": inference_time,
            "yolo_skipped": False
        }
    
    def visualize(self, frame: np.ndarray, detection_result: Dict[str, Any], show_text: bool = True) -> np.ndarray:
        """
        Visualize detection results on the frame.
        
        Args:
            frame: Current video frame
            detection_result: Result from detect method
            show_text: Whether to show text overlays (set to False when used in integrated detector)
            
        Returns:
            Frame with detection visualizations
        """
        # Skip if YOLO was skipped
        if detection_result.get("yolo_skipped", False):
            return frame
        
        # Create a copy of the frame
        vis_frame = frame.copy()
        
        # Draw boxes and labels
        boxes = detection_result.get("boxes", [])
        classes = detection_result.get("classes", [])
        confidences = detection_result.get("confidences", [])
        
        # Set line thickness based on show_text
        line_thickness = 1 if not show_text else 2
        
        for i in range(len(boxes)):
            # Get box coordinates
            x, y, w, h = boxes[i]
            
            # Get class and confidence
            class_id = classes[i]
            confidence = confidences[i]
            class_name = self.CLASSES_OF_INTEREST.get(class_id, "unknown")
            
            # Select color based on class
            if class_id == self.PERSON_CLASS_ID:
                color = (0, 0, 255)  # Red for people
            else:
                color = (255, 0, 0)  # Blue for other objects
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, line_thickness)
            
            # Draw label only if show_text is True
            if show_text:
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(vis_frame, label, (x, y - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add inference time only if show_text is True
        if show_text:
            inference_time = detection_result.get("inference_time", 0)
            cv2.putText(vis_frame, f"YOLO Inference: {inference_time:.3f}s", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return vis_frame
