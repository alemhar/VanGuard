"""
SmartVan Monitor - Camera Utilities
----------------------------------
Utilities for managing camera connections and video streams.
"""

import cv2
import time
import logging
import threading
from typing import Dict, Optional, Union, Tuple, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("camera_utils")

class Camera:
    """Class for managing camera connections and video streams."""
    
    def __init__(self, camera_id: Union[int, str], name: str = "", 
                 resolution: Tuple[int, int] = (640, 480),
                 fps: int = 15):
        """
        Initialize camera connection.
        
        Args:
            camera_id: Camera index (0, 1, etc.) or path to video file/stream URL
            name: Descriptive name for the camera
            resolution: Desired resolution as (width, height)
            fps: Desired frames per second
        """
        self.camera_id = camera_id
        self.name = name or f"Camera_{camera_id}"
        self.resolution = resolution
        self.fps = fps
        self.cap = None
        self.is_running = False
        self.last_frame = None
        self.last_frame_time = 0
        self.frame_count = 0
        self.actual_fps = 0
        self._capture_thread = None
        self._lock = threading.Lock()
        
        logger.info(f"Initialized {self.name} with resolution {resolution} at {fps} FPS")
    
    def connect(self) -> bool:
        """
        Connect to the camera.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera: {self.camera_id}")
                return False
                
            # Try to set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Read actual properties
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Connected to {self.name} with resolution " 
                       f"{actual_width}x{actual_height} at {actual_fps} FPS")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to camera {self.camera_id}: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the camera."""
        self.stop_capture()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info(f"Disconnected from {self.name}")
    
    def start_capture(self) -> bool:
        """
        Start capturing frames in a background thread.
        
        Returns:
            True if capture started successfully, False otherwise
        """
        if self.is_running:
            logger.warning(f"{self.name} capture already running")
            return True
            
        if self.cap is None or not self.cap.isOpened():
            if not self.connect():
                return False
        
        self.is_running = True
        self._capture_thread = threading.Thread(target=self._capture_loop)
        self._capture_thread.daemon = True
        self._capture_thread.start()
        logger.info(f"Started capture thread for {self.name}")
        return True
    
    def stop_capture(self) -> None:
        """Stop the frame capture thread."""
        self.is_running = False
        if self._capture_thread is not None:
            if self._capture_thread.is_alive():
                self._capture_thread.join(timeout=1.0)
            self._capture_thread = None
            logger.info(f"Stopped capture thread for {self.name}")
    
    def _capture_loop(self) -> None:
        """Background thread function to continuously capture frames."""
        last_fps_time = time.time()
        frames_since_last_fps_calc = 0
        
        while self.is_running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    logger.warning(f"{self.name} disconnected, attempting to reconnect")
                    if not self.connect():
                        time.sleep(1)  # Wait before retrying
                        continue
                
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning(f"Failed to read frame from {self.name}")
                    time.sleep(0.1)
                    continue
                
                current_time = time.time()
                
                # Update FPS calculation every second
                frames_since_last_fps_calc += 1
                if current_time - last_fps_time >= 1.0:
                    self.actual_fps = frames_since_last_fps_calc / (current_time - last_fps_time)
                    frames_since_last_fps_calc = 0
                    last_fps_time = current_time
                
                # Update frame data with thread safety
                with self._lock:
                    self.last_frame = frame
                    self.last_frame_time = current_time
                    self.frame_count += 1
                
                # Regulate capture rate
                sleep_time = 1.0/self.fps - (time.time() - current_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                logger.error(f"Error in capture loop for {self.name}: {e}")
                time.sleep(0.1)  # Prevent tight loop on error
    
    def get_frame(self) -> Tuple[Optional[float], Optional[int], Optional[float]]:
        """
        Get the most recent frame from the camera.
        
        Returns:
            Tuple of (frame, timestamp, fps) or None values if no frame available
        """
        with self._lock:
            if self.last_frame is None:
                return None, None, None
            return self.last_frame.copy(), self.last_frame_time, self.actual_fps

class CameraManager:
    """Manager for multiple camera connections."""
    
    def __init__(self):
        """Initialize the camera manager."""
        self.cameras = {}
        logger.info("Camera manager initialized")
    
    def add_camera(self, camera_id: Union[int, str], name: str = "",
                  resolution: Tuple[int, int] = (640, 480),
                  fps: int = 15) -> str:
        """
        Add a camera to the manager.
        
        Args:
            camera_id: Camera index or connection string
            name: Optional descriptive name
            resolution: Desired resolution
            fps: Desired FPS
            
        Returns:
            Camera name (either provided or auto-generated)
        """
        camera_name = name or f"Camera_{camera_id}"
        
        if camera_name in self.cameras:
            logger.warning(f"Camera {camera_name} already exists, overwriting")
            self.remove_camera(camera_name)
        
        camera = Camera(camera_id, camera_name, resolution, fps)
        self.cameras[camera_name] = camera
        logger.info(f"Added camera {camera_name}")
        return camera_name
    
    def remove_camera(self, camera_name: str) -> bool:
        """
        Remove and disconnect a camera.
        
        Args:
            camera_name: Name of the camera to remove
            
        Returns:
            True if camera was removed, False if not found
        """
        if camera_name in self.cameras:
            camera = self.cameras[camera_name]
            camera.disconnect()
            del self.cameras[camera_name]
            logger.info(f"Removed camera {camera_name}")
            return True
        else:
            logger.warning(f"Camera {camera_name} not found")
            return False
    
    def start_all_cameras(self) -> None:
        """Start capturing from all cameras."""
        for camera_name, camera in self.cameras.items():
            success = camera.start_capture()
            if success:
                logger.info(f"Started {camera_name}")
            else:
                logger.error(f"Failed to start {camera_name}")
    
    def stop_all_cameras(self) -> None:
        """Stop capturing from all cameras."""
        for camera_name, camera in self.cameras.items():
            camera.stop_capture()
        logger.info("Stopped all cameras")
    
    def get_frame(self, camera_name: str) -> Tuple[Optional[float], Optional[int], Optional[float]]:
        """
        Get the latest frame from a specific camera.
        
        Args:
            camera_name: Name of the camera
            
        Returns:
            Tuple of (frame, timestamp, fps) or None values if camera not found or no frame
        """
        if camera_name in self.cameras:
            return self.cameras[camera_name].get_frame()
        else:
            logger.warning(f"Camera {camera_name} not found")
            return None, None, None
    
    def get_all_frames(self) -> Dict[str, Tuple[Optional[float], Optional[int], Optional[float]]]:
        """
        Get the latest frames from all cameras.
        
        Returns:
            Dictionary of camera_name -> (frame, timestamp, fps)
        """
        frames = {}
        for camera_name, camera in self.cameras.items():
            frames[camera_name] = camera.get_frame()
        return frames
        
    def cleanup(self) -> None:
        """Disconnect and clean up all camera resources."""
        for camera_name, camera in self.cameras.items():
            camera.disconnect()
        self.cameras.clear()
        logger.info("Cleaned up all camera resources")
