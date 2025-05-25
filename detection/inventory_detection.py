"""
SmartVan Monitor - Inventory Change Detection Module
---------------------------------------------------
This module implements inventory change detection for the SmartVan Monitor system.

Key features:
- Capture and compare before/after inventory states
- Basic image differencing for change detection
- Item removal/addition detection
- Classification of inventory interaction events
"""

import cv2
import numpy as np
import time
import logging
import os
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("inventory_detection")

class InventoryChangeDetector:
    """Class for detecting changes in inventory areas and classifying inventory events."""
    
    # Event types
    EVENT_INVENTORY_ACCESS = "INVENTORY_ACCESS"  # General access to inventory area
    EVENT_ITEM_REMOVAL = "ITEM_REMOVAL"          # Item likely removed from inventory
    EVENT_ITEM_ADDITION = "ITEM_ADDITION"        # Item likely added to inventory
    EVENT_NO_CHANGE = "NO_CHANGE"                # Access without apparent inventory change
    
    # Change confidence levels
    CONFIDENCE_HIGH = 3
    CONFIDENCE_MEDIUM = 2
    CONFIDENCE_LOW = 1
    CONFIDENCE_NONE = 0
    
    def __init__(self, 
                output_dir: str = "output",
                change_threshold: float = 0.15,   # Percentage change required for detection
                min_change_area: int = 200,       # Minimum size of change area in pixels
                state_memory_window: int = 3600,  # How long to remember inventory states (seconds)
                config: Dict[str, Any] = None):   # Configuration dictionary
        """
        Initialize the inventory change detector.
        
        Args:
            output_dir: Directory to store inventory state images
            change_threshold: Threshold for significant pixel changes (0.0-1.0)
            min_change_area: Minimum contour area to consider as meaningful change
            state_memory_window: How long to keep inventory states in memory (seconds)
            config: Optional configuration dictionary with zone-specific settings
        """
        self.output_dir = Path(output_dir)
        self.inventory_states_dir = self.output_dir / "inventory_states"
        self.inventory_states_dir.mkdir(exist_ok=True)
        
        self.change_threshold = change_threshold
        self.min_change_area = min_change_area
        self.state_memory_window = state_memory_window
        
        # Initialize configuration dictionary
        self.config = config if config is not None else {}
        
        # Inventory state storage
        # camera_name -> zone_id -> {"image": image, "timestamp": timestamp}
        self.inventory_states = {}
        
        # Track ongoing inventory access events
        # camera_name -> {"start_time": time, "zone_id": zone_id, "start_image": image}
        self.active_access_events = {}
        
        logger.info("Inventory change detector initialized")
        logger.info(f"Change threshold: {self.change_threshold}, Min area: {self.min_change_area}")
    
    def register_inventory_zones(self, camera_name: str, zones: List[Dict[str, Any]]) -> None:
        """
        Register inventory zones for a camera.
        
        Args:
            camera_name: Name of the camera
            zones: List of zones, each with {id, name, x, y, width, height}
        """
        if camera_name not in self.inventory_states:
            self.inventory_states[camera_name] = {}
        
        for zone in zones:
            zone_id = zone["id"]
            self.inventory_states[camera_name][zone_id] = {
                "image": None,
                "timestamp": 0,
                "metadata": {
                    "name": zone.get("name", f"Zone-{zone_id}"),
                    "x": zone["x"],
                    "y": zone["y"],
                    "width": zone["width"],
                    "height": zone["height"]
                }
            }
        
        logger.info(f"Registered {len(zones)} inventory zones for camera {camera_name}")
    
    def _extract_zone_image(self, frame: np.ndarray, zone: Dict[str, Any]) -> np.ndarray:
        """
        Extract the portion of the frame corresponding to the specified zone.
        
        Args:
            frame: Full camera frame
            zone: Zone metadata with x, y, width, height
            
        Returns:
            Image of the zone area
        """
        x = zone["x"]
        y = zone["y"]
        width = zone["width"]
        height = zone["height"]
        
        # Ensure coordinates are within frame bounds
        frame_height, frame_width = frame.shape[:2]
        x = max(0, min(x, frame_width - 1))
        y = max(0, min(y, frame_height - 1))
        width = min(width, frame_width - x)
        height = min(height, frame_height - y)
        
        return frame[y:y+height, x:x+width].copy()
    
    def update_inventory_state(self, camera_name: str, frame: np.ndarray, 
                              zone_id: str, timestamp: float = None) -> None:
        """
        Update the stored state of an inventory zone.
        
        Args:
            camera_name: Name of the camera
            frame: Current camera frame
            zone_id: ID of the inventory zone
            timestamp: Current timestamp (default: current time)
        """
        if timestamp is None:
            timestamp = time.time()
        
        if camera_name not in self.inventory_states:
            logger.warning(f"Camera {camera_name} not registered for inventory tracking")
            return
        
        if zone_id not in self.inventory_states[camera_name]:
            logger.warning(f"Zone {zone_id} not registered for camera {camera_name}")
            return
        
        # Extract zone image
        zone_metadata = self.inventory_states[camera_name][zone_id]["metadata"]
        zone_image = self._extract_zone_image(frame, zone_metadata)
        
        # Update state
        self.inventory_states[camera_name][zone_id]["image"] = zone_image
        self.inventory_states[camera_name][zone_id]["timestamp"] = timestamp
        
        # Save image for reference
        zone_name = zone_metadata["name"]
        dt = datetime.fromtimestamp(timestamp)
        timestamp_str = dt.strftime("%Y%m%d_%H%M%S")
        image_filename = f"{camera_name}_{zone_name}_{timestamp_str}_state.jpg"
        image_path = os.path.join(self.inventory_states_dir, image_filename)
        cv2.imwrite(image_path, zone_image)
        
        logger.debug(f"Updated inventory state for {camera_name}, zone {zone_id}")
    
    def start_inventory_access(self, camera_name: str, frame: np.ndarray, 
                               zone_id: str, timestamp: float = None) -> Dict[str, Any]:
        """
        Start tracking an inventory access event.
        
        Args:
            camera_name: Name of the camera
            frame: Current camera frame
            zone_id: ID of the inventory zone being accessed
            timestamp: Current timestamp (default: current time)
            
        Returns:
            Event metadata
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Extract zone image
        if camera_name in self.inventory_states and zone_id in self.inventory_states[camera_name]:
            zone_metadata = self.inventory_states[camera_name][zone_id]["metadata"]
            zone_image = self._extract_zone_image(frame, zone_metadata)
            
            # Create access event
            access_key = f"{camera_name}_{zone_id}"
            self.active_access_events[access_key] = {
                "start_time": timestamp,
                "zone_id": zone_id,
                "start_image": zone_image.copy(),
                "zone_metadata": zone_metadata
            }
            
            logger.info(f"Started inventory access tracking: {camera_name}, zone {zone_id}")
            
            return {
                "event_type": self.EVENT_INVENTORY_ACCESS,
                "camera": camera_name,
                "zone_id": zone_id,
                "zone_name": zone_metadata["name"],
                "timestamp": timestamp,
                "status": "started"
            }
        else:
            logger.warning(f"Cannot start inventory access - invalid camera/zone: {camera_name}/{zone_id}")
            return {
                "event_type": self.EVENT_INVENTORY_ACCESS,
                "status": "error",
                "error": "Invalid camera or zone"
            }
    
    def end_inventory_access(self, camera_name: str, frame: np.ndarray, 
                            zone_id: str, timestamp: float = None) -> Dict[str, Any]:
        """
        End tracking an inventory access event and analyze changes.
        
        Args:
            camera_name: Name of the camera
            frame: Current camera frame
            zone_id: ID of the inventory zone being accessed
            timestamp: Current timestamp (default: current time)
            
        Returns:
            Dictionary with change detection results
        """
        if timestamp is None:
            timestamp = time.time()
        
        access_key = f"{camera_name}_{zone_id}"
        if access_key not in self.active_access_events:
            logger.warning(f"No active inventory access for {camera_name}, zone {zone_id}")
            return {
                "event_type": self.EVENT_NO_CHANGE,
                "camera": camera_name,
                "zone_id": zone_id,
                "change_detected": False,
                "confidence": self.CONFIDENCE_NONE,
                "error": "No active access event"
            }
        
        # Get access event data
        access_event = self.active_access_events[access_key]
        start_time = access_event["start_time"]
        zone_metadata = access_event["zone_metadata"]
        start_image = access_event["start_image"]
        
        # Get current zone image
        end_image = self._extract_zone_image(frame, zone_metadata)
        
        # Calculate duration
        duration = timestamp - start_time
        
        # Analyze change
        change_result = self._analyze_inventory_change(
            start_image, end_image, duration, camera_name, zone_id
        )
        
        # Remove from active events
        del self.active_access_events[access_key]
        
        # Save the before/after images
        dt = datetime.fromtimestamp(timestamp)
        timestamp_str = dt.strftime("%Y%m%d_%H%M%S")
        zone_name = zone_metadata["name"]
        
        # Save before image
        before_filename = f"{camera_name}_{zone_name}_{timestamp_str}_before.jpg"
        before_path = os.path.join(self.inventory_states_dir, before_filename)
        cv2.imwrite(before_path, start_image)
        
        # Save after image
        after_filename = f"{camera_name}_{zone_name}_{timestamp_str}_after.jpg"
        after_path = os.path.join(self.inventory_states_dir, after_filename)
        cv2.imwrite(after_path, end_image)
        
        # Save difference image if significant change
        if change_result["change_detected"]:
            diff_image = change_result.get("diff_image")
            if diff_image is not None:
                diff_filename = f"{camera_name}_{zone_name}_{timestamp_str}_diff.jpg"
                diff_path = os.path.join(self.inventory_states_dir, diff_filename)
                cv2.imwrite(diff_path, diff_image)
        
        # Add file paths to result
        change_result.update({
            "before_image_path": before_path,
            "after_image_path": after_path,
            "camera": camera_name,
            "zone_id": zone_id,
            "zone_name": zone_name,
            "start_time": start_time,
            "end_time": timestamp,
            "duration": duration,
            # Additional fields for backend rule processing
            "sync_status": "PENDING",
            "van_id": self.config.get("system", {}).get("van_id", "VAN001"),  # Include van_id for fleet identification
            "backend_rules_data": {
                "access_duration": duration,
                "change_percentage": change_result.get("change_percentage", 0),
                "change_pattern": self._categorize_change_pattern(before_image, after_image),
                "item_count_estimate": self._estimate_item_count_change(change_result),
                "concealment_indicators": self._detect_concealment_indicators(change_result)
            }
        })
        
        # Update inventory state with the new image
        self.update_inventory_state(camera_name, frame, zone_id, timestamp)
        
        logger.info(f"Ended inventory access: {camera_name}, zone {zone_id}, " +
                   f"{'change detected' if change_result['change_detected'] else 'no change'}")
        
        return change_result
    
    def _analyze_inventory_change(self, before_image: np.ndarray, after_image: np.ndarray, 
                                 duration: float, camera_name: str, zone_id: str) -> Dict[str, Any]:
        """
        Analyze changes between before and after images of inventory zone.
        Implements lighting-invariant detection and improved item classification.
        
        Args:
            before_image: Image of zone before access
            after_image: Image of zone after access
            duration: Duration of access in seconds
            camera_name: Camera name
            zone_id: Zone ID
            
        Returns:
            Dictionary with change detection results
        """
        # Ensure images are the same size
        if before_image.shape != after_image.shape:
            logger.warning("Before/after images have different dimensions - resizing")
            after_image = cv2.resize(after_image, (before_image.shape[1], before_image.shape[0]))
            
        # Get zone-specific settings if available
        zone_config = self.config.get("zones", {}).get(zone_id, {})
        zone_threshold = zone_config.get("change_threshold", self.change_threshold)
        zone_min_area = zone_config.get("min_change_area", self.min_change_area)
        
        # Apply lighting normalization techniques
        normalized_before, normalized_after = self._normalize_lighting(before_image, after_image)
        
        # Compute differences using multiple color spaces for robustness
        diff_hsv = self._compute_hsv_difference(normalized_before, normalized_after)
        diff_lab = self._compute_lab_difference(normalized_before, normalized_after)
        diff_gray = self._compute_gray_difference(normalized_before, normalized_after)
        
        # Combine differences using weighted approach
        combined_diff = self._combine_differences(diff_hsv, diff_lab, diff_gray)
        
        # Apply adaptive thresholding based on local contrast
        thresholded = cv2.adaptiveThreshold(
            combined_diff,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # Block size
            2    # Constant subtracted from mean
        )
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
        thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
        
        # Find contours of changed regions
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size (using zone-specific settings)
        significant_contours = [c for c in contours if cv2.contourArea(c) > zone_min_area]
        
        # Calculate percentage of changed pixels
        total_pixels = combined_diff.size
        changed_pixels = np.count_nonzero(thresholded)
        change_percentage = changed_pixels / total_pixels
        
        # Create visualization of differences
        diff_visualization = after_image.copy()
        
        # Identify if change is significant (using zone-specific threshold)
        change_detected = change_percentage > zone_threshold and len(significant_contours) > 0
        
        # Determine event type and confidence
        if not change_detected:
            event_type = self.EVENT_NO_CHANGE
            confidence = self.CONFIDENCE_LOW
        else:
            # Advanced item classification using color and texture features
            event_type, item_confidence = self._classify_inventory_change(
                normalized_before, normalized_after, significant_contours
            )
                
            # Draw contours on visualization with color based on event type
            color = (0, 0, 255) if event_type == self.EVENT_ITEM_ADDITION else (255, 0, 0)
            cv2.drawContours(diff_visualization, significant_contours, -1, color, 2)
            
            # Set confidence based on multiple factors
            if change_percentage > 0.3 and item_confidence > 0.7:
                confidence = self.CONFIDENCE_HIGH
            elif change_percentage > 0.2 and item_confidence > 0.5:
                confidence = self.CONFIDENCE_MEDIUM
            else:
                confidence = self.CONFIDENCE_LOW
                
            # Higher confidence for longer interactions
            if duration > 5.0 and confidence < self.CONFIDENCE_HIGH:
                confidence += 1
        
        # Prepare enhanced result with more detailed metrics
        result = {
            "event_type": event_type,
            "change_detected": change_detected,
            "confidence": confidence,
            "change_percentage": change_percentage,
            "changed_pixels": changed_pixels,
            "total_pixels": total_pixels,
            "contour_count": len(significant_contours),
            "diff_image": diff_visualization if change_detected else None,
            "zone_id": zone_id,
            "camera_name": camera_name,
            "analysis": {
                "significant_contours": len(significant_contours),
                "change_threshold": zone_threshold,  # Zone-specific threshold
                "min_change_area": zone_min_area,    # Zone-specific area
                "lighting_difference": self._calculate_lighting_difference(before_image, after_image),
                "normalized_change_score": self._calculate_normalized_change_score(change_percentage, len(significant_contours))
            }
        }
        
        return result
    
    def _normalize_lighting(self, before_image: np.ndarray, after_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply lighting normalization to make change detection robust to lighting variations.
        
        Args:
            before_image: Image before inventory access
            after_image: Image after inventory access
            
        Returns:
            Tuple of normalized before and after images
        """
        # Convert to LAB color space which separates luminance from color
        before_lab = cv2.cvtColor(before_image, cv2.COLOR_BGR2LAB)
        after_lab = cv2.cvtColor(after_image, cv2.COLOR_BGR2LAB)
        
        # Split channels
        before_l, before_a, before_b = cv2.split(before_lab)
        after_l, after_a, after_b = cv2.split(after_lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to luminance channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        before_l_eq = clahe.apply(before_l)
        after_l_eq = clahe.apply(after_l)
        
        # Merge channels back
        before_lab_eq = cv2.merge([before_l_eq, before_a, before_b])
        after_lab_eq = cv2.merge([after_l_eq, after_a, after_b])
        
        # Convert back to BGR
        before_normalized = cv2.cvtColor(before_lab_eq, cv2.COLOR_LAB2BGR)
        after_normalized = cv2.cvtColor(after_lab_eq, cv2.COLOR_LAB2BGR)
        
        return before_normalized, after_normalized
    
    def _compute_hsv_difference(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Compute difference in HSV color space, which is more robust to lighting changes.
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            Difference image (grayscale)
        """
        # Convert to HSV
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        
        # Split channels
        h1, s1, v1 = cv2.split(hsv1)
        h2, s2, v2 = cv2.split(hsv2)
        
        # Calculate differences (with circular handling for hue)
        h_diff = cv2.absdiff(h1, h2)
        h_diff = np.minimum(h_diff, 180 - h_diff)
        s_diff = cv2.absdiff(s1, s2)
        
        # Normalize and combine (ignore value channel which is most affected by lighting)
        h_diff = (h_diff / 90.0) * 255  # Hue difference (max 180) normalized
        s_diff = s_diff.astype(np.float32)
        
        # Weight hue more than saturation for color changes
        combined = (0.7 * h_diff + 0.3 * s_diff).astype(np.uint8)
        
        return combined
    
    def _compute_lab_difference(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Compute difference in LAB color space, focusing on the color channels (a,b).
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            Difference image (grayscale)
        """
        # Convert to LAB
        lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
        lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
        
        # Split channels
        _, a1, b1 = cv2.split(lab1)
        _, a2, b2 = cv2.split(lab2)
        
        # Calculate differences in a and b channels
        a_diff = cv2.absdiff(a1, a2)
        b_diff = cv2.absdiff(b1, b2)
        
        # Combine a and b differences
        combined = cv2.addWeighted(a_diff, 0.5, b_diff, 0.5, 0)
        
        return combined
    
    def _compute_gray_difference(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Compute difference in grayscale with local contrast normalization.
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            Difference image (grayscale)
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
        gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        return diff
    
    def _combine_differences(self, hsv_diff: np.ndarray, lab_diff: np.ndarray, gray_diff: np.ndarray) -> np.ndarray:
        """
        Combine differences from multiple color spaces for robust change detection.
        
        Args:
            hsv_diff: Difference in HSV space
            lab_diff: Difference in LAB space
            gray_diff: Difference in grayscale
            
        Returns:
            Combined difference image (grayscale)
        """
        # Ensure all differences are same size and type
        hsv_diff = cv2.resize(hsv_diff, (gray_diff.shape[1], gray_diff.shape[0]))
        lab_diff = cv2.resize(lab_diff, (gray_diff.shape[1], gray_diff.shape[0]))
        
        # Weighted combination (empirical weights that prioritize color differences)
        combined = cv2.addWeighted(
            hsv_diff, 0.4,
            cv2.addWeighted(lab_diff, 0.4, gray_diff, 0.2, 0),
            1.0, 0
        )
        
        return combined
    
    def _classify_inventory_change(self, before_image: np.ndarray, after_image: np.ndarray, 
                                  contours: List) -> Tuple[str, float]:
        """
        Classify inventory change as item addition or removal using advanced features.
        
        Args:
            before_image: Normalized before image
            after_image: Normalized after image
            contours: List of significant contours
            
        Returns:
            Tuple of (event_type, confidence)
        """
        # Convert to grayscale for basic analysis
        before_gray = cv2.cvtColor(before_image, cv2.COLOR_BGR2GRAY)
        after_gray = cv2.cvtColor(after_image, cv2.COLOR_BGR2GRAY)
        
        # Features to analyze
        addition_score = 0.0
        removal_score = 0.0
        
        # For each contour, analyze the region
        for contour in contours:
            # Create mask for this contour
            mask = np.zeros(before_gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            
            # Extract region stats for before and after
            before_mean = cv2.mean(before_gray, mask=mask)[0]
            after_mean = cv2.mean(after_gray, mask=mask)[0]
            
            # Calculate masked histograms for before and after
            before_hist = cv2.calcHist([before_gray], [0], mask, [32], [0, 256])
            after_hist = cv2.calcHist([after_gray], [0], mask, [32], [0, 256])
            
            # Normalize histograms
            cv2.normalize(before_hist, before_hist, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(after_hist, after_hist, 0, 1, cv2.NORM_MINMAX)
            
            # Calculate histogram difference
            hist_diff = cv2.compareHist(before_hist, after_hist, cv2.HISTCMP_CORREL)
            
            # Analyze texture complexity (variance)
            before_std = self._calculate_masked_std(before_gray, mask)
            after_std = self._calculate_masked_std(after_gray, mask)
            
            # Object has appeared: after is more textured or darker than before
            if after_std > before_std * 1.2 or after_mean < before_mean * 0.9:
                addition_score += cv2.contourArea(contour) * (1 - hist_diff)
            # Object has disappeared: before is more textured or darker than after
            elif before_std > after_std * 1.2 or before_mean < after_mean * 0.9:
                removal_score += cv2.contourArea(contour) * (1 - hist_diff)
            # General brightness change without texture change likely means lighting change, not object change
            elif abs(before_mean - after_mean) > 15 and abs(before_std - after_std) < 3:
                # Likely a lighting change, reduce both scores
                continue
            else:
                # Use brightness difference as a tiebreaker
                if after_mean < before_mean:
                    addition_score += cv2.contourArea(contour) * 0.5
                else:
                    removal_score += cv2.contourArea(contour) * 0.5
        
        # Determine final classification
        total_area = sum(cv2.contourArea(c) for c in contours)
        if total_area == 0:
            return self.EVENT_NO_CHANGE, 0.5
        
        # Normalize scores
        addition_score /= total_area
        removal_score /= total_area
        
        # Calculate confidence based on score difference
        confidence = abs(addition_score - removal_score) / max(addition_score + removal_score, 0.001)
        
        # Make final decision
        if addition_score > removal_score:
            return self.EVENT_ITEM_ADDITION, confidence
        else:
            return self.EVENT_ITEM_REMOVAL, confidence
    
    def _calculate_masked_std(self, image: np.ndarray, mask: np.ndarray) -> float:
        """
        Calculate standard deviation within a masked region.
        
        Args:
            image: Input image
            mask: Binary mask
            
        Returns:
            Standard deviation within the mask
        """
        # Get pixels within mask
        pixels = image[mask > 0]
        if len(pixels) > 0:
            return np.std(pixels)
        return 0.0
    
    def _calculate_lighting_difference(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate a metric for overall lighting difference between two images.
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            Lighting difference score (0-1)
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Calculate mean and std of each image
        mean1, std1 = cv2.meanStdDev(gray1)
        mean2, std2 = cv2.meanStdDev(gray2)
        
        # Calculate difference in means (lighting level)
        mean_diff = abs(mean1[0][0] - mean2[0][0]) / 255.0
        
        # Calculate difference in standard deviations (contrast)
        std_diff = abs(std1[0][0] - std2[0][0]) / (max(std1[0][0], std2[0][0]) + 1e-5)
        
        # Combine (more weight to mean difference)
        lighting_diff = 0.7 * mean_diff + 0.3 * std_diff
        
        return min(lighting_diff, 1.0)  # Clamp to 0-1 range
    
    def _calculate_normalized_change_score(self, change_percentage: float, contour_count: int) -> float:
        """
        Calculate a normalized change score that accounts for both area and contour count.
        
        Args:
            change_percentage: Percentage of changed pixels
            contour_count: Number of significant contours
            
        Returns:
            Normalized change score (0-1)
        """
        # Normalize contour count (empirical max of 20 contours)
        norm_contours = min(contour_count / 20.0, 1.0)
        
        # Combine (more weight to changed area)
        score = 0.7 * change_percentage + 0.3 * norm_contours
        
        return score
    
    def _categorize_change_pattern(self, before_image: np.ndarray, after_image: np.ndarray) -> str:
        """
        Categorize the pattern of change between before and after images.
        
        Args:
            before_image: Image before inventory access
            after_image: Image after inventory access
            
        Returns:
            String describing the change pattern
        """
        # Convert to grayscale for analysis
        before_gray = cv2.cvtColor(before_image, cv2.COLOR_BGR2GRAY) if len(before_image.shape) > 2 else before_image
        after_gray = cv2.cvtColor(after_image, cv2.COLOR_BGR2GRAY) if len(after_image.shape) > 2 else after_image
        
        # Calculate absolute difference
        diff = cv2.absdiff(before_gray, after_gray)
        
        # Apply threshold
        _, thresholded = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        significant_contours = [c for c in contours if cv2.contourArea(c) > self.min_change_area]
        
        # Analyze contour locations and patterns
        if len(significant_contours) == 0:
            return "NO_CHANGE"
            
        # Determine if changes are concentrated in one area or distributed
        height, width = before_gray.shape
        quadrants_with_changes = set()
        
        for c in significant_contours:
            # Get contour center
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Determine quadrant (1, 2, 3, or 4)
                quadrant = 1
                if cx <= width/2 and cy <= height/2:
                    quadrant = 1
                elif cx > width/2 and cy <= height/2:
                    quadrant = 2
                elif cx <= width/2 and cy > height/2:
                    quadrant = 3
                else:
                    quadrant = 4
                    
                quadrants_with_changes.add(quadrant)
        
        # Determine pattern based on quadrant distribution
        if len(quadrants_with_changes) == 1:
            return "SINGLE_AREA_CHANGE"
        elif len(quadrants_with_changes) == 2:
            return "DUAL_AREA_CHANGE"
        else:
            return "DISTRIBUTED_CHANGE"
    
    def _estimate_item_count_change(self, change_result: Dict[str, Any]) -> int:
        """
        Provide a rough estimate of how many items were added/removed.
        This is a simplistic approximation based on contour count and change percentage.
        
        Args:
            change_result: Result from change detection
            
        Returns:
            Estimated number of items changed (positive for additions, negative for removals)
        """
        if not change_result.get("change_detected", False):
            return 0
            
        contour_count = change_result.get("contour_count", 0)
        change_percentage = change_result.get("change_percentage", 0)
        event_type = change_result.get("event_type", self.EVENT_NO_CHANGE)
        
        # Basic heuristic: divide contour count by 2 for a conservative estimate
        # but ensure at least 1 item if change is detected
        estimated_count = max(1, contour_count // 2)
        
        # For very small changes, reduce estimate
        if change_percentage < 0.1:
            estimated_count = 1
        
        # Negative count for removals, positive for additions
        if event_type == self.EVENT_ITEM_REMOVAL:
            return -estimated_count
        elif event_type == self.EVENT_ITEM_ADDITION:
            return estimated_count
        else:
            return 0
    
    def _detect_concealment_indicators(self, change_result: Dict[str, Any]) -> List[str]:
        """
        Detect potential indicators of concealment behavior.
        
        Args:
            change_result: Result from change detection
            
        Returns:
            List of concealment indicators detected
        """
        indicators = []
        
        # Check for specific patterns that might indicate concealment
        if not change_result.get("change_detected", False):
            # No change detected but significant access duration could indicate concealment
            duration = change_result.get("duration", 0)
            if duration > 10.0:  # More than 10 seconds with no visible change
                indicators.append("LONG_ACCESS_NO_CHANGE")
        else:
            # Check for minimal change percentage that might indicate careful concealment
            change_percentage = change_result.get("change_percentage", 0)
            if change_percentage < 0.05 and change_result.get("event_type") == self.EVENT_ITEM_REMOVAL:
                indicators.append("MINIMAL_VISIBLE_CHANGE")
        
        return indicators
        
    def visualize(self, frame: np.ndarray, camera_name: str, results: Dict[str, Any] = None) -> np.ndarray:
        """
        Visualize inventory zones and change detection results.
        
        Args:
            frame: Current camera frame
            camera_name: Name of the camera
            results: Optional change detection results to visualize
            
        Returns:
            Frame with visualizations
        """
        vis_frame = frame.copy()
        
        # Draw all inventory zones for this camera
        if camera_name in self.inventory_states:
            for zone_id, zone_data in self.inventory_states[camera_name].items():
                zone = zone_data["metadata"]
                
                # Default zone color is green
                color = (0, 255, 0)
                
                # If we have active access, make zone red
                access_key = f"{camera_name}_{zone_id}"
                if access_key in self.active_access_events:
                    color = (0, 0, 255)  # Red for active access
                
                # Draw zone rectangle
                cv2.rectangle(
                    vis_frame, 
                    (zone["x"], zone["y"]), 
                    (zone["x"] + zone["width"], zone["y"] + zone["height"]), 
                    color, 
                    2
                )
                
                # Add zone name
                cv2.putText(
                    vis_frame,
                    zone["name"],
                    (zone["x"], zone["y"] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )
        
        # If we have change detection results, visualize them
        if results and results.get("change_detected", False):
            # Add change information text
            event_type = results.get("event_type", "UNKNOWN")
            confidence = results.get("confidence", 0)
            change_pct = results.get("change_percentage", 0) * 100
            
            # Position text at top of frame
            text_lines = [
                f"Change: {event_type}",
                f"Confidence: {confidence}",
                f"Changed: {change_pct:.1f}%"
            ]
            
            for i, text in enumerate(text_lines):
                cv2.putText(
                    vis_frame,
                    text,
                    (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),  # Red
                    2
                )
        
        return vis_frame
