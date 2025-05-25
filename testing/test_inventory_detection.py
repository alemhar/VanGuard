"""
SmartVan Monitor - Inventory Detection Tests
-------------------------------------------
Tests the inventory change detection capabilities, with focus on:
1. Lighting-invariant detection accuracy
2. Item addition/removal classification
3. Zone-specific sensitivity settings
"""

import cv2
import numpy as np
import os
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from testing.test_framework import TestResult, framework
from detection.inventory_detection import InventoryChangeDetector

# Configure logging
logger = logging.getLogger("inventory_detection_test")

class InventoryDetectionTests:
    """Tests for the inventory change detection system"""
    
    def __init__(self, test_data_dir: str = "testing/test_data/inventory"):
        """
        Initialize inventory detection tests.
        
        Args:
            test_data_dir: Directory containing test data
        """
        self.test_data_dir = Path(test_data_dir)
        self.test_data_dir.mkdir(exist_ok=True, parents=True)
        
        # Standard test configuration
        self.config = {
            "system": {
                "van_id": "TEST_VAN"
            },
            "zones": {
                "test_zone_standard": {
                    "change_threshold": 0.15,
                    "min_change_area": 200
                },
                "test_zone_sensitive": {
                    "change_threshold": 0.08,
                    "min_change_area": 100
                },
                "test_zone_tolerant": {
                    "change_threshold": 0.25,
                    "min_change_area": 300
                }
            }
        }
        
        # Create test detector instance
        self.detector = InventoryChangeDetector(
            output_dir="testing/output",
            change_threshold=0.15,
            min_change_area=200,
            config=self.config
        )
        
        # Register test zones
        self._register_test_zones()
        
        logger.info("Inventory detection tests initialized")
        
    def _register_test_zones(self):
        """Register test zones with the detector"""
        zones = [
            {"id": "test_zone_standard", "name": "Standard Zone", "x": 100, "y": 100, "width": 400, "height": 300},
            {"id": "test_zone_sensitive", "name": "Sensitive Zone", "x": 100, "y": 450, "width": 400, "height": 300},
            {"id": "test_zone_tolerant", "name": "Tolerant Zone", "x": 550, "y": 100, "width": 400, "height": 300}
        ]
        
        self.detector.register_inventory_zones("test_camera", zones)
    
    def _generate_synthetic_before_after(self, 
                                        lighting_change: float = 0.0, 
                                        add_item: bool = False,
                                        remove_item: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic before/after images for testing.
        
        Args:
            lighting_change: Amount of lighting change (-1.0 to 1.0)
            add_item: Whether to add an item in the after image
            remove_item: Whether to remove an item in the before image
            
        Returns:
            Tuple of (before_image, after_image)
        """
        # Create base image (shelf with items)
        img_size = (800, 600, 3)
        before = np.ones(img_size, dtype=np.uint8) * 200  # Gray background
        
        # Add shelf lines
        for y in range(100, 501, 100):
            cv2.line(before, (50, y), (750, y), (120, 120, 120), 5)
        
        # Add standard items (boxes of different colors)
        items = [
            {"pos": (100, 150), "size": (80, 60), "color": (50, 50, 200)},   # Blue box
            {"pos": (220, 150), "size": (60, 80), "color": (50, 200, 50)},   # Green box
            {"pos": (330, 150), "size": (70, 70), "color": (200, 50, 50)},   # Red box
            {"pos": (450, 150), "size": (90, 50), "color": (200, 200, 50)},  # Yellow box
            {"pos": (600, 150), "size": (75, 65), "color": (200, 50, 200)},  # Purple box
            
            {"pos": (150, 280), "size": (85, 55), "color": (50, 150, 200)},  # Light blue box
            {"pos": (280, 280), "size": (65, 75), "color": (150, 200, 50)},  # Light green box
            {"pos": (390, 280), "size": (75, 65), "color": (200, 150, 50)},  # Orange box
            {"pos": (520, 280), "size": (95, 45), "color": (100, 100, 200)}, # Another blue box
            
            {"pos": (200, 400), "size": (80, 60), "color": (150, 50, 50)},   # Dark red box
            {"pos": (350, 400), "size": (70, 70), "color": (50, 150, 50)},   # Dark green box
            {"pos": (480, 400), "size": (65, 75), "color": (70, 70, 170)},   # Dark blue box
        ]
        
        # Draw items on before image
        for item in items:
            cv2.rectangle(before, 
                        item["pos"], 
                        (item["pos"][0] + item["size"][0], item["pos"][1] + item["size"][1]), 
                        item["color"], 
                        -1)
        
        # Create after image (initially the same)
        after = before.copy()
        
        # Apply lighting change if requested
        if lighting_change != 0:
            # Convert to HSV for better lighting adjustment
            after_hsv = cv2.cvtColor(after, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(after_hsv)
            
            # Adjust value (brightness) channel
            adjustment = int(lighting_change * 50)  # Scale to reasonable range
            v = np.clip(v.astype(np.int32) + adjustment, 0, 255).astype(np.uint8)
            
            # Merge and convert back
            after_hsv = cv2.merge([h, s, v])
            after = cv2.cvtColor(after_hsv, cv2.COLOR_HSV2BGR)
        
        # Add a new item if requested
        if add_item:
            new_item = {"pos": (550, 400), "size": (80, 60), "color": (180, 120, 40)}
            cv2.rectangle(after, 
                        new_item["pos"], 
                        (new_item["pos"][0] + new_item["size"][0], new_item["pos"][1] + new_item["size"][1]), 
                        new_item["color"], 
                        -1)
                        
        # Remove an item if requested
        if remove_item:
            remove_idx = 2  # Remove the red box
            item = items[remove_idx]
            cv2.rectangle(before, 
                        item["pos"], 
                        (item["pos"][0] + item["size"][0], item["pos"][1] + item["size"][1]), 
                        item["color"], 
                        -1)
            # Replace with background in after image
            cv2.rectangle(after, 
                        item["pos"], 
                        (item["pos"][0] + item["size"][0], item["pos"][1] + item["size"][1]), 
                        (200, 200, 200),  # Background color
                        -1)
        
        return before, after
    
    def test_lighting_invariance(self) -> Tuple[str, Dict[str, Any]]:
        """
        Test lighting-invariant detection with different lighting conditions.
        
        Returns:
            Tuple of (test_status, test_details)
        """
        logger.info("Running lighting invariance test")
        
        # Create a series of tests with different lighting changes
        lighting_changes = [-0.8, -0.5, -0.2, 0.2, 0.5, 0.8]
        results = []
        
        # Standard test: no items added/removed, just lighting changes
        for light_change in lighting_changes:
            # Generate test images
            before, after = self._generate_synthetic_before_after(lighting_change=light_change)
            
            # Save test images for reference
            cv2.imwrite(str(self.test_data_dir / f"light_{light_change:.1f}_before.jpg"), before)
            cv2.imwrite(str(self.test_data_dir / f"light_{light_change:.1f}_after.jpg"), after)
            
            # Analyze with the detector
            result = self.detector._analyze_inventory_change(
                before, after, 3.0, "test_camera", "lighting_test_zone"
            )
            
            # Should NOT detect change (only lighting changed)
            results.append({
                "lighting_change": light_change,
                "change_detected": result["change_detected"],
                "lighting_difference": result["analysis"]["lighting_difference"],
                "change_percentage": result["change_percentage"]
            })
        
        # Calculate success rate
        false_positives = sum(1 for r in results if r["change_detected"])
        success_rate = (len(results) - false_positives) / len(results)
        
        # Determine test status
        if success_rate >= 0.9:
            status = TestResult.STATUS_PASS
        elif success_rate >= 0.7:
            status = TestResult.STATUS_WARNING
        else:
            status = TestResult.STATUS_FAIL
        
        return status, {
            "success_rate": success_rate,
            "false_positives": false_positives,
            "total_tests": len(results),
            "lighting_results": results
        }
    
    def test_item_addition_detection(self) -> Tuple[str, Dict[str, Any]]:
        """
        Test item addition detection under various lighting conditions.
        
        Returns:
            Tuple of (test_status, test_details)
        """
        logger.info("Running item addition detection test")
        
        # Create a series of tests with different lighting changes
        lighting_changes = [-0.5, -0.2, 0.0, 0.2, 0.5]
        results = []
        
        for light_change in lighting_changes:
            # Generate test images with item addition
            before, after = self._generate_synthetic_before_after(
                lighting_change=light_change, 
                add_item=True
            )
            
            # Save test images for reference
            cv2.imwrite(str(self.test_data_dir / f"add_light_{light_change:.1f}_before.jpg"), before)
            cv2.imwrite(str(self.test_data_dir / f"add_light_{light_change:.1f}_after.jpg"), after)
            
            # Analyze with the detector
            result = self.detector._analyze_inventory_change(
                before, after, 3.0, "test_camera", "addition_test_zone"
            )
            
            # Should detect change and classify as ITEM_ADDITION
            results.append({
                "lighting_change": light_change,
                "change_detected": result["change_detected"],
                "event_type": result["event_type"],
                "confidence": result["confidence"],
                "correct_classification": result["event_type"] == InventoryChangeDetector.EVENT_ITEM_ADDITION
            })
        
        # Calculate success metrics
        detection_rate = sum(1 for r in results if r["change_detected"]) / len(results)
        classification_rate = sum(1 for r in results if r.get("correct_classification", False)) / len(results)
        
        # Determine test status
        if detection_rate >= 0.9 and classification_rate >= 0.8:
            status = TestResult.STATUS_PASS
        elif detection_rate >= 0.7 and classification_rate >= 0.6:
            status = TestResult.STATUS_WARNING
        else:
            status = TestResult.STATUS_FAIL
        
        return status, {
            "detection_rate": detection_rate,
            "classification_rate": classification_rate,
            "total_tests": len(results),
            "item_addition_results": results
        }
        
    def test_item_removal_detection(self) -> Tuple[str, Dict[str, Any]]:
        """
        Test item removal detection under various lighting conditions.
        
        Returns:
            Tuple of (test_status, test_details)
        """
        logger.info("Running item removal detection test")
        
        # Create a series of tests with different lighting changes
        lighting_changes = [-0.5, -0.2, 0.0, 0.2, 0.5]
        results = []
        
        for light_change in lighting_changes:
            # Generate test images with item removal
            before, after = self._generate_synthetic_before_after(
                lighting_change=light_change, 
                remove_item=True
            )
            
            # Save test images for reference
            cv2.imwrite(str(self.test_data_dir / f"remove_light_{light_change:.1f}_before.jpg"), before)
            cv2.imwrite(str(self.test_data_dir / f"remove_light_{light_change:.1f}_after.jpg"), after)
            
            # Analyze with the detector
            result = self.detector._analyze_inventory_change(
                before, after, 3.0, "test_camera", "removal_test_zone"
            )
            
            # Should detect change and classify as ITEM_REMOVAL
            results.append({
                "lighting_change": light_change,
                "change_detected": result["change_detected"],
                "event_type": result["event_type"],
                "confidence": result["confidence"],
                "correct_classification": result["event_type"] == "ITEM_REMOVAL"
            })
        
        # Calculate success metrics
        detection_rate = sum(1 for r in results if r["change_detected"]) / len(results)
        classification_rate = sum(1 for r in results if r["correct_classification"]) / len(results)
        
        # Determine test status
        if detection_rate >= 0.9 and classification_rate >= 0.8:
            status = TestResult.STATUS_PASS
        elif detection_rate >= 0.6 and classification_rate >= 0.6:
            status = TestResult.STATUS_WARNING
        else:
            status = TestResult.STATUS_FAIL
            
        return status, {
            "detection_rate": detection_rate,
            "classification_rate": classification_rate,
            "total_tests": len(results),
            "item_removal_results": results
        }
    
    def test_zone_specific_settings(self) -> Tuple[str, Dict[str, Any]]:
        """
        Test that zone-specific sensitivity settings are applied correctly.
        
        Returns:
            Tuple of (test_status, test_details)
        """
        logger.info("Running zone-specific settings test")
        
        # Generate subtle test case (small item addition with lighting change)
        before, after = self._generate_synthetic_before_after(
            lighting_change=0.3,
            add_item=True
        )
        
        # Make the added item smaller/subtler for this test
        # Draw a small object in the after image
        cv2.rectangle(after, (550, 400), (590, 430), (180, 120, 40), -1)
        
        # Save test images for reference
        cv2.imwrite(str(self.test_data_dir / "zone_test_before.jpg"), before)
        cv2.imwrite(str(self.test_data_dir / "zone_test_after.jpg"), after)
        
        # Test with different zone settings
        results = {}
        
        # Test standard zone
        std_result = self.detector._analyze_inventory_change(
            before, after, 3.0, "test_camera", "test_zone_standard"
        )
        results["standard"] = {
            "change_detected": std_result["change_detected"],
            "threshold_used": std_result["analysis"]["change_threshold"],
            "min_area_used": std_result["analysis"]["min_change_area"]
        }
        
        # Test sensitive zone (should detect even subtle changes)
        sensitive_result = self.detector._analyze_inventory_change(
            before, after, 3.0, "test_camera", "test_zone_sensitive"
        )
        results["sensitive"] = {
            "change_detected": sensitive_result["change_detected"],
            "threshold_used": sensitive_result["analysis"]["change_threshold"],
            "min_area_used": sensitive_result["analysis"]["min_change_area"]
        }
        
        # Test tolerant zone (should ignore subtle changes)
        tolerant_result = self.detector._analyze_inventory_change(
            before, after, 3.0, "test_camera", "test_zone_tolerant"
        )
        results["tolerant"] = {
            "change_detected": tolerant_result["change_detected"],
            "threshold_used": tolerant_result["analysis"]["change_threshold"],
            "min_area_used": tolerant_result["analysis"]["min_change_area"]
        }
        
        # Check that the correct zone-specific settings were used
        correct_settings = (
            results["standard"]["threshold_used"] == self.config["zones"]["test_zone_standard"]["change_threshold"] and
            results["standard"]["min_area_used"] == self.config["zones"]["test_zone_standard"]["min_change_area"] and
            results["sensitive"]["threshold_used"] == self.config["zones"]["test_zone_sensitive"]["change_threshold"] and
            results["sensitive"]["min_area_used"] == self.config["zones"]["test_zone_sensitive"]["min_change_area"] and
            results["tolerant"]["threshold_used"] == self.config["zones"]["test_zone_tolerant"]["change_threshold"] and
            results["tolerant"]["min_area_used"] == self.config["zones"]["test_zone_tolerant"]["min_change_area"]
        )
        
        # Check that sensitivity differences work as expected
        sensitivity_correct = (
            results["sensitive"]["change_detected"] is True and  # Sensitive zone should detect
            results["tolerant"]["change_detected"] is False      # Tolerant zone should not
        )
        
        # Determine test status
        if correct_settings and sensitivity_correct:
            status = TestResult.STATUS_PASS
        elif correct_settings:  # Settings correct but sensitivity behavior not as expected
            status = TestResult.STATUS_WARNING
        else:
            status = TestResult.STATUS_FAIL
            
        return status, {
            "correct_zone_settings": correct_settings,
            "sensitivity_behavior_correct": sensitivity_correct,
            "zone_results": results
        }
    
    def run_all_tests(self):
        """Run all inventory detection tests"""
        # Test lighting invariance
        framework.run_test(
            "Lighting Invariance", 
            "Inventory Detection", 
            self.test_lighting_invariance
        )
        
        # Test item addition detection
        framework.run_test(
            "Item Addition Detection", 
            "Inventory Detection", 
            self.test_item_addition_detection
        )
        
        # Test item removal detection
        framework.run_test(
            "Item Removal Detection", 
            "Inventory Detection", 
            self.test_item_removal_detection
        )
        
        # Test zone-specific settings
        framework.run_test(
            "Zone-Specific Settings", 
            "Inventory Detection", 
            self.test_zone_specific_settings
        )
        
        # Save results
        framework.save_results("inventory_detection_results.json")
        framework.generate_report("inventory_detection_report.md")


def main():
    """Run all inventory detection tests"""
    # Create test instances
    inventory_tests = InventoryDetectionTests()
    
    # Run tests
    inventory_tests.run_all_tests()
    
    logger.info("All inventory detection tests completed")


if __name__ == "__main__":
    main()
