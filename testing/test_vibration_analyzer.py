"""
SmartVan Monitor - Vibration Analysis Tests
------------------------------------------
Tests the vibration analysis capabilities, with focus on:
1. Distinguishing vibration from human movement
2. Vibration type categorization (IDLE_ENGINE, VEHICLE_MOVEMENT, etc.)
3. Detection of repetitive movements vs genuine human activity
"""

import cv2
import numpy as np
import os
import logging
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from testing.test_framework import TestResult, framework
from detection.vibration_analyzer import VibrationAnalyzer

# Configure logging
logger = logging.getLogger("vibration_analyzer_test")

class VibrationAnalyzerTests:
    """Tests for the vibration analysis system"""
    
    def __init__(self, test_data_dir: str = "testing/test_data/vibration"):
        """
        Initialize vibration analyzer tests.
        
        Args:
            test_data_dir: Directory containing test data
        """
        self.test_data_dir = Path(test_data_dir)
        self.test_data_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize analyzer with test settings
        self.analyzer = VibrationAnalyzer(
            flow_magnitude_threshold=3.5,
            flow_uniformity_threshold=0.5,
            significant_motion_threshold=6.0
        )
        
        logger.info("Vibration analyzer tests initialized")
    
    def _generate_flow_history(self, 
                             pattern_type: str, 
                             duration_seconds: float = 3.0,
                             frame_rate: float = 2.0) -> List[Dict[str, Any]]:
        """
        Generate synthetic optical flow history for testing.
        
        Args:
            pattern_type: Type of pattern to generate 
                         (human, idle_engine, vehicle_movement, rough_road, door_slam, repetitive)
            duration_seconds: Duration of the pattern in seconds
            frame_rate: Frame rate in frames per second
            
        Returns:
            List of flow history records
        """
        num_frames = int(duration_seconds * frame_rate)
        flow_history = []
        
        # Base values for each pattern type
        base_values = {
            "human": {
                "magnitude_range": (4.0, 8.0),
                "uniformity_range": (0.3, 0.7),
                "direction_variability": 45,  # Degrees of variation
                "baseline_direction": random.randint(0, 359)
            },
            "idle_engine": {
                "magnitude_range": (1.5, 3.0),
                "uniformity_range": (0.7, 0.9),
                "direction_variability": 10,
                "baseline_direction": random.randint(0, 359)
            },
            "vehicle_movement": {
                "magnitude_range": (3.0, 6.0),
                "uniformity_range": (0.8, 0.95),
                "direction_variability": 15,
                "baseline_direction": random.randint(0, 359)
            },
            "rough_road": {
                "magnitude_range": (5.0, 10.0),
                "uniformity_range": (0.6, 0.8),
                "direction_variability": 30,
                "baseline_direction": random.randint(0, 359)
            },
            "door_slam": {
                "magnitude_range": (8.0, 15.0),
                "uniformity_range": (0.7, 0.9),
                "direction_variability": 20,
                "baseline_direction": random.randint(0, 359),
                "spike_duration": 0.5  # Seconds
            },
            "repetitive": {
                "magnitude_range": (3.0, 7.0),
                "uniformity_range": (0.6, 0.8),
                "direction_variability": 10,
                "baseline_direction": random.randint(0, 359),
                "frequency": 1.0  # Hz (cycles per second)
            }
        }
        
        values = base_values[pattern_type]
        base_timestamp = time.time()
        
        for i in range(num_frames):
            timestamp = base_timestamp + (i / frame_rate)
            
            # Special handling for different pattern types
            if pattern_type == "door_slam" and i / frame_rate < values["spike_duration"]:
                # Spike for door slam at the beginning
                magnitude = random.uniform(*values["magnitude_range"])
                uniformity = random.uniform(*values["uniformity_range"])
            elif pattern_type == "door_slam":
                # After spike, return to low values
                magnitude = random.uniform(1.0, 2.0)
                uniformity = random.uniform(0.3, 0.5)
            elif pattern_type == "repetitive":
                # Sinusoidal pattern for repetitive movement
                cycle_position = (i / frame_rate) * values["frequency"] * 2 * np.pi
                magnitude_base = np.sin(cycle_position) * 0.5 + 0.5  # 0 to 1
                magnitude = values["magnitude_range"][0] + magnitude_base * (
                    values["magnitude_range"][1] - values["magnitude_range"][0]
                )
                uniformity = random.uniform(*values["uniformity_range"])
            else:
                # Normal random within range for other types
                magnitude = random.uniform(*values["magnitude_range"])
                uniformity = random.uniform(*values["uniformity_range"])
            
            # Randomize direction around baseline
            direction = (values["baseline_direction"] + 
                        random.uniform(-values["direction_variability"], values["direction_variability"])) % 360
            
            # Create flow history record
            flow_record = {
                "timestamp": timestamp,
                "avg_magnitude": magnitude,
                "uniformity": uniformity,
                "dominant_angle": direction,
                "significant_motion": magnitude > self.analyzer.significant_motion_threshold
            }
            
            flow_history.append(flow_record)
        
        return flow_history
    
    def test_vibration_type_categorization(self) -> Tuple[str, Dict[str, Any]]:
        """
        Test vibration type categorization to ensure different patterns are correctly identified.
        
        Returns:
            Tuple of (test_status, test_details)
        """
        logger.info("Running vibration type categorization test")
        
        # Test patterns
        patterns = ["idle_engine", "vehicle_movement", "rough_road", "door_slam", "repetitive"]
        results = {}
        
        for pattern in patterns:
            # Generate flow history for this pattern
            flow_history = self._generate_flow_history(pattern, duration_seconds=5.0)
            
            # Analyze the pattern
            result = self.analyzer.analyze_vibration_pattern(flow_history)
            
            # Store results
            results[pattern] = {
                "detected_type": result["vibration_type"],
                "confidence": result["confidence"],
                "frequency": result["frequency"],
                "regularity": result["regularity"],
                "is_vibration": result["is_vibration"]
            }
        
        # Expected categorization for each pattern
        expected_types = {
            "idle_engine": "IDLE_ENGINE",
            "vehicle_movement": "VEHICLE_MOVEMENT",
            "rough_road": "ROUGH_ROAD",
            "door_slam": "DOOR_SLAM",
            "repetitive": "REPETITIVE_MOTION"
        }
        
        # Check if types were correctly identified
        correct_count = 0
        for pattern, expected in expected_types.items():
            if results[pattern]["detected_type"] == expected:
                correct_count += 1
        
        # Calculate accuracy
        accuracy = correct_count / len(patterns)
        
        # Determine test status
        if accuracy >= 0.8:
            status = TestResult.STATUS_PASS
        elif accuracy >= 0.6:
            status = TestResult.STATUS_WARNING
        else:
            status = TestResult.STATUS_FAIL
            
        return status, {
            "accuracy": accuracy,
            "correct_identifications": correct_count,
            "total_tests": len(patterns),
            "pattern_results": results
        }
    
    def test_human_vs_vibration_detection(self) -> Tuple[str, Dict[str, Any]]:
        """
        Test the system's ability to distinguish human movement from various vibration types.
        
        Returns:
            Tuple of (test_status, test_details)
        """
        logger.info("Running human vs vibration detection test")
        
        # Generate test cases
        test_cases = [
            {"type": "human", "expected_human": True},
            {"type": "idle_engine", "expected_human": False},
            {"type": "vehicle_movement", "expected_human": False},
            {"type": "rough_road", "expected_human": False},
            {"type": "door_slam", "expected_human": False},
            {"type": "repetitive", "expected_human": False}
        ]
        
        results = []
        
        for case in test_cases:
            # Generate flow history
            flow_history = self._generate_flow_history(case["type"], duration_seconds=3.0)
            
            # Test human detection
            is_human = not self.analyzer.is_likely_vibration(flow_history)
            
            # Store result
            results.append({
                "pattern_type": case["type"],
                "expected_human": case["expected_human"],
                "detected_human": is_human,
                "correct": is_human == case["expected_human"]
            })
        
        # Calculate metrics
        correct_count = sum(1 for r in results if r["correct"])
        accuracy = correct_count / len(results)
        
        # False positives: detected human but was vibration
        false_positives = sum(1 for r in results 
                           if r["detected_human"] and not r["expected_human"])
        
        # False negatives: didn't detect human but was human
        false_negatives = sum(1 for r in results 
                            if not r["detected_human"] and r["expected_human"])
        
        # Determine test status
        if accuracy >= 0.9 and false_positives == 0:
            status = TestResult.STATUS_PASS
        elif accuracy >= 0.7 and false_positives <= 1:
            status = TestResult.STATUS_WARNING
        else:
            status = TestResult.STATUS_FAIL
            
        return status, {
            "accuracy": accuracy,
            "correct_classifications": correct_count,
            "total_tests": len(results),
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "detailed_results": results
        }
    
    def test_repetitive_motion_filtering(self) -> Tuple[str, Dict[str, Any]]:
        """
        Test the system's ability to identify and filter repetitive motions (like shaking fingers).
        
        Returns:
            Tuple of (test_status, test_details)
        """
        logger.info("Running repetitive motion filtering test")
        
        # Generate test flow history for repetitive motion
        repetitive_flow = self._generate_flow_history("repetitive", duration_seconds=10.0)
        
        # Analyze the pattern
        result = self.analyzer.analyze_vibration_pattern(repetitive_flow)
        
        # Check that it was correctly identified as repetitive
        is_repetitive = result["vibration_type"] == "REPETITIVE_MOTION"
        
        # Check that frequency and regularity are correctly measured
        has_frequency = result["frequency"] > 0.0
        has_regularity = result["regularity"] > 0.0
        
        # Determine if it would be filtered properly
        would_filter = result["is_vibration"]
        
        # Determine test status
        if is_repetitive and would_filter:
            status = TestResult.STATUS_PASS
        elif is_repetitive or would_filter:
            status = TestResult.STATUS_WARNING
        else:
            status = TestResult.STATUS_FAIL
            
        return status, {
            "identified_as_repetitive": is_repetitive,
            "would_be_filtered": would_filter,
            "measured_frequency": result["frequency"],
            "measured_regularity": result["regularity"],
            "confidence": result["confidence"]
        }
    
    def run_all_tests(self):
        """Run all vibration analyzer tests"""
        # Test vibration type categorization
        framework.run_test(
            "Vibration Type Categorization", 
            "Vibration Analysis", 
            self.test_vibration_type_categorization
        )
        
        # Test human vs vibration detection
        framework.run_test(
            "Human vs Vibration Detection", 
            "Vibration Analysis", 
            self.test_human_vs_vibration_detection
        )
        
        # Test repetitive motion filtering
        framework.run_test(
            "Repetitive Motion Filtering", 
            "Vibration Analysis", 
            self.test_repetitive_motion_filtering
        )
        
        # Save results
        framework.save_results("vibration_analyzer_results.json")
        framework.generate_report("vibration_analyzer_report.md")


def main():
    """Run all vibration analyzer tests"""
    # Create test instances
    vibration_tests = VibrationAnalyzerTests()
    
    # Run tests
    vibration_tests.run_all_tests()
    
    logger.info("All vibration analyzer tests completed")


if __name__ == "__main__":
    main()
