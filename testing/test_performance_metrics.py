"""
SmartVan Monitor - Performance Metrics Testing
---------------------------------------------
Tests the performance metrics tracking and visualization capabilities.
"""

import os
import time
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

from detection.inventory_detection import InventoryChangeDetector
from detection.performance_metrics import PerformanceTracker

def generate_test_data(detector, count=100):
    """Generate synthetic test data for performance metrics."""
    print(f"Generating {count} synthetic test detection events...")
    
    # Create base images for detection tests
    base_img_size = (640, 480, 3)
    empty_img = np.ones(base_img_size, dtype=np.uint8) * 200  # Gray background
    
    # Create some shelf lines for realism
    for y in range(100, 401, 100):
        cv2.line(empty_img, (50, y), (590, y), (150, 150, 150), 3)
    
    # Register inventory zones - use larger heights to accommodate test items
    zones = [
        {"id": "shelf_top", "name": "Top Shelf", "x": 100, "y": 50, "width": 400, "height": 120},
        {"id": "shelf_middle", "name": "Middle Shelf", "x": 100, "y": 180, "width": 400, "height": 120},
        {"id": "shelf_bottom", "name": "Bottom Shelf", "x": 100, "y": 310, "width": 400, "height": 120}
    ]
    detector.register_inventory_zones("test_camera", zones)
    
    # Generate a variety of detection events with different conditions
    
    # Set up possible conditions
    event_types = ["ITEM_ADDED", "ITEM_REMOVED", "NO_CHANGE"]
    size_categories = ["small", "medium", "large"]
    
    # Timing for events (more recent events first)
    base_time = time.time()
    
    # Environment conditions
    vehicle_states = [
        {"is_moving": False, "movement_type": "stationary"},
        {"is_moving": True, "movement_type": "driving"},
        {"is_moving": True, "movement_type": "stop_and_go"}
    ]
    
    environment_states = [
        {"light_level": "normal", "time_of_day": "day"},
        {"light_level": "dark", "time_of_day": "night"},
        {"light_level": "bright", "time_of_day": "day"}
    ]
    
    # Generate detection events
    for i in range(count):
        # Randomly select conditions
        event_time = base_time - random.randint(0, 86400)  # Random time in last 24 hours
        zone_id = random.choice([z["id"] for z in zones])
        event_type = random.choice(event_types)
        dominant_size = random.choice(size_categories)
        
        # Set up environmental conditions
        vehicle_state = random.choice(vehicle_states)
        environment_state = random.choice(environment_states)
        detector.update_vehicle_state(**vehicle_state)
        detector.update_environment_state(**environment_state)
        
        # Create before/after images based on event type
        before_img = empty_img.copy()
        after_img = empty_img.copy()
        
        # For item added or removed, add/remove items
        if event_type != "NO_CHANGE":
            # Add item to appropriate image
            item_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            # Find zone coordinates
            zone = next(z for z in zones if z["id"] == zone_id)
            x, y, w, h = zone["x"], zone["y"], zone["width"], zone["height"]
            
            # Create random item size based on dominant_size and zone size
            # Ensure item fits within zone with padding
            max_width = max(20, min(w - 30, 160))  # Ensure at least 15px padding on each side
            max_height = max(20, min(h - 30, 160))  # Ensure at least 15px padding on each side
            
            # Determine size ranges based on zone size and dominant_size
            if dominant_size == "small":
                min_size = 20
                max_width_size = min(40, max_width)
                max_height_size = min(40, max_height)
            elif dominant_size == "medium":
                min_size = 40
                max_width_size = min(80, max_width)
                max_height_size = min(80, max_height)
            else:  # large
                min_size = 60
                max_width_size = max_width
                max_height_size = max_height
                
            # Ensure valid ranges (min must be <= max)
            if min_size > max_width_size:
                min_size = max_width_size
            if min_size > max_height_size:
                min_size = max_height_size
                
            # Generate random dimensions
            item_width = random.randint(min_size, max_width_size)
            item_height = random.randint(min_size, max_height_size)
                
            # Calculate available space for positioning
            width_space = w - item_width - 20  # 10px padding on each side
            height_space = h - item_height - 20  # 10px padding on each side
            
            # Ensure we have valid ranges
            width_space = max(1, width_space)
            height_space = max(1, height_space)
            
            # Random position within zone
            item_x = x + random.randint(10, 10 + width_space)
            item_y = y + random.randint(10, 10 + height_space)
            
            # Draw item on appropriate image
            if event_type == "ITEM_ADDED":
                cv2.rectangle(after_img, (item_x, item_y), 
                            (item_x + item_width, item_y + item_height), 
                            item_color, -1)
            else:  # ITEM_REMOVED
                cv2.rectangle(before_img, (item_x, item_y), 
                            (item_x + item_width, item_y + item_height), 
                            item_color, -1)
        
        # Analyze the change
        # For testing, we'll add a duration parameter of 2 seconds
        result = detector._analyze_inventory_change(
            before_img, after_img, 2.0, "test_camera", zone_id
        )
        
        # Sometimes add user feedback
        if random.random() < 0.3:  # 30% of events get feedback
            is_true_positive = random.random() < 0.8  # 80% are true positives
            feedback_type = "manual_review"
            
            # Generate a random detection ID
            detection_id = f"test_{i}_{int(event_time)}"
            
            # Add feedback
            detector.log_user_feedback(
                detection_id=detection_id,
                feedback_type=feedback_type,
                is_correct=is_true_positive,
                notes="Synthetic test feedback",
                camera="test_camera",
                zone=zone_id
            )
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{count} events")
    
    print("Test data generation complete.")

def main():
    """Run performance metrics test"""
    # Create output directory
    output_dir = "testing/test_output/performance_metrics"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize detector with performance tracking
    detector = InventoryChangeDetector(
        output_dir=output_dir,
        enable_performance_tracking=True
    )
    
    # Generate test data
    generate_test_data(detector, count=100)
    
    # Generate reports
    print("\nGenerating performance reports...")
    
    # HTML report
    html_report = detector.generate_performance_report(output_format="html")
    html_path = os.path.join(output_dir, "performance_report.html")
    with open(html_path, "w") as f:
        f.write(html_report)
    print(f"HTML report saved to: {html_path}")
    
    # Markdown report
    md_report = detector.generate_performance_report(output_format="markdown")
    md_path = os.path.join(output_dir, "performance_report.md")
    with open(md_path, "w") as f:
        f.write(md_report)
    print(f"Markdown report saved to: {md_path}")
    
    # Generate visualizations
    print("\nGenerating performance visualizations...")
    viz_paths = detector.generate_performance_visualizations()
    for viz_type, path in viz_paths.items():
        print(f"  {viz_type}: {path}")
    
    print("\nPerformance metrics testing complete.")

if __name__ == "__main__":
    main()
