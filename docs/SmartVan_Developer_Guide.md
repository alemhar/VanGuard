# SmartVan Monitor Developer Guide

## Overview

This guide provides technical information for developers working on the SmartVan Monitor system. It covers architecture, key components, and implementation details to help you understand and extend the system.

## Core Modules

### 1. Inventory Detection Module

The inventory detection module is responsible for detecting changes in inventory zones, classifying items by size, and providing detailed analysis of inventory events.

#### Key Features

- **Lighting-invariant detection**: Uses histogram normalization and edge-based techniques to detect true inventory changes while ignoring lighting variations
- **Size-based classification**: Classifies detected items as small, medium, or large based on their area
- **Zone-specific sensitivity**: Applies different detection thresholds to different inventory zones
- **Spatial pattern analysis**: Determines if changes are concentrated in one area or distributed across multiple areas

#### Key Classes

- `InventoryChangeDetector`: Main class for inventory change detection
  - `detect_inventory_change()`: Detects changes between two frames
  - `_analyze_inventory_change()`: Core analysis method for detailed change detection
  - `_classify_item_by_size()`: Categorizes items based on contour area
  - `_categorize_change_pattern()`: Analyzes spatial patterns of detected changes

#### Detection Result Structure

```json
{
  "event_type": "DISTRIBUTED_MEDIUM_CHANGE",  // Detailed event type with size info
  "change_detected": true,                    // Whether a change was detected
  "confidence": 0.85,                         // Confidence score (0-1)
  "change_percentage": 0.23,                  // Percentage of pixels changed
  "contour_count": 3,                         // Number of detected contours
  "camera_name": "Rear_Inventory",            // Camera where change was detected
  "zone_id": "left_shelf",                    // Zone where change was detected
  "item_sizes": ["small", "medium", "medium"], // Size of each detected item
  "size_distribution": {                      // Count of items by size
    "small": 1,
    "medium": 2,
    "large": 0
  },
  "dominant_size": "medium",                  // Most common item size
  "analysis": {                               // Detailed analysis metrics
    "significant_contours": 3,
    "change_threshold": 0.15,
    "min_change_area": 200,
    "lighting_difference": 0.12,
    "histogram_difference": 0.45,
    "texture_similarity": 0.38,
    "normalized_change_score": 0.67
  }
}
```

### 2. Motion Detection Module

The motion detection module distinguishes between vehicle vibrations and actual human presence (inventory access events).

#### Key Features

- Vibration filtering using optical flow analysis
- Human movement pattern recognition
- Motion stability tracking to confirm consistent motion

#### Optimized Parameters

- Camera settings: 2 FPS (500ms intervals)
- Human detection thresholds:
  - `flow_magnitude_threshold`: 3.5
  - `flow_uniformity_threshold`: 0.5
  - `significant_motion_threshold`: 6.0
  - `min_duration`: 1.5
- 7 consecutive frames required for motion confirmation

## Implementation Guidelines

### Size-Based Classification

Size classification is implemented in the `_classify_item_by_size()` method of the `InventoryChangeDetector` class. The current thresholds are:

```python
if area < 500:  # Small items (e.g., tools, small boxes)
    return "small"
elif area < 2000:  # Medium items (e.g., medium boxes, containers)
    return "medium"
else:  # Large items (e.g., large boxes, equipment)
    return "large"
```

These thresholds can be adjusted in the configuration file:

```json
"inventory_detection": {
    "size_thresholds": {
        "small": 500,
        "medium": 2000
    }
}
```

#### Customizing Size Thresholds

For vehicle-specific customization, you can adjust these thresholds based on:

1. Camera distance from inventory
2. Typical item sizes in the specific vehicle
3. Shelf or compartment dimensions

### Lighting Invariance Implementation

The system uses multiple techniques to handle varying lighting conditions:

1. **Histogram normalization**: Equalizes image histograms before comparison
2. **Edge-based detection**: Focuses on structural changes rather than intensity changes
3. **Texture analysis**: Compares texture patterns which are less affected by lighting
4. **Adaptive thresholding**: Adjusts thresholds based on lighting differences

## Testing Guidelines

### Inventory Detection Tests

The testing framework includes specialized tests for the inventory detection system:

- `test_lighting_invariance()`: Tests detection accuracy under different lighting conditions
- `test_item_size_classification()`: Validates the size classification feature
- `test_multi_item_detection()`: Tests detection of multiple items simultaneously
- `test_zone_specific_settings()`: Validates zone-specific sensitivity settings

### Running Tests

```bash
# Run all inventory detection tests
python -m testing.run_tests --module inventory --report

# Run specific test for item size classification
python -m testing.test_inventory_detection
```

### Test Output

Tests generate detailed reports in both JSON and Markdown formats:
- `test_results/inventory_detection_results.json`: Raw results data
- `test_results/inventory_detection_report.md`: Human-readable report

## Best Practices

1. **Minimizing False Positives**
   - Always apply vibration filtering for in-vehicle deployments
   - Consider zone-specific settings for areas prone to vibration
   - Use size classification to filter out very small changes

2. **Performance Optimization**
   - The inventory detection pipeline is computationally intensive
   - Process at a lower frame rate (2 FPS recommended)
   - Apply detection only to relevant zones rather than full frames

3. **Extending Size Classification**
   - To add more size categories, modify `_classify_item_by_size()` method
   - Add corresponding entries to the size distribution dictionary
   - Update visualization method to display the new categories

## Troubleshooting

### Common Issues

1. **False Positives from Vibrations**
   - Increase `min_change_area` threshold (e.g., to 300-500)
   - Ensure vibration filtering is enabled
   - Set appropriate `motion_intensity_threshold` (40+ recommended)

2. **Missed Detections**
   - Check lighting_invariance setting is enabled
   - Decrease `change_threshold` for more sensitivity
   - Verify zone definitions cover the intended area

3. **Size Classification Issues**
   - Adjust size thresholds based on your specific camera setup
   - Test with known object sizes to calibrate thresholds
   - Use the visualization to verify classifications

## Future Development

Planned enhancements to the inventory detection system:

1. **Deep learning-based item classification**
   - Recognize specific inventory items (tools, parts, etc.)
   - Track specific item movement between zones

2. **Learning mode**
   - Adapt to specific inventory patterns over time
   - Build vehicle-specific detection profiles

3. **Cloud integration**
   - Synchronize inventory events with fleet management systems
   - Provide analytics on inventory access patterns
