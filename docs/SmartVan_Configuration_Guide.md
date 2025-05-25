# SmartVan Monitor Configuration Guide

## Optimized Default Settings

These settings have been optimized for detecting human presence (inventory access) in a moving vehicle environment while filtering out false positives from vehicle vibrations and minor movements.

### Motion Detection Parameters

```json
"motion_detection": {
    "algorithm": "MOG2",
    "learning_rate": 0.001,
    "motion_threshold": 25,
    "min_area": 300,
    "blur_size": 21,
    "noise_filter": 0.6,
    "movement_stability": 7,
    "human_detection": {
        "enabled": true,
        "flow_magnitude_threshold": 3.5,
        "flow_uniformity_threshold": 0.5,
        "significant_motion_threshold": 6.0,
        "min_duration": 1.5
    }
}
```

### Event Classification Parameters

```json
"event_classification": {
    "business_hours_start": 7,
    "business_hours_end": 19,
    "high_frequency_threshold": 5,
    "high_frequency_window": 900,
    "event_memory_window": 3600,
    "min_alert_interval": 120,
    "motion_intensity_threshold": 40
}
```

### Object Detection Parameters

```json
"object_detection": {
    "enabled": true,
    "model_size": "tiny",
    "confidence_threshold": 0.5,
    "nms_threshold": 0.4,
    "only_on_motion": true,
    "gpu_enabled": false,
    "yolo_interval": 5,
    "human_confirmation_needed": true,
    "object_recording_threshold": 0.6,
    "classes_of_interest": ["person", "backpack", "handbag", "suitcase", "bottle"]
}
```

### Inventory Detection Parameters

```json
"inventory_detection": {
    "enabled": true,
    "change_threshold": 0.15,
    "min_change_area": 200,
    "size_thresholds": {
        "small": 500,
        "medium": 2000
    },
    "zone_specific_settings": true,
    "lighting_invariance": true
}
```

### Camera Settings

```json
"cameras": [
    {
        "id": 0,
        "name": "Rear_Inventory",
        "resolution": [640, 480],
        "fps": 2,
        "roi": [
            [10, 10, 620, 460]
        ]
    }
]
```

## Parameter Explanations

### Motion Detection

| Parameter | Description | Recommended Value |
|-----------|-------------|------------------|
| `algorithm` | Background subtraction algorithm | `"MOG2"` |
| `learning_rate` | How quickly the system adapts to background changes | `0.001` |
| `motion_threshold` | Minimum pixel difference to be considered motion | `25` |
| `min_area` | Minimum contour area (in pixels) to be considered significant | `300` |
| `blur_size` | Size of Gaussian blur kernel to reduce noise | `21` |
| `noise_filter` | Threshold for filtering sporadic noise (0-1) | `0.6` |
| `movement_stability` | Consecutive frames required to confirm motion | `7` |

### Human Detection

| Parameter | Description | Recommended Value |
|-----------|-------------|------------------|
| `enabled` | Whether human detection is enabled | `true` |
| `flow_magnitude_threshold` | Minimum optical flow magnitude for human detection | `3.5` |
| `flow_uniformity_threshold` | Threshold for flow direction uniformity | `0.5` |
| `significant_motion_threshold` | Threshold for significant motion detection | `6.0` |
| `min_duration` | Minimum duration (seconds) for human movement | `1.5` |

### Event Classification

| Parameter | Description | Recommended Value |
|-----------|-------------|------------------|
| `business_hours_start` | Start of business hours (24-hour format) | `7` |
| `business_hours_end` | End of business hours (24-hour format) | `19` |
| `high_frequency_threshold` | Number of events to trigger high frequency alert | `5` |
| `high_frequency_window` | Time window for high frequency events (seconds) | `900` |
| `event_memory_window` | How long to remember events (seconds) | `3600` |
| `min_alert_interval` | Minimum seconds between alerts for same camera | `120` |
| `motion_intensity_threshold` | Minimum intensity to be considered significant | `40` |

### Object Detection

| Parameter | Description | Recommended Value |
|-----------|-------------|------------------|
| `enabled` | Whether YOLO object detection is enabled | `true` |
| `model_size` | Size of YOLO model (tiny, medium, large) | `"tiny"` |
| `confidence_threshold` | Minimum confidence for object detection | `0.5` |
| `nms_threshold` | Non-maximum suppression threshold | `0.4` |
| `only_on_motion` | Only run YOLO when motion is detected | `true` |
| `gpu_enabled` | Use GPU acceleration if available | `false` |
| `yolo_interval` | Run YOLO every X frames when no motion | `5` |
| `human_confirmation_needed` | Require both motion and YOLO to confirm human | `true` |
| `object_recording_threshold` | Confidence threshold to trigger recording | `0.6` |
| `classes_of_interest` | COCO classes to detect | `["person", "backpack", "handbag", "suitcase", "bottle"]` |

### Inventory Detection

| Parameter | Description | Recommended Value |
|-----------|-------------|------------------|
| `enabled` | Whether inventory change detection is enabled | `true` |
| `change_threshold` | Minimum percentage change to trigger detection | `0.15` |
| `min_change_area` | Minimum contour area (in pixels) for items | `200` |
| `size_thresholds.small` | Maximum area for small items | `500` |
| `size_thresholds.medium` | Maximum area for medium items, larger is "large" | `2000` |
| `zone_specific_settings` | Use zone-specific sensitivity settings | `true` |
| `lighting_invariance` | Apply lighting-invariant detection techniques | `true` |

### System Settings

| Parameter | Description | Recommended Value |
|-----------|-------------|------------------|
| `van_id` | Unique identifier for this vehicle in the fleet | `"VAN001"` |
| `display` | Whether to display video feed | `true` |
| `output_dir` | Directory for storing output files | `"output"` |
| `log_level` | Logging level (INFO, DEBUG, WARNING, ERROR) | `"INFO"` |

### Camera Settings

| Parameter | Description | Recommended Value |
|-----------|-------------|------------------|
| `fps` | Frames per second to capture | `2` |
| `resolution` | Camera resolution as [width, height] | `[640, 480]` |

## Tuning Guidelines

### For More Sensitive Detection
- Decrease `flow_magnitude_threshold` (e.g., to 2.5-3.0)
- Decrease `motion_intensity_threshold` (e.g., to 30-35)
- Decrease `movement_stability` (e.g., to 5-6)

### For Less False Positives
- Increase `flow_magnitude_threshold` (e.g., to 4.0-5.0)
- Increase `motion_intensity_threshold` (e.g., to 50-60)
- Increase `movement_stability` (e.g., to 8-10)
- Increase `min_alert_interval` (e.g., to 180-300)
- Increase `min_change_area` for inventory detection (e.g., to 300-500)

## Command Line Options

The SmartVan Monitor supports several command-line options that can override settings in the configuration file:

| Option | Description | Example |
|--------|-------------|--------|
| `--config` | Path to configuration file | `--config fleet_van12.json` |
| `--no-display` | Disable video display window | `--no-display` |
| `--vibration-filter` | Set vibration filter threshold (0-100) | `--vibration-filter 45` |
| `--van-id` | Set unique van identifier | `--van-id FLEET_VAN_12` |

### Examples

```bash
# Basic usage with default config.json
python main.py

# Specify a different config file
python main.py --config delivery_van.json

# Run in headless mode (no display)
python main.py --no-display

# Set a specific van ID for fleet deployment
python main.py --van-id DELIVERY_VAN_42

# Combine multiple options
python main.py --van-id FLEET_VAN_12 --vibration-filter 45 --no-display
```

## Example config.json

```json
{
    "van_id": "VAN001",
    "inventory_detection": {
        "enabled": true,
        "change_threshold": 0.15,
        "min_change_area": 200,
        "size_thresholds": {
            "small": 500,
            "medium": 2000
        },
        "zone_specific_settings": true,
        "lighting_invariance": true
    },
    "cameras": [
        {
            "id": 0,
            "name": "Rear_Inventory",
            "resolution": [640, 480],
            "fps": 2,
            "roi": [
                [10, 10, 620, 460]
            ]
        },
        {
            "id": 1,
            "name": "Side_Wall",
            "resolution": [640, 480],
            "fps": 2,
            "roi": [
                [10, 10, 620, 460]
            ]
        }
    ],
    "motion_detection": {
        "algorithm": "MOG2",
        "learning_rate": 0.001,
        "motion_threshold": 25,
        "min_area": 300,
        "blur_size": 21,
        "noise_filter": 0.6,
        "movement_stability": 7,
        "human_detection": {
            "enabled": true,
            "flow_magnitude_threshold": 3.5,
            "flow_uniformity_threshold": 0.5,
            "significant_motion_threshold": 6.0,
            "min_duration": 1.5
        }
    },
    "event_classification": {
        "business_hours_start": 7,
        "business_hours_end": 19,
        "high_frequency_threshold": 5,
        "high_frequency_window": 900,
        "event_memory_window": 3600,
        "min_alert_interval": 120,
        "motion_intensity_threshold": 40
    },
    "object_detection": {
        "enabled": true,
        "model_size": "tiny",
        "confidence_threshold": 0.5,
        "nms_threshold": 0.4,
        "only_on_motion": true,
        "gpu_enabled": false,
        "yolo_interval": 5,
        "human_confirmation_needed": true,
        "object_recording_threshold": 0.6,
        "classes_of_interest": ["person", "backpack", "handbag", "suitcase", "bottle"]
    },
    "recording": {
        "min_intensity": 20,
        "pre_motion_seconds": 2,
        "post_motion_seconds": 3,
        "min_recording_gap": 10
    },
    "output_dir": "output",
    "display": true
}
```
