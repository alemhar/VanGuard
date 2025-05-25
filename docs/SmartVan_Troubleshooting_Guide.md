# SmartVan Monitor Troubleshooting Guide

## Common Issues and Solutions

This guide addresses common issues encountered with the SmartVan Monitor system and provides detailed solutions with a focus on reducing false positives from vehicle vibrations.

## False Positive Detection Issues

### Vibration-Related False Positives

**Symptoms:**
- System reports inventory access events when no one has accessed the inventory
- Multiple low-confidence detections during vehicle movement
- High-frequency events triggered by minor vibrations or shaking

**Solutions:**

1. **Adjust Vibration Filtering Parameters**
   ```json
   "motion_detection": {
     "human_detection": {
       "flow_magnitude_threshold": 4.5,  // Increase from default 3.5
       "flow_uniformity_threshold": 0.6  // Increase from default 0.5
     }
   }
   ```

2. **Increase Motion Stability Requirements**
   ```json
   "motion_detection": {
     "movement_stability": 9  // Increase from default 7
   }
   ```

3. **Filter Low-Intensity Vibrations**
   ```json
   "event_classification": {
     "motion_intensity_threshold": 50  // Increase from default 40
   }
   ```

4. **Implement Rate Limiting for Alerts**
   ```json
   "event_classification": {
     "min_alert_interval": 180  // Increase from default 120 seconds
   }
   ```

5. **Use Size-Based Filtering**
   ```json
   "inventory_detection": {
     "min_change_area": 350,  // Increase from default 200
     "size_thresholds": {
       "small": 600  // Increase small item threshold to ignore tiny changes
     }
   }
   ```

### Distinguishing Vibrations from Human Presence

The system distinguishes between vibration patterns and human presence using:

1. **Optical Flow Analysis**
   - Vehicle vibrations typically create random, low-magnitude flow patterns
   - Human movements create directional, higher-magnitude flow

2. **Motion Duration**
   - Vibrations are typically short and intermittent
   - Human presence creates sustained motion patterns

3. **Spatial Pattern Analysis**
   - Vibrations affect the entire frame somewhat uniformly
   - Human access creates localized, directional changes

### Fine-Tuning for Specific Vehicle Types

Different vehicle types may require different settings:

| Vehicle Type | Recommended Settings |
|--------------|---------------------|
| **Delivery Van** | `flow_magnitude_threshold`: 4.0<br>`movement_stability`: 8<br>`motion_intensity_threshold`: 45 |
| **Service Truck** | `flow_magnitude_threshold`: 4.5<br>`movement_stability`: 9<br>`motion_intensity_threshold`: 55 |
| **Utility Vehicle** | `flow_magnitude_threshold`: 5.0<br>`movement_stability`: 10<br>`motion_intensity_threshold`: 60 |

## Size-Based Classification Issues

### Incorrect Size Classification

**Symptoms:**
- Items consistently classified in wrong size category
- Size distribution doesn't match actual inventory changes

**Solutions:**

1. **Calibrate Size Thresholds**
   ```json
   "inventory_detection": {
     "size_thresholds": {
       "small": 500,    // Adjust based on typical items
       "medium": 2000   // Adjust based on typical items
     }
   }
   ```

2. **Test with Known Items**
   - Place items of known sizes in the field of view
   - Run the detection system and observe classification
   - Adjust thresholds based on results

3. **Consider Camera Distance**
   - Closer cameras will make objects appear larger
   - Further cameras will make objects appear smaller
   - Adjust thresholds accordingly for each camera

## Lighting-Related Issues

### Poor Detection in Varying Light Conditions

**Symptoms:**
- System works well during day but fails at night
- False positives when lights turn on/off
- Missed detections in low light

**Solutions:**

1. **Enable Lighting Invariance**
   ```json
   "inventory_detection": {
     "lighting_invariance": true
   }
   ```

2. **Adjust Histogram Parameters**
   - Modify `inventory_detection.py` to adjust histogram normalization parameters:
   ```python
   def _normalize_histogram(self, image):
       # Adjust clahe parameters for better lighting adaptation
       clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
       # Apply processing
   ```

3. **Consider IR Camera for Night Operation**
   - Infrared cameras provide consistent imaging regardless of visible light
   - Update camera settings accordingly

## System Performance Issues

### High CPU Usage

**Symptoms:**
- System is slow to respond
- High CPU utilization
- Delayed detection notifications

**Solutions:**

1. **Reduce Frame Rate**
   ```json
   "cameras": [
     {
       "fps": 1  // Decrease from default 2
     }
   ]
   ```

2. **Limit Detection Zones**
   - Define smaller, more focused inventory zones
   - Avoid processing the entire frame

3. **Disable Unused Features**
   ```json
   "object_detection": {
     "enabled": false  // Disable if not needed
   }
   ```

## Installation and Setup Issues

### Camera Connection Problems

**Symptoms:**
- "Camera not found" errors
- Intermittent camera disconnections
- Poor image quality

**Solutions:**

1. **Check USB Connections**
   - Ensure cables are securely connected
   - Try different USB ports
   - Use USB hubs with dedicated power supply

2. **Update Camera Drivers**
   - Install latest drivers for your camera model
   - Test with standard camera applications

3. **Adjust Camera Settings**
   - Lower resolution for more reliable operation
   - Ensure compatible format (MJPEG recommended)

## Advanced Troubleshooting

### Enable Debug Logging

For detailed diagnostics, enable debug logging:

```json
"logging": {
  "level": "DEBUG",
  "file": "smartvan_debug.log"
}
```

### Analyze Detection Events

Review detection event logs to identify patterns:

1. Check `output/events/` directory for JSON event files
2. Look for patterns in false positive occurrences
3. Compare detection confidence scores

### Test in Controlled Environment

Before deploying in a moving vehicle:

1. Test in stationary environment first
2. Gradually introduce controlled vibrations
3. Compare detection results and adjust settings

## Getting Help

If issues persist after trying these solutions:

1. Generate a diagnostic report:
   ```bash
   python -m diagnostics.generate_report
   ```

2. Contact support with:
   - Your diagnostic report
   - System configuration file
   - Description of the issue
   - Any error messages or logs
