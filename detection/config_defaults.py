"""
SmartVan Monitor - Configuration Defaults
-----------------------------------------
This module defines default configuration values for the SmartVan Monitor system.
"""

# Default configuration with tuned parameters based on testing
DEFAULT_CONFIG = {
    # Camera settings - using list format for compatibility with existing code
    "cameras": [
        {
            "id": 0,
            "name": "Rear_Inventory",
            "fps": 2,  # 500ms intervals - optimized for detection quality vs CPU usage
            "resolution": [640, 480]
        }
    ],
    # Global camera settings
    "camera_defaults": {
        "fps": 2,
        "resolution": [640, 480]
    },
    
    # Motion detection settings
    "motion_detection": {
        "enabled": True,
        "frame_stability": 7,  # Consecutive frames required to confirm motion
        "noise_filter": 0.6,   # Threshold for filtering minor movements
        "min_area": 300,       # Minimum pixel area for significant motion
        "flow_magnitude_threshold": 3.5,  # For vibration detection
        "flow_uniformity_threshold": 0.5, # For vibration detection
        "significant_motion_threshold": 6.0,  # For distinguishing genuine motion
        "min_duration": 1.5    # Minimum duration for genuine motion
    },
    
    # Event classification settings
    "event_classification": {
        "min_alert_interval": 120,  # Seconds between alerts from same camera
        "motion_intensity_threshold": 40,  # Threshold for significant motion
        "high_frequency_threshold": 5,  # Number of events for high frequency alert
        "business_hours_start": 7,   # 7 AM
        "business_hours_end": 19,    # 7 PM
    },
    
    # Human detection settings (YOLO)
    "human_detection": {
        "enabled": True,
        "confidence_threshold": 0.5,  # Minimum confidence for human detection
        "nms_threshold": 0.4,        # Non-maximum suppression threshold
        "model_size": "tiny",        # YOLO model size (tiny, small, medium)
        "min_human_size": 0.05,      # Minimum size relative to frame
        "always_run": False,         # Run YOLO on every frame or only on motion
        "run_interval": 5            # Run YOLO every N frames when no motion
    },
    
    # Inventory detection settings
    "inventory_detection": {
        "enabled": True,
        "change_threshold": 0.08,    # Percentage change required for detection (0.0-1.0)
        "min_change_area": 200,      # Minimum contour area for meaningful change
        "access_timeout": 5.0,       # Seconds after access to check for changes
        "human_confidence_threshold": 0.5  # Minimum confidence for human detection
    },
    
    # Recording settings
    "recording": {
        "enabled": True,
        "min_intensity": 20,         # Minimum intensity for recording
        "min_recording_gap": 10,     # Minimum seconds between recordings
        "max_clip_duration": 30,     # Maximum clip duration in seconds
        "pre_record_buffer": 3,      # Seconds of pre-event footage to save
        "post_record_buffer": 5      # Seconds of post-event footage to save
    },
    
    # System settings
    "system": {
        "display": True,             # Whether to display video feed
        "output_dir": "output",      # Directory for output files
        "log_level": "INFO",         # Logging level
        "van_id": "VAN001"           # Unique identifier for this van
    }
}

# Function to get default config with potential overrides
def get_default_config(overrides=None):
    """
    Get default configuration with optional overrides.
    
    Args:
        overrides: Dictionary of configuration overrides
        
    Returns:
        Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    if overrides:
        # Deep update the config with overrides
        _deep_update(config, overrides)
        
    return config

def _deep_update(d, u):
    """Recursively update a dictionary."""
    import collections.abc
    
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = _deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
