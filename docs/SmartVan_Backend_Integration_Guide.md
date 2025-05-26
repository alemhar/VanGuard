# SmartVan Monitor: Backend Integration Guide

## Overview

The SmartVan Monitor system uses an Edge Computing approach for backend communication, prioritizing system reliability while minimizing bandwidth requirements. This guide documents the integration points between the SmartVan Monitor application and backend servers.

## Architecture

The backend integration follows these key principles:

1. **Edge Computing First**: Processing happens locally on the van, with only relevant summarized data sent to the backend
2. **Offline Resilience**: All data is stored locally and queued if connectivity is lost
3. **Heartbeat Mechanism**: Regular status updates ensure the system is operational
4. **Prioritized Communication**: Critical events take precedence in transmission

```
┌────────────────────┐         ┌─────────────────────┐
│                    │         │                     │
│  SmartVan Monitor  │─────────▶  Backend Server     │
│  (Edge Device)     │◀────────│                     │
│                    │         │                     │
└────────────────────┘         └─────────────────────┘
     │                                   ▲
     │                                   │
     │         ┌─────────────────┐       │
     └────────▶│ Offline Queue   │───────┘
               │ (When Needed)   │
               └─────────────────┘
```

## API Endpoints

The system communicates with the following endpoints on the backend server:

| Endpoint     | Purpose                                | Frequency              | Priority |
|--------------|----------------------------------------|------------------------|----------|
| `/heartbeat` | System status and health check         | Every 5 minutes        | Medium   |
| `/events`    | Individual detection events            | As events occur        | High     |
| `/summary`   | Aggregated statistics and metrics      | Hourly                 | Low      |
| `/metrics`   | Performance and detection metrics      | As metrics update      | Low      |

## Data Formats

### Heartbeat Data

```json
{
  "van_id": "VAN001",              // Unique identifier for the vehicle
  "timestamp": 1621904400,        // Unix timestamp (seconds since epoch)
  "system_hash": "a1b2c3d4e5f6...", // Hash of critical system files to detect tampering
  "uptime": 7200,                // System uptime in seconds
  "free_disk_space": 8500,       // Available disk space in megabytes
  "cpu_usage": 35.5,             // CPU usage as percentage
  "memory_usage": 42.3,          // Memory usage as percentage
  "cameras_active": {            // Camera subsystem status
    "camera_count": 2,           // Number of connected cameras
    "all_active": true           // Whether all registered cameras are functioning
  },
  "detectors_status": {          // Detection subsystem status
    "detectors_active": true,    // Whether all detection algorithms are functioning
    "motion_detector": true,     // Status of motion detection subsystem
    "inventory_detector": true,  // Status of inventory change detection
    "yolo_detector": true,      // Status of YOLO human detection
    "yolo_model": "tiny"        // Current YOLO model size (tiny, medium, large)
  }
}
```

### Event Data (Edge-Filtered)

```json
{
  "van_id": "VAN001",                      // Unique identifier for the vehicle
  "camera": "rear_camera",                 // Camera that captured the event
  "timestamp": 1621904500,                // Unix timestamp (seconds since epoch)
  "datetime": "2025-05-25T17:15:00.000Z", // ISO 8601 formatted date/time (for human readability)
  "event_type": "INVENTORY_CHANGE",        // Type of event (INVENTORY_CHANGE, HUMAN_PRESENT, MOTION)
  "event_id": "motion_20250525_1715",     // Unique ID for correlating events across systems
  "human_detection": {                    // Human presence information (if detected)
    "detected": true,                     // Whether a human was detected in the frame
    "confidence": 0.85,                   // Confidence score for human detection (0-1)
    "detection_method": "YOLO",           // Detection method (YOLO, MOTION, COMBINED)
    "count": 1                            // Number of humans detected in frame
  },
  "inventory": {                          // Inventory-related information
    "zone_id": "shelf_1",                 // Specific monitored area within the van
    "access_detected": true,              // Whether someone accessed the inventory zone
    "change_detected": true,              // Whether items were added/removed
    "change_type": "ITEM_REMOVED"          // Type of change (ITEM_ADDED, ITEM_REMOVED, ITEM_MOVED)
  }
}
```

### Summary Data

```json
{
  "van_id": "VAN001",               // Unique identifier for the vehicle
  "timestamp": 1621904400,         // Unix timestamp when summary was generated
  "start_time": 1621900800,        // Beginning of time period for this summary (Unix timestamp)
  "end_time": 1621904400,          // End of time period for this summary (Unix timestamp)
  "event_count": 12,               // Total number of events in this time period
  "event_types": {                 // Breakdown of events by type
    "INVENTORY_CHANGE": 5,         // Number of inventory change events
    "HUMAN_PRESENT": 2,            // Number of human presence events
    "MOTION": 5                    // Number of motion-only events
  },
  "camera_counts": {               // Events broken down by camera
    "rear_camera": 8,              // Events from the rear camera
    "side_camera": 4               // Events from the side camera
  },
  "high_confidence_events": 7,     // Events with high confidence score (>0.7)
  "inventory_access_count": 7,     // Number of times inventory zones were accessed
  "inventory_change_count": 5,     // Number of times items were added/removed
  "human_detection_count": 2       // Number of times humans were detected
}
```

### Performance Metrics Data

```json
{
  "van_id": "VAN001",              // Unique identifier for the vehicle
  "timestamp": 1621904400,        // Unix timestamp when metrics were generated
  "detection_counts": {           // Raw detection performance counts
    "true_positives": 18,         // Correct detections (verified by user feedback)
    "false_positives": 3,         // Incorrect detections (falsely reported changes)
    "false_negatives": 1          // Missed detections (changes not detected)
  },
  "accuracy": 0.818,              // Overall accuracy: (TP + TN) / (TP + TN + FP + FN)
  "precision": 0.857,             // Precision: TP / (TP + FP) - how many detections were correct
  "recall": 0.947,                // Recall: TP / (TP + FN) - how many actual events were detected
  "size_distribution": {          // Distribution of detected items by size
    "small": 7,                   // Items classified as small (<500 px²)
    "medium": 9,                  // Items classified as medium (500-2000 px²)
    "large": 5                    // Items classified as large (>2000 px²)
  },
  "false_positive_reasons": {     // Categorized reasons for false positives
    "vibration": 2,               // False positives due to vehicle vibration
    "lighting_change": 1,         // False positives due to lighting changes
    "no_human_detected": 3        // Events without human detection confirmation
  },
  "yolo_metrics": {              // YOLO object detection metrics
    "avg_inference_time": 0.145, // Average inference time in seconds
    "detections_count": 127,     // Total number of successful detections
    "objects_detected": {        // Count of detected object classes
      "person": 42,             // Number of person detections
      "backpack": 15,           // Number of backpack detections
      "handbag": 7              // Number of handbag detections
    }
  }
}
```

## Configuration

Backend communication is configured in the `config.json` file:

```json
"backend": {
  "server_url": "https://api.smartvan-monitor.com/v1",
  "api_key": "YOUR_API_KEY_HERE",
  "heartbeat_interval": 300,  // 5 minutes
  "summary_interval": 3600,   // 1 hour
  "max_retries": 3,
  "retry_delay": 30
}
```

Or via command-line arguments:

```bash
python main.py --backend-url https://api.smartvan-monitor.com/v1 --api-key YOUR_API_KEY_HERE
```

## Authentication

All requests to the backend include the following headers:

```
Content-Type: application/json
X-API-Key: YOUR_API_KEY_HERE
X-Van-ID: VAN001
```

## Offline Queue System

When the backend is unreachable:

1. All data is saved to files in the `output/backend_queue` directory
2. Each file follows the naming pattern: `{data_type}_{timestamp}_{uuid}.json`
3. Files are processed in order when connectivity is restored
4. Old heartbeats (>24 hours) are automatically discarded

## Integration with `BackendClient`

To use the backend client in custom code:

```python
from utils.backend_client import BackendClient

# Initialize the client
backend_client = BackendClient(config, output_dir)
backend_client.start()

# Send an event
backend_client.send_event({
    "event_type": "CUSTOM_EVENT",
    "data": "Custom event data"
}, priority="normal")

# Send performance metrics
backend_client.send_performance_metrics({
    "custom_metric": 42,
    "another_metric": 12.5
})

# Stop the client when done
backend_client.stop()
```

## Implementing a Backend Server

When implementing a backend server to receive data from SmartVan Monitor systems:

1. Create endpoints for `/heartbeat`, `/events`, `/summary`, and `/metrics`
2. Validate the API key in the `X-API-Key` header
3. Process the `van_id` from the `X-Van-ID` header for fleet management
4. Implement appropriate error handling and response codes
5. Consider implementing a heartbeat monitoring system to detect offline vans

## Security Considerations

1. **API Keys**: Never hardcode API keys in the application; use configuration files or environment variables
2. **Data Validation**: Always validate incoming data before processing
3. **HTTPS**: Ensure all communication uses HTTPS
4. **Access Control**: Implement proper access control for the backend API
5. **System Hash**: The `system_hash` in heartbeat data can be used to detect tampering

## Error Handling

The backend client implements automatic retries and error handling:

1. Failed requests are retried up to `max_retries` times (default: 3)
2. Each retry uses an increasing delay (`retry_delay` * attempt)
3. If all retries fail, the data is queued for later
4. Connection errors and timeouts are handled gracefully

## Development Best Practices

1. **Minimize Data**: Only send essential data to reduce bandwidth
2. **Prioritize Critical Events**: Human detection and inventory changes should have higher priority
3. **Batch Processing**: Use summary data where possible instead of individual events
4. **Test Offline Mode**: Ensure the system works properly when offline
5. **Monitor Queue Size**: Large queues may indicate connectivity issues

## Next Steps (Future Development)

1. **Real-time Updates**: Implement WebSocket connections for real-time updates
2. **Two-Way Communication**: Enable the backend to send commands to vans
3. **Advanced Authentication**: Implement OAuth or JWT for more secure authentication
4. **Data Compression**: Implement compression for larger data payloads
5. **Certificate Pinning**: Enhance security with certificate pinning
