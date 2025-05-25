"""
SmartVan Monitor - Backend Communication
----------------------------------------
Handles all communication with the backend server using an edge computing approach.
Only sends summarized data and heartbeats to minimize bandwidth usage.
"""

import os
import json
import time
import logging
import threading
import hashlib
import uuid
import requests
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

# Custom JSON encoder to handle NumPy types
class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyJSONEncoder, self).default(obj)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("backend_client")

class BackendClient:
    """Client for communicating with the SmartVan backend server."""
    
    def __init__(self, config: Dict[str, Any], output_dir: Path):
        """
        Initialize the backend client.
        
        Args:
            config: Configuration dictionary
            output_dir: Base output directory for local storage
        """
        # Load configuration
        self.config = config
        self.output_dir = output_dir
        
        # Get backend configuration with defaults
        self.backend_config = config.get("backend", {})
        self.server_url = self.backend_config.get("server_url", "https://api.smartvan-monitor.com/v1")
        self.api_key = self.backend_config.get("api_key", "")
        self.van_id = config.get("van_id", self.backend_config.get("van_id", "VAN001"))
        
        # Heartbeat settings
        self.heartbeat_interval = self.backend_config.get("heartbeat_interval", 300)  # 5 minutes default
        self.last_heartbeat_time = 0
        self.heartbeat_thread = None
        self.running = False
        
        # Summary data settings
        self.summary_interval = self.backend_config.get("summary_interval", 3600)  # 1 hour default
        self.last_summary_time = 0
        self.summary_thread = None
        
        # Connectivity and retry settings
        self.max_retries = self.backend_config.get("max_retries", 3)
        self.retry_delay = self.backend_config.get("retry_delay", 30)  # 30 seconds
        self.offline_queue_dir = output_dir / "backend_queue"
        self.offline_queue_dir.mkdir(exist_ok=True)
        
        # Security and tamper detection
        self.system_hash = self._calculate_system_hash()
        
        logger.info(f"Backend client initialized for van: {self.van_id}")
    
    def start(self):
        """Start the backend client threads."""
        if self.running:
            return
        
        self.running = True
        
        # Start heartbeat thread
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
        
        # Start summary thread
        self.summary_thread = threading.Thread(target=self._summary_loop, daemon=True)
        self.summary_thread.start()
        
        # Start offline queue processor
        threading.Thread(target=self._process_offline_queue, daemon=True).start()
        
        logger.info("Backend client started")
    
    def stop(self):
        """Stop the backend client threads."""
        self.running = False
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=1.0)
        if self.summary_thread:
            self.summary_thread.join(timeout=1.0)
        logger.info("Backend client stopped")
    
    def send_event(self, event_data: Dict[str, Any], priority: str = "normal"):
        """
        Send an important event to the backend.
        
        Args:
            event_data: Event data to send
            priority: Priority level (high, normal, low)
        """
        # Add van_id and timestamp if not present
        if "van_id" not in event_data:
            event_data["van_id"] = self.van_id
        
        if "timestamp" not in event_data:
            event_data["timestamp"] = time.time()
            
        # Add priority
        event_data["priority"] = priority
        
        # Try to send immediately if high priority, otherwise queue
        if priority == "high":
            success = self._send_to_backend("events", event_data)
            if not success:
                self._queue_for_later("events", event_data)
        else:
            self._queue_for_later("events", event_data)
    
    def send_performance_metrics(self, metrics_data: Dict[str, Any]):
        """
        Send performance metrics to the backend.
        
        Args:
            metrics_data: Performance metrics data
        """
        # Add van_id and timestamp if not present
        if "van_id" not in metrics_data:
            metrics_data["van_id"] = self.van_id
        
        if "timestamp" not in metrics_data:
            metrics_data["timestamp"] = time.time()
            
        # Queue metrics for later sending
        self._queue_for_later("metrics", metrics_data)
    
    def _heartbeat_loop(self):
        """Background thread that sends regular heartbeats to the backend."""
        while self.running:
            current_time = time.time()
            if current_time - self.last_heartbeat_time >= self.heartbeat_interval:
                self._send_heartbeat()
                self.last_heartbeat_time = current_time
            
            # Sleep for a short interval to avoid busy-waiting
            time.sleep(min(10, self.heartbeat_interval / 5))
    
    def _summary_loop(self):
        """Background thread that sends regular summary data to the backend."""
        while self.running:
            current_time = time.time()
            if current_time - self.last_summary_time >= self.summary_interval:
                self._send_summary()
                self.last_summary_time = current_time
            
            # Sleep for a short interval to avoid busy-waiting
            time.sleep(min(30, self.summary_interval / 10))
    
    def _send_heartbeat(self):
        """Send a heartbeat to the backend with basic system health information."""
        # Get system health metrics
        heartbeat_data = {
            "van_id": self.van_id,
            "timestamp": time.time(),
            "system_hash": self.system_hash,
            "uptime": self._get_uptime(),
            "free_disk_space": self._get_free_disk_space(),
            "cpu_usage": self._get_cpu_usage(),
            "memory_usage": self._get_memory_usage(),
            "cameras_active": self._get_camera_status(),
            "detectors_status": self._get_detector_status()
        }
        
        # Send heartbeat
        success = self._send_to_backend("heartbeat", heartbeat_data)
        if success:
            logger.debug(f"Sent heartbeat for van {self.van_id}")
        else:
            logger.warning(f"Failed to send heartbeat for van {self.van_id}")
            self._queue_for_later("heartbeat", heartbeat_data)
    
    def _send_summary(self):
        """Collect and send summary data to the backend."""
        # Time range for summary
        end_time = time.time()
        start_time = self.last_summary_time if self.last_summary_time > 0 else end_time - self.summary_interval
        
        # Collect summary data
        summary_data = self._collect_summary_data(start_time, end_time)
        if not summary_data:
            logger.debug("No summary data to send")
            return
        
        # Add van_id and timestamp
        summary_data["van_id"] = self.van_id
        summary_data["timestamp"] = end_time
        summary_data["start_time"] = start_time
        summary_data["end_time"] = end_time
        
        # Send summary
        success = self._send_to_backend("summary", summary_data)
        if success:
            logger.info(f"Sent summary data for van {self.van_id}")
        else:
            logger.warning(f"Failed to send summary data for van {self.van_id}")
            self._queue_for_later("summary", summary_data)
    
    def _send_to_backend(self, endpoint: str, data: Dict[str, Any]) -> bool:
        """
        Send data to the backend server.
        
        Args:
            endpoint: API endpoint
            data: Data to send
            
        Returns:
            Success status
        """
        # Check if API key is set
        if not self.api_key:
            logger.warning("API key not set, skipping backend communication")
            return False
        
        # Prepare URL and headers
        url = f"{self.server_url}/{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
            "X-Van-ID": self.van_id
        }
        
        # Try to send with retries
        for attempt in range(self.max_retries):
            try:
                # Convert data to JSON string first to ensure NumPy types are properly handled
                json_data = json.dumps(data, cls=NumpyJSONEncoder)
                
                # Ensure content-type is set to application/json
                headers["Content-Type"] = "application/json"
                
                response = requests.post(
                    url, 
                    headers=headers,
                    data=json_data,  # Use data instead of json parameter
                    timeout=10
                )
                
                if response.status_code >= 200 and response.status_code < 300:
                    return True
                else:
                    logger.warning(f"Backend request failed with status {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Backend request failed: {e}")
            
            # Wait before retry
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))
        
        return False
    
    def _queue_for_later(self, data_type: str, data: Dict[str, Any]):
        """
        Queue data for later sending when connectivity is restored.
        
        Args:
            data_type: Type of data (heartbeat, event, summary, metrics)
            data: Data to queue
        """
        # Create a unique filename
        timestamp = data.get("timestamp", time.time())
        dt = datetime.fromtimestamp(timestamp)
        timestamp_str = dt.strftime("%Y%m%d_%H%M%S")
        filename = f"{data_type}_{timestamp_str}_{uuid.uuid4().hex[:8]}.json"
        file_path = self.offline_queue_dir / filename
        
        # Save to queue
        try:
            with open(file_path, 'w') as f:
                json.dump({
                    "data_type": data_type,
                    "queued_at": time.time(),
                    "data": data
                }, f, cls=NumpyJSONEncoder)
            logger.debug(f"Queued {data_type} data for later sending")
        except Exception as e:
            logger.error(f"Failed to queue data: {e}")
    
    def _process_offline_queue(self):
        """Background thread that processes the offline queue when connectivity is restored."""
        while self.running:
            # Find queue files
            queue_files = list(self.offline_queue_dir.glob("*.json"))
            if not queue_files:
                time.sleep(60)  # Sleep longer if no files to process
                continue
            
            # Process each file
            for file_path in sorted(queue_files):
                try:
                    # Read queue item
                    with open(file_path, 'r') as f:
                        queue_item = json.load(f)
                    
                    data_type = queue_item["data_type"]
                    data = queue_item["data"]
                    
                    # Try to send
                    success = self._send_to_backend(data_type, data)
                    if success:
                        # Remove file on success
                        os.remove(file_path)
                        logger.info(f"Successfully sent queued {data_type} data")
                    else:
                        # Check age of item and possibly discard old heartbeats
                        queued_at = queue_item.get("queued_at", 0)
                        age_hours = (time.time() - queued_at) / 3600
                        
                        if data_type == "heartbeat" and age_hours > 24:
                            # Discard old heartbeats
                            os.remove(file_path)
                            logger.info(f"Discarded old heartbeat data (age: {age_hours:.1f} hours)")
                
                except Exception as e:
                    logger.error(f"Error processing queue item {file_path}: {e}")
            
            # Sleep before next check
            time.sleep(30)
    
    def _collect_summary_data(self, start_time: float, end_time: float) -> Dict[str, Any]:
        """
        Collect summary data for the given time range.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            Summary data dictionary
        """
        # Load event data from files
        clips_dir = self.output_dir / "clips"
        if not clips_dir.exists():
            return {}
        
        # Find all event metadata files
        event_files = list(clips_dir.glob("*.json"))
        events_in_range = []
        
        # Filter events in time range
        for event_file in event_files:
            try:
                with open(event_file, 'r') as f:
                    event_data = json.load(f)
                
                event_timestamp = event_data.get("timestamp", 0)
                if start_time <= event_timestamp <= end_time:
                    events_in_range.append(event_data)
            except Exception as e:
                logger.error(f"Error reading event file {event_file}: {e}")
        
        if not events_in_range:
            return {}
        
        # Build summary counts
        summary = {
            "event_count": len(events_in_range),
            "event_types": {},
            "camera_counts": {},
            "high_confidence_events": 0,
            "inventory_access_count": 0,
            "inventory_change_count": 0,
            "human_detection_count": 0
        }
        
        # Process each event
        for event in events_in_range:
            # Count by event type
            event_type = event.get("event_type", "UNKNOWN")
            summary["event_types"][event_type] = summary["event_types"].get(event_type, 0) + 1
            
            # Count by camera
            camera = event.get("camera", "UNKNOWN")
            summary["camera_counts"][camera] = summary["camera_counts"].get(camera, 0) + 1
            
            # Count high confidence events (human detection > 0.7)
            human_data = event.get("human", {})
            if human_data.get("detected", False) and human_data.get("confidence", 0) > 0.7:
                summary["high_confidence_events"] += 1
                summary["human_detection_count"] += 1
            
            # Count inventory events
            inventory_data = event.get("inventory", {})
            if inventory_data.get("access_detected", False):
                summary["inventory_access_count"] += 1
            if inventory_data.get("change_detected", False):
                summary["inventory_change_count"] += 1
        
        return summary
    
    def _calculate_system_hash(self) -> str:
        """
        Calculate a hash of critical system files to detect tampering.
        
        Returns:
            System hash string
        """
        # List of critical files to check
        main_dir = Path(__file__).parent.parent
        critical_files = [
            main_dir / "main.py",
            main_dir / "detection" / "inventory_detection.py",
            main_dir / "detection" / "motion_detection.py",
            main_dir / "detection" / "enhanced_monitor.py",
            Path(__file__)  # Include this backend client
        ]
        
        # Calculate combined hash
        hasher = hashlib.sha256()
        
        for file_path in critical_files:
            if file_path.exists():
                try:
                    with open(file_path, 'rb') as f:
                        file_data = f.read()
                        hasher.update(file_data)
                except Exception as e:
                    logger.error(f"Error hashing file {file_path}: {e}")
            
        return hasher.hexdigest()
    
    def _get_uptime(self) -> float:
        """Get system uptime in seconds."""
        try:
            # This is a simple approximation - we'd need OS-specific code for true uptime
            return time.time() - self.last_heartbeat_time if self.last_heartbeat_time > 0 else 0
        except Exception:
            return 0
    
    def _get_free_disk_space(self) -> float:
        """Get free disk space in megabytes."""
        try:
            # Get disk space information for the output directory
            total, used, free = os.statvfs(self.output_dir)
            return (free * total.f_frsize) / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0
    
    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            return 0
    
    def _get_memory_usage(self) -> float:
        """Get memory usage percentage."""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 0
    
    def _get_camera_status(self) -> Dict[str, bool]:
        """Get status of all cameras."""
        # This would be populated from the actual camera manager
        # For now, return a placeholder
        return {"camera_count": 0, "all_active": False}
    
    def _get_detector_status(self) -> Dict[str, Any]:
        """Get status of all detectors."""
        # This would be populated from the actual detectors
        # For now, return a placeholder
        return {"detectors_active": True}
