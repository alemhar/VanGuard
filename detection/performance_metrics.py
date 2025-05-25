"""
SmartVan Monitor - Performance Metrics Module
--------------------------------------------
This module implements performance tracking and analysis for the SmartVan Monitor system.

Key features:
- Track detection performance (true/false positives, false negatives)
- Generate performance reports and visualizations
- Provide tools for analyzing system performance under different conditions
"""

import os
import time
import json
import csv
import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logger = logging.getLogger("performance_metrics")

class PerformanceTracker:
    """
    Class for tracking and analyzing detection performance metrics.
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the performance tracker.
        
        Args:
            output_dir: Directory to store performance metrics data
        """
        self.output_dir = Path(output_dir)
        self.metrics_dir = self.output_dir / "performance_metrics"
        self.metrics_dir.mkdir(exist_ok=True)
        
        # Create performance log files if they don't exist
        self.performance_log_path = self.metrics_dir / "detection_performance.csv"
        self.feedback_log_path = self.metrics_dir / "user_feedback.csv"
        
        # Visualization output directory
        self.visualization_dir = self.metrics_dir / "visualizations"
        self.visualization_dir.mkdir(exist_ok=True)
        
        # State tracking for context
        self.environment_state = {"light_level": "normal", "time_of_day": "day"}
        self.vehicle_state = {"is_moving": False, "movement_type": "stationary"}
        
        # Initialize tracking infrastructure
        self._initialize_tracking()
        
        logger.info(f"Performance tracker initialized, logs at {self.metrics_dir}")
    
    def _initialize_tracking(self) -> None:
        """Initialize the performance tracking infrastructure."""
        # Create performance log file if it doesn't exist
        if not self.performance_log_path.exists():
            with open(self.performance_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'camera', 'zone', 'event_type', 'confidence',
                    'change_percentage', 'contour_count', 'dominant_size',
                    'lighting_difference', 'is_true_positive', 'false_positive_reason',
                    'false_negative_reason', 'vehicle_moving', 'light_level'
                ])
                
        # Create user feedback log if it doesn't exist
        if not self.feedback_log_path.exists():
            with open(self.feedback_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'detection_id', 'feedback_type', 'is_correct',
                    'notes', 'camera', 'zone'
                ])
    
    def log_detection(self, detection_result: Dict[str, Any], 
                     is_true_positive: bool = None, 
                     false_positive_reason: str = None,
                     false_negative_reason: str = None,
                     notes: str = None) -> None:
        """
        Log detection results for performance analysis.
        
        Args:
            detection_result: The detection result dictionary
            is_true_positive: If manually verified, was this a true detection
            false_positive_reason: If false positive, reason why
            false_negative_reason: If false negative, reason why
            notes: Optional notes about this detection
        """
        try:
            timestamp = time.time()
            camera = detection_result.get("camera_name", "unknown")
            zone = detection_result.get("zone_id", "unknown")
            event_type = detection_result.get("event_type", "UNKNOWN")
            confidence = detection_result.get("confidence", 0)
            change_percentage = detection_result.get("change_percentage", 0)
            contour_count = detection_result.get("contour_count", 0)
            dominant_size = detection_result.get("dominant_size", "unknown")
            
            # Get analysis data if available
            analysis = detection_result.get("analysis", {})
            lighting_difference = analysis.get("lighting_difference", 0)
            
            # Add environmental context
            vehicle_moving = self.vehicle_state.get("is_moving", False)
            light_level = self.environment_state.get("light_level", "unknown")
            
            # Log to CSV
            with open(self.performance_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, camera, zone, event_type, confidence,
                    change_percentage, contour_count, dominant_size,
                    lighting_difference, is_true_positive, false_positive_reason,
                    false_negative_reason, vehicle_moving, light_level
                ])
                
            logger.debug(f"Logged detection performance: {camera}/{zone}, {event_type}")
            
        except Exception as e:
            logger.error(f"Error logging detection performance: {e}")
    
    def log_user_feedback(self, detection_id: str, feedback_type: str, 
                         is_correct: bool, notes: str = None,
                         camera: str = None, zone: str = None) -> None:
        """
        Log user feedback about a detection result.
        
        Args:
            detection_id: Identifier for the detection event
            feedback_type: Type of feedback (e.g., 'manual_review', 'alert_response')
            is_correct: Whether the detection was correct according to user
            notes: Optional notes from the user
            camera: Camera name (if known)
            zone: Zone ID (if known)
        """
        try:
            timestamp = time.time()
            
            # Log to CSV
            with open(self.feedback_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, detection_id, feedback_type, is_correct,
                    notes, camera, zone
                ])
                
            logger.info(f"Logged user feedback for detection {detection_id}: {is_correct}")
            
        except Exception as e:
            logger.error(f"Error logging user feedback: {e}")
    
    def update_environment_state(self, light_level: str = None, 
                               time_of_day: str = None) -> None:
        """
        Update the tracked environment state.
        
        Args:
            light_level: Current light level (e.g., 'dark', 'normal', 'bright')
            time_of_day: Current time of day (e.g., 'day', 'night')
        """
        if light_level is not None:
            self.environment_state["light_level"] = light_level
            
        if time_of_day is not None:
            self.environment_state["time_of_day"] = time_of_day
    
    def update_vehicle_state(self, is_moving: bool = None, 
                           movement_type: str = None) -> None:
        """
        Update the tracked vehicle state.
        
        Args:
            is_moving: Whether the vehicle is currently moving
            movement_type: Type of movement (e.g., 'stationary', 'driving', 'stop_and_go')
        """
        if is_moving is not None:
            self.vehicle_state["is_moving"] = is_moving
            
        if movement_type is not None:
            self.vehicle_state["movement_type"] = movement_type
    
    def generate_report(self, start_time: float = None, 
                      end_time: float = None,
                      output_format: str = "html") -> str:
        """
        Generate a performance report for the specified time period.
        
        Args:
            start_time: Start of analysis period (default: 24 hours ago)
            end_time: End of analysis period (default: now)
            output_format: Format of the report ('html', 'markdown', 'json')
            
        Returns:
            Report in the specified format
        """
        # Set default time range if not specified
        if end_time is None:
            end_time = time.time()
            
        if start_time is None:
            start_time = end_time - (24 * 60 * 60)  # 24 hours ago
            
        try:
            # Load performance data
            if not self.performance_log_path.exists():
                return "No performance data available"
                
            # Load data into pandas DataFrame for analysis
            df = pd.read_csv(self.performance_log_path)
            
            # Filter by time range
            df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
            
            if len(df) == 0:
                return "No performance data in the specified time range"
                
            # Calculate metrics
            total_detections = len(df)
            verified_detections = df['is_true_positive'].notnull().sum()
            true_positives = df[df['is_true_positive'] == True].shape[0]
            false_positives = df[df['is_true_positive'] == False].shape[0]
            
            # If we have verified detections, calculate precision
            if verified_detections > 0:
                precision = true_positives / verified_detections
            else:
                precision = "Unknown (no verified detections)"
                
            # Group by various dimensions
            by_event_type = df.groupby('event_type').size().to_dict()
            by_zone = df.groupby('zone').size().to_dict()
            by_dominant_size = df.groupby('dominant_size').size().to_dict()
            
            # False positive analysis
            fp_reasons = df[df['is_true_positive'] == False].groupby('false_positive_reason').size().to_dict()
            
            # Generate report based on format
            if output_format == "json":
                report = {
                    "time_range": {"start": start_time, "end": end_time},
                    "metrics": {
                        "total_detections": total_detections,
                        "verified_detections": verified_detections,
                        "true_positives": true_positives,
                        "false_positives": false_positives,
                        "precision": precision
                    },
                    "by_event_type": by_event_type,
                    "by_zone": by_zone,
                    "by_dominant_size": by_dominant_size,
                    "false_positive_reasons": fp_reasons
                }
                return json.dumps(report, indent=2)
            elif output_format == "markdown":
                return self._format_report_markdown({
                    "time_range": {"start": start_time, "end": end_time},
                    "metrics": {
                        "total_detections": total_detections,
                        "verified_detections": verified_detections,
                        "true_positives": true_positives,
                        "false_positives": false_positives,
                        "precision": precision
                    },
                    "by_event_type": by_event_type,
                    "by_zone": by_zone,
                    "by_dominant_size": by_dominant_size,
                    "false_positive_reasons": fp_reasons
                })
            else:  # html default
                return self._format_report_html({
                    "time_range": {"start": start_time, "end": end_time},
                    "metrics": {
                        "total_detections": total_detections,
                        "verified_detections": verified_detections,
                        "true_positives": true_positives,
                        "false_positives": false_positives,
                        "precision": precision
                    },
                    "by_event_type": by_event_type,
                    "by_zone": by_zone,
                    "by_dominant_size": by_dominant_size,
                    "false_positive_reasons": fp_reasons
                })
                
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return f"Error generating report: {e}"
    
    def _format_report_markdown(self, data: Dict[str, Any]) -> str:
        """
        Format performance report as markdown.
        
        Args:
            data: Report data dictionary
            
        Returns:
            Markdown formatted report
        """
        # Format timestamp range
        start_time = datetime.fromtimestamp(data["time_range"]["start"]).strftime("%Y-%m-%d %H:%M")
        end_time = datetime.fromtimestamp(data["time_range"]["end"]).strftime("%Y-%m-%d %H:%M")
        
        report = []
        report.append("# SmartVan Inventory Detection Performance Report")
        report.append(f"**Time Range:** {start_time} to {end_time}\n")
        
        # Summary metrics
        report.append("## Summary Metrics")
        report.append(f"- **Total Detections:** {data['metrics']['total_detections']}")
        report.append(f"- **Verified Detections:** {data['metrics']['verified_detections']}")
        report.append(f"- **True Positives:** {data['metrics']['true_positives']}")
        report.append(f"- **False Positives:** {data['metrics']['false_positives']}")
        report.append(f"- **Precision:** {data['metrics']['precision']}\n")
        
        # Event types
        report.append("## Detection by Event Type")
        for event_type, count in data["by_event_type"].items():
            report.append(f"- **{event_type}:** {count}")
        report.append("")
        
        # Zones
        report.append("## Detection by Zone")
        for zone, count in data["by_zone"].items():
            report.append(f"- **{zone}:** {count}")
        report.append("")
        
        # Size distribution
        report.append("## Detection by Item Size")
        for size, count in data["by_dominant_size"].items():
            report.append(f"- **{size}:** {count}")
        report.append("")
        
        # False positive reasons
        if data["false_positive_reasons"]:
            report.append("## False Positive Analysis")
            for reason, count in data["false_positive_reasons"].items():
                report.append(f"- **{reason}:** {count}")
        
        return "\n".join(report)
    
    def _format_report_html(self, data: Dict[str, Any]) -> str:
        """
        Format performance report as HTML.
        
        Args:
            data: Report data dictionary
            
        Returns:
            HTML formatted report
        """
        # Format timestamp range
        start_time = datetime.fromtimestamp(data["time_range"]["start"]).strftime("%Y-%m-%d %H:%M")
        end_time = datetime.fromtimestamp(data["time_range"]["end"]).strftime("%Y-%m-%d %H:%M")
        
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html>")
        html.append("<head>")
        html.append("<title>SmartVan Performance Report</title>")
        html.append("<style>")
        html.append("body { font-family: Arial, sans-serif; margin: 20px; }")
        html.append("h1 { color: #2c3e50; }")
        html.append("h2 { color: #3498db; }")
        html.append(".metric { margin-bottom: 5px; }")
        html.append(".chart { width: 100%; height: 300px; margin-bottom: 20px; }")
        html.append("</style>")
        html.append("</head>")
        html.append("<body>")
        
        # Header
        html.append(f"<h1>SmartVan Inventory Detection Performance Report</h1>")
        html.append(f"<p><strong>Time Range:</strong> {start_time} to {end_time}</p>")
        
        # Summary metrics
        html.append("<h2>Summary Metrics</h2>")
        html.append("<div class='metrics'>")
        html.append(f"<p class='metric'><strong>Total Detections:</strong> {data['metrics']['total_detections']}</p>")
        html.append(f"<p class='metric'><strong>Verified Detections:</strong> {data['metrics']['verified_detections']}</p>")
        html.append(f"<p class='metric'><strong>True Positives:</strong> {data['metrics']['true_positives']}</p>")
        html.append(f"<p class='metric'><strong>False Positives:</strong> {data['metrics']['false_positives']}</p>")
        html.append(f"<p class='metric'><strong>Precision:</strong> {data['metrics']['precision']}</p>")
        html.append("</div>")
        
        # Event types
        html.append("<h2>Detection by Event Type</h2>")
        html.append("<ul>")
        for event_type, count in data["by_event_type"].items():
            html.append(f"<li><strong>{event_type}:</strong> {count}</li>")
        html.append("</ul>")
        
        # Zones
        html.append("<h2>Detection by Zone</h2>")
        html.append("<ul>")
        for zone, count in data["by_zone"].items():
            html.append(f"<li><strong>{zone}:</strong> {count}</li>")
        html.append("</ul>")
        
        # Size distribution
        html.append("<h2>Detection by Item Size</h2>")
        html.append("<ul>")
        for size, count in data["by_dominant_size"].items():
            html.append(f"<li><strong>{size}:</strong> {count}</li>")
        html.append("</ul>")
        
        # False positive reasons
        if data["false_positive_reasons"]:
            html.append("<h2>False Positive Analysis</h2>")
            html.append("<ul>")
            for reason, count in data["false_positive_reasons"].items():
                html.append(f"<li><strong>{reason}:</strong> {count}</li>")
            html.append("</ul>")
        
        # Close tags
        html.append("</body>")
        html.append("</html>")
        
        return "\n".join(html)
    
    def generate_visualizations(self, start_time: float = None, 
                               end_time: float = None) -> Dict[str, str]:
        """
        Generate visualizations for performance metrics.
        
        Args:
            start_time: Start of analysis period (default: 24 hours ago)
            end_time: End of analysis period (default: now)
            
        Returns:
            Dictionary mapping chart types to file paths
        """
        # Set default time range if not specified
        if end_time is None:
            end_time = time.time()
            
        if start_time is None:
            start_time = end_time - (24 * 60 * 60)  # 24 hours ago
            
        try:
            # Load performance data
            if not self.performance_log_path.exists():
                return {"error": "No performance data available"}
                
            # Load data into pandas DataFrame for analysis
            df = pd.read_csv(self.performance_log_path)
            
            # Filter by time range
            df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
            
            if len(df) == 0:
                return {"error": "No performance data in the specified time range"}
                
            # Generate visualizations
            file_paths = {}
            
            # 1. Detection counts by event type
            plt.figure(figsize=(10, 6))
            event_type_counts = df['event_type'].value_counts()
            event_type_counts.plot(kind='bar', color='skyblue')
            plt.title('Detections by Event Type')
            plt.xlabel('Event Type')
            plt.ylabel('Count')
            plt.tight_layout()
            event_type_path = self.visualization_dir / "event_type_counts.png"
            plt.savefig(event_type_path)
            plt.close()
            file_paths["event_type"] = str(event_type_path)
            
            # 2. Detection accuracy (verified detections)
            if df['is_true_positive'].notnull().sum() > 0:
                plt.figure(figsize=(8, 8))
                accuracy_counts = df['is_true_positive'].value_counts()
                plt.pie(accuracy_counts, labels=['True Positive', 'False Positive'], 
                       autopct='%1.1f%%', colors=['#4CAF50', '#F44336'])
                plt.title('Detection Accuracy')
                plt.tight_layout()
                accuracy_path = self.visualization_dir / "accuracy_pie.png"
                plt.savefig(accuracy_path)
                plt.close()
                file_paths["accuracy"] = str(accuracy_path)
            
            # 3. Detections by size category
            plt.figure(figsize=(10, 6))
            size_counts = df['dominant_size'].value_counts()
            size_counts.plot(kind='bar', color='lightgreen')
            plt.title('Detections by Size Category')
            plt.xlabel('Size Category')
            plt.ylabel('Count')
            plt.tight_layout()
            size_path = self.visualization_dir / "size_counts.png"
            plt.savefig(size_path)
            plt.close()
            file_paths["size"] = str(size_path)
            
            # 4. False positive reasons (if any)
            fp_df = df[df['is_true_positive'] == False]
            if len(fp_df) > 0 and fp_df['false_positive_reason'].notnull().sum() > 0:
                plt.figure(figsize=(10, 6))
                fp_reasons = fp_df['false_positive_reason'].value_counts()
                fp_reasons.plot(kind='bar', color='salmon')
                plt.title('False Positive Reasons')
                plt.xlabel('Reason')
                plt.ylabel('Count')
                plt.tight_layout()
                fp_path = self.visualization_dir / "false_positive_reasons.png"
                plt.savefig(fp_path)
                plt.close()
                file_paths["false_positive_reasons"] = str(fp_path)
            
            return file_paths
                
        except Exception as e:
            logger.error(f"Error generating performance visualizations: {e}")
            return {"error": f"Error generating visualizations: {e}"}
