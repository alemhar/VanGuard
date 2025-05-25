"""
SmartVan Monitor - Test Framework
--------------------------------
Provides a comprehensive framework for testing and validating the SmartVan Monitor system.
Focuses on testing core detection capabilities with an emphasis on:
1. Motion detection accuracy
2. Vibration filtering
3. Inventory change detection under varying lighting conditions
4. Human presence detection
"""

import cv2
import numpy as np
import os
import json
import time
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable

# Custom JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("smartvan_test_results.log")
    ]
)
logger = logging.getLogger("smartvan_test")

class TestResult:
    """Class to store and report test results"""
    
    STATUS_PASS = "PASS"
    STATUS_FAIL = "FAIL"
    STATUS_WARNING = "WARNING"
    
    def __init__(self, test_name: str, component: str):
        """
        Initialize a test result.
        
        Args:
            test_name: Name of the test
            component: Component being tested
        """
        self.test_name = test_name
        self.component = component
        self.status = None
        self.details = {}
        self.start_time = time.time()
        self.end_time = None
        self.duration = None
    
    def complete(self, status: str, details: Dict[str, Any] = None):
        """
        Complete the test with results.
        
        Args:
            status: Test status (PASS, FAIL, WARNING)
            details: Additional test details and metrics
        """
        self.status = status
        if details:
            self.details.update(details)
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        
        # Log the result
        status_msg = f"Test '{self.test_name}' ({self.component}): {self.status}"
        if self.status == self.STATUS_PASS:
            logger.info(status_msg)
        elif self.status == self.STATUS_WARNING:
            logger.warning(status_msg)
        else:
            logger.error(status_msg)
            
        # Log additional details
        for key, value in self.details.items():
            logger.info(f"  {key}: {value}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the test result to a dictionary for serialization"""
        return {
            "test_name": self.test_name,
            "component": self.component,
            "status": self.status,
            "details": self.details,
            "duration": self.duration,
            "timestamp": datetime.datetime.now().isoformat()
        }


class SmartVanTestFramework:
    """Framework for testing SmartVan Monitor system components"""
    
    def __init__(self, output_dir: str = "test_results"):
        """
        Initialize the test framework.
        
        Args:
            output_dir: Directory to store test results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Store test results
        self.results = []
        
        # Test data paths
        self.test_data_dir = Path("testing/test_data")
        self.test_data_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"SmartVan Test Framework initialized. Output dir: {self.output_dir}")
    
    def run_test(self, test_name: str, component: str, test_func: Callable, **kwargs) -> TestResult:
        """
        Run a test and record results.
        
        Args:
            test_name: Name of the test
            component: Component being tested
            test_func: Function to execute for the test
            **kwargs: Additional arguments to pass to the test function
            
        Returns:
            TestResult object
        """
        logger.info(f"Starting test: {test_name} ({component})")
        
        # Create test result
        result = TestResult(test_name, component)
        
        try:
            # Run the test
            test_status, test_details = test_func(**kwargs)
            result.complete(test_status, test_details)
        except Exception as e:
            logger.exception(f"Test '{test_name}' failed with exception")
            result.complete(TestResult.STATUS_FAIL, {"error": str(e)})
        
        # Store the result
        self.results.append(result)
        return result
    
    def save_results(self, filename: str = None):
        """
        Save test results to a JSON file.
        
        Args:
            filename: Optional filename for results
        """
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.json"
        
        result_path = self.output_dir / filename
        
        # Convert results to dictionaries
        result_dicts = [r.to_dict() for r in self.results]
        
        # Save to file with custom NumPy encoder
        with open(result_path, 'w') as f:
            json.dump(result_dicts, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Test results saved to {result_path}")
        
    def generate_report(self, filename: str = None):
        """
        Generate a markdown report of test results.
        
        Args:
            filename: Optional filename for report
        """
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_report_{timestamp}.md"
        
        report_path = self.output_dir / filename
        
        # Count results by status
        pass_count = sum(1 for r in self.results if r.status == TestResult.STATUS_PASS)
        fail_count = sum(1 for r in self.results if r.status == TestResult.STATUS_FAIL)
        warn_count = sum(1 for r in self.results if r.status == TestResult.STATUS_WARNING)
        
        with open(report_path, 'w') as f:
            # Write header
            f.write("# SmartVan Monitor System Test Report\n\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write summary
            f.write("## Summary\n\n")
            f.write(f"- Total tests: {len(self.results)}\n")
            f.write(f"- Passed: {pass_count}\n")
            f.write(f"- Failed: {fail_count}\n")
            f.write(f"- Warnings: {warn_count}\n\n")
            
            # Write results by component
            f.write("## Results by Component\n\n")
            
            components = sorted(set(r.component for r in self.results))
            for component in components:
                f.write(f"### {component}\n\n")
                
                # Filter results for this component
                component_results = [r for r in self.results if r.component == component]
                
                # Create a table
                f.write("| Test | Status | Duration (s) | Details |\n")
                f.write("|------|--------|--------------|--------|\n")
                
                for r in component_results:
                    # Format details
                    if r.details:
                        details_str = "; ".join(f"{k}: {v}" for k, v in r.details.items())
                    else:
                        details_str = "-"
                    
                    # Format status (avoid Unicode emoji to prevent encoding issues)
                    if r.status == TestResult.STATUS_PASS:
                        status_str = "[PASS]"
                    elif r.status == TestResult.STATUS_WARNING:
                        status_str = "[WARNING]"
                    else:
                        status_str = "[FAIL]"
                    
                    f.write(f"| {r.test_name} | {status_str} | {r.duration:.2f} | {details_str} |\n")
                
                f.write("\n")
        
        logger.info(f"Test report generated at {report_path}")
        return report_path


# Global instance for convenience
framework = SmartVanTestFramework()
"""Global test framework instance for convenience"""
