"""
SmartVan Monitor - Test Runner
-----------------------------
Main script to run all SmartVan Monitor tests or specific test modules.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("smartvan_test_results.log")
    ]
)
logger = logging.getLogger("test_runner")

# Import test modules
from testing.test_framework import framework
from testing.test_inventory_detection import InventoryDetectionTests
from testing.test_vibration_analyzer import VibrationAnalyzerTests

def setup_test_environment():
    """Set up the testing environment"""
    # Create necessary directories
    Path("testing/test_data").mkdir(exist_ok=True, parents=True)
    Path("testing/output").mkdir(exist_ok=True, parents=True)
    Path("testing/test_results").mkdir(exist_ok=True, parents=True)
    
    logger.info("Test environment set up successfully")

def run_all_tests(generate_report=True):
    """Run all available tests"""
    logger.info("Starting all SmartVan Monitor tests")
    start_time = time.time()
    
    # Initialize test modules
    inventory_tests = InventoryDetectionTests()
    vibration_tests = VibrationAnalyzerTests()
    
    # Run all tests
    inventory_tests.run_all_tests()
    vibration_tests.run_all_tests()
    
    # Generate combined report if requested
    if generate_report:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = framework.generate_report(f"smartvan_full_test_report_{timestamp}.md")
        logger.info(f"Combined test report generated: {report_path}")
    
    # Log summary
    duration = time.time() - start_time
    logger.info(f"All tests completed in {duration:.2f} seconds")
    
    # Count results by status
    results = framework.results
    pass_count = sum(1 for r in results if r.status == "PASS")
    fail_count = sum(1 for r in results if r.status == "FAIL")
    warn_count = sum(1 for r in results if r.status == "WARNING")
    
    logger.info(f"Test summary: {len(results)} tests, {pass_count} passed, {fail_count} failed, {warn_count} warnings")
    
    return pass_count, fail_count, warn_count

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SmartVan Monitor Test Runner")
    
    parser.add_argument(
        "--module", 
        choices=["inventory", "vibration", "all"],
        default="all",
        help="Test module to run (default: all)"
    )
    
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate test report"
    )
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Set up test environment
    setup_test_environment()
    
    # Initialize counters
    pass_count, fail_count, warn_count = 0, 0, 0
    
    if args.module == "all":
        pass_count, fail_count, warn_count = run_all_tests(args.report)
    elif args.module == "inventory":
        logger.info("Running inventory detection tests")
        tests = InventoryDetectionTests()
        tests.run_all_tests()
        if args.report:
            framework.generate_report("inventory_detection_report.md")
        # Count results for this specific module
        results = framework.results
        pass_count = sum(1 for r in results if r.status == "PASS")
        fail_count = sum(1 for r in results if r.status == "FAIL")
        warn_count = sum(1 for r in results if r.status == "WARNING")
    elif args.module == "vibration":
        logger.info("Running vibration analyzer tests")
        tests = VibrationAnalyzerTests()
        tests.run_all_tests()
        if args.report:
            framework.generate_report("vibration_analyzer_report.md")
        # Count results for this specific module
        results = framework.results
        pass_count = sum(1 for r in results if r.status == "PASS")
        fail_count = sum(1 for r in results if r.status == "FAIL")
        warn_count = sum(1 for r in results if r.status == "WARNING")
    
    # Log summary for specific modules too
    logger.info(f"Test summary: {len(framework.results)} tests, {pass_count} passed, {fail_count} failed, {warn_count} warnings")
    
    # Set exit code based on test results
    if fail_count > 0:
        return 1
    else:
        return 0

if __name__ == "__main__":
    sys.exit(main())
