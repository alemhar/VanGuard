"""
SmartVan Monitor - Van ID Utilities
-----------------------------------
Utilities for ensuring van_id is included in all event records.
This module helps maintain consistent van identification across the fleet.
"""

import os
import json
import glob
import logging
from pathlib import Path

logger = logging.getLogger("van_id_utils")

def ensure_van_id_in_event(event_data, van_id):
    """
    Ensure that the van_id is included in the event data.
    Modifies the event_data in place.
    
    Args:
        event_data: The event data dictionary
        van_id: The van ID to use if not already present
        
    Returns:
        The modified event data
    """
    if "van_id" not in event_data:
        event_data["van_id"] = van_id
        logger.debug(f"Added van_id {van_id} to event {event_data.get('event_id', 'unknown')}")
    return event_data

def fix_event_records(events_dir, van_id="VAN001"):
    """
    Add van_id to all existing event records in the events directory.
    
    Args:
        events_dir: Path to events directory
        van_id: Van ID to add to events
        
    Returns:
        Number of files updated
    """
    # Make sure events directory exists
    if not os.path.exists(events_dir):
        logger.warning(f"Events directory not found: {events_dir}")
        return 0
    
    # Get all JSON files in events directory
    event_files = glob.glob(os.path.join(events_dir, "*.json"))
    if not event_files:
        logger.info("No event files found")
        return 0
    
    logger.info(f"Adding van_id to {len(event_files)} event files...")
    updated_count = 0
    
    # Process each event file
    for event_file in event_files:
        try:
            # Read event data
            with open(event_file, 'r') as f:
                event_data = json.load(f)
            
            # Add van_id if not already present
            if "van_id" not in event_data:
                event_data["van_id"] = van_id
                
                # Write updated event data
                with open(event_file, 'w') as f:
                    json.dump(event_data, f, indent=4)
                updated_count += 1
        except Exception as e:
            logger.error(f"Error processing {event_file}: {e}")
    
    logger.info(f"Updated {updated_count} event files with van_id")
    return updated_count

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Default to 'output/events' directory
    events_dir = os.path.join("output", "events")
    
    # Allow custom directory and van_id via command line
    import argparse
    parser = argparse.ArgumentParser(description="Add van_id to event records")
    parser.add_argument("--events-dir", default=events_dir, help="Path to events directory")
    parser.add_argument("--van-id", default="VAN001", help="Van ID to add to events")
    args = parser.parse_args()
    
    # Run the utility
    fix_event_records(args.events_dir, args.van_id)
