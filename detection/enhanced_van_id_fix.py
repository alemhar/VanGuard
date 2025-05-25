"""
SmartVan Monitor - Van ID Fix
----------------------------
This small utility adds van_id to event records by modifying the event_framework.py file.
Run this once to update your events to include fleet identification.
"""

import os
import json
import glob
from pathlib import Path

def add_van_id_to_events(events_dir, van_id="VAN001"):
    """
    Add van_id to all event records in the events directory.
    
    Args:
        events_dir: Path to events directory
        van_id: Van ID to add to events (from config)
    """
    # Make sure events directory exists
    if not os.path.exists(events_dir):
        print(f"Events directory not found: {events_dir}")
        return
    
    # Get all JSON files in events directory
    event_files = glob.glob(os.path.join(events_dir, "*.json"))
    if not event_files:
        print("No event files found")
        return
    
    print(f"Adding van_id to {len(event_files)} event files...")
    
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
        except Exception as e:
            print(f"Error processing {event_file}: {e}")
    
    print("Done adding van_id to event files")

if __name__ == "__main__":
    # Default to 'output/events' directory
    events_dir = os.path.join("output", "events")
    
    # Allow custom directory and van_id via command line
    import argparse
    parser = argparse.ArgumentParser(description="Add van_id to event records")
    parser.add_argument("--events-dir", default=events_dir, help="Path to events directory")
    parser.add_argument("--van-id", default="VAN001", help="Van ID to add to events")
    args = parser.parse_args()
    
    # Run the utility
    add_van_id_to_events(args.events_dir, args.van_id)
