#!/usr/bin/env python3
"""
Script to extract and deduplicate competitor names from CSV files in competitor_names directory.
"""

import csv
import os
from pathlib import Path

def extract_competitor_names():
    """Extract all competitor names from CSV files and return deduplicated list."""
    competitor_names = set()
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    competitor_dir = script_dir / "competitor_names"
    
    # Check if directory exists
    if not competitor_dir.exists():
        print(f"Error: Directory '{competitor_dir}' not found.")
        return []
    
    # Process all CSV files in the directory
    csv_files = list(competitor_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in '{competitor_dir}'.")
        return []
    
    print(f"Processing {len(csv_files)} CSV file(s)...")
    
    for csv_file in csv_files:
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                # Get header row (first row)
                headers = next(reader, None)
                
                if headers:
                    # Extract competitor names (skip "DateTime" and "Annotations")
                    for header in headers:
                        header = header.strip()
                        # Skip empty, DateTime, Annotations, and BOM artifacts
                        if (header and 
                            header not in ["DateTime", "Annotations", "Date"] and
                            not header.startswith('\ufeff') and
                            not header.lower().startswith('remitly')):
                            competitor_names.add(header)
        except Exception as e:
            print(f"Warning: Could not process {csv_file.name}: {e}")
    
    # Convert to sorted list for consistent output
    competitor_list = sorted(list(competitor_names))
    
    return competitor_list

if __name__ == "__main__":
    competitors = extract_competitor_names()
    
    print(f"\nFound {len(competitors)} unique competitor names:\n")
    print(competitors)
    
    # Also print as a Python list format for easy copying
    print("\n\nAs Python list:")
    print(competitors)

