#!/usr/bin/env python3
"""
Data Preparation and CSV Generation
===================================
This module handles:
1. Splitting enriched data into separate CSV files by breach name
2. Adding simulated breach times
3. Merging all CSV files into final training data
"""

import json
import pandas as pd
import os
from datetime import datetime, timedelta
import random
from typing import List


# =============================================================================
# Split Data into Separate CSV Files by Breach Name
# =============================================================================

def split_by_breach_name(
    input_file: str = "trainData_enriched_with_honeywords.json",
    output_dir: str = "breach_csv_files"
):
    """
    Split trainData_enriched_with_honeywords.json into separate CSV files by breach name.
    
    Each CSV contains:
    - email
    - password  
    - honeywords (as comma-separated string)
    - breach_date
    - breach_description
    - pwn_count
    - data_classes (as comma-separated string)
    
    Args:
        input_file: Input JSON file with enriched breach data
        output_dir: Output directory for CSV files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Group by breach source
    breach_groups = {}
    for entry in data:
        breach_name = entry.get("breach_source", "Unknown")
        
        if breach_name not in breach_groups:
            breach_groups[breach_name] = []
        
        breach_groups[breach_name].append(entry)
    
    # Save each breach as a separate CSV
    for breach_name, entries in breach_groups.items():
        # Sanitize filename
        safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in breach_name)
        output_file = os.path.join(output_dir, f"{safe_name}.csv")
        
        # Prepare data for CSV
        rows = []
        for entry in entries:
            row = {
                "email": entry.get("email", ""),
                "password": entry.get("password", ""),
                "honeywords": ",".join(entry.get("honeywords", [])),
                "breach_date": entry.get("breach_date", ""),
                "breach_description": entry.get("breach_description", ""),
                "pwn_count": entry.get("pwn_count", 0),
                "data_classes": ",".join(entry.get("data_classes", []))
            }
            rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False, encoding="utf-8")
        
        print(f"Created {output_file} with {len(rows)} entries")
    
    print(f"\nTotal breaches: {len(breach_groups)}")
    return breach_groups


# =============================================================================
# Add Breach Time Simulation
# =============================================================================

def add_breach_time_to_csvs(
    input_dir: str = "breach_csv_files",
    output_dir: str = "breach_csv_files_with_time"
):
    """
    Add breachTime column to all CSV files in breach_csv_files directory.
    
    Simulates realistic breach times based on breach_date with some randomization.
    
    Args:
        input_dir: Input directory with CSV files
        output_dir: Output directory for CSVs with breach times
    """
    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        input_path = os.path.join(input_dir, csv_file)
        output_path = os.path.join(output_dir, csv_file)
        
        # Read CSV
        df = pd.read_csv(input_path)
        
        # Add breach time
        breach_times = []
        
        for _, row in df.iterrows():
            breach_date_str = row.get('breach_date', '')
            
            if breach_date_str and breach_date_str != '':
                try:
                    # Parse breach date
                    breach_date = datetime.strptime(breach_date_str, '%Y-%m-%d')
                    
                    # Add random hours/minutes for more realistic simulation
                    random_hours = random.randint(0, 23)
                    random_minutes = random.randint(0, 59)
                    random_seconds = random.randint(0, 59)
                    
                    breach_time = breach_date + timedelta(
                        hours=random_hours,
                        minutes=random_minutes,
                        seconds=random_seconds
                    )
                    
                    # Add some random days variation (±7 days)
                    days_variation = random.randint(-7, 7)
                    breach_time = breach_time + timedelta(days=days_variation)
                    
                    breach_times.append(breach_time.strftime('%Y-%m-%d %H:%M:%S'))
                    
                except Exception as e:
                    # If parsing fails, use a default time
                    default_time = datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1000))
                    breach_times.append(default_time.strftime('%Y-%m-%d %H:%M:%S'))
            else:
                # No breach date, use random time in past
                default_time = datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1000))
                breach_times.append(default_time.strftime('%Y-%m-%d %H:%M:%S'))
        
        df['breachTime'] = breach_times
        
        # Save with breach time
        df.to_csv(output_path, index=False, encoding="utf-8")
        
        print(f"Processed {csv_file}")
    
    print(f"\nProcessed {len(csv_files)} CSV files")


# =============================================================================
# Merge All CSV Files
# =============================================================================

def merge_all_csv_files(
    input_dir: str = "breach_csv_files_with_time",
    output_file: str = "merged_training_data.csv"
):
    """
    Merge all CSV files with breach source as first column.
    
    Args:
        input_dir: Directory containing CSV files to merge
        output_file: Output merged CSV file path
    """
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    all_dfs = []
    
    for csv_file in csv_files:
        file_path = os.path.join(input_dir, csv_file)
        
        # Read CSV
        df = pd.read_csv(file_path)
        
        # Add breach source column (from filename)
        breach_source = csv_file.replace('.csv', '')
        df.insert(0, 'breach_source', breach_source)
        
        all_dfs.append(df)
        print(f"Loaded {csv_file}: {len(df)} rows")
    
    # Concatenate all dataframes
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    # Reorder columns to have breach_source first
    columns = ['breach_source'] + [col for col in merged_df.columns if col != 'breach_source']
    merged_df = merged_df[columns]
    
    # Save merged data
    merged_df.to_csv(output_file, index=False, encoding="utf-8")
    
    print(f"\nMerged {len(csv_files)} files into {output_file}")
    print(f"Total rows: {len(merged_df)}")
    print(f"Columns: {list(merged_df.columns)}")
    
    return merged_df


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("Data Preparation Pipeline")
    print("=" * 80)
    
    # Step 1: Split by breach name
    # split_by_breach_name()
    
    # Step 2: Add breach times
    # add_breach_time_to_csvs()
    
    # Step 3: Merge all CSV files
    # merge_all_csv_files()
    
    print("\nNote: Uncomment and configure the steps above to run the pipeline")
