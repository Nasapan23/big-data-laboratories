"""
Data Loading Module
Step 1: Load CSV, filter rows, create combined text field
"""

import pandas as pd


def load_data(csv_path):
    """
    Load CSV data and prepare it for analysis.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        DataFrame with columns: Unique Key, Complaint Type, Descriptor, Created Date, combined_text
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Map column names (the CSV uses different names)
    # 'Problem (formerly Complaint Type)' -> 'Complaint Type'
    # 'Problem Detail (formerly Descriptor)' -> 'Descriptor'
    column_mapping = {
        'Problem (formerly Complaint Type)': 'Complaint Type',
        'Problem Detail (formerly Descriptor)': 'Descriptor'
    }
    df = df.rename(columns=column_mapping)
    
    # Keep needed columns (required + location columns)
    required_columns = ['Unique Key', 'Complaint Type', 'Descriptor', 'Created Date']
    location_columns = ['Incident Address', 'Street Name', 'Incident Zip', 'Borough', 'City']
    
    # Check which columns exist
    required_available = [col for col in required_columns if col in df.columns]
    location_available = [col for col in location_columns if col in df.columns]
    available_columns = required_available + location_available
    
    df = df[available_columns]
    
    # Drop rows where Descriptor is missing or empty
    df = df.dropna(subset=['Descriptor'])
    df = df[df['Descriptor'].str.strip() != '']
    
    # Create combined text field: Complaint Type + " - " + Descriptor
    df['combined_text'] = df['Complaint Type'].astype(str) + " - " + df['Descriptor'].astype(str)
    
    # Reset index
    df = df.reset_index(drop=True)
    
    print(f"Loaded {len(df)} records after filtering")
    
    return df
