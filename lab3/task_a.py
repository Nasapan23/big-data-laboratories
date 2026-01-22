#!/usr/bin/env python3
"""
Lab 3 - Task A: Create table visualization
Creates a table from worldchamps_mens_gymnastics.csv with columns:
Name, Overall Rank, Nationality, Apparatus, Total Score
Saves as table.png
"""

import csv
import matplotlib.pyplot as plt
from matplotlib.table import Table

def read_gymnastics_data(filename):
    """Read the gymnastics CSV and extract required columns"""
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'Name': row['Name'],
                'Overall Rank': row['Overall Rank'],
                'Nationality': row['Nationality'],
                'Apparatus': row['Apparatus'],
                'Total Score': row['Total Score']
            })
    return data

def create_table(data, output_filename):
    """Create a table visualization and save as PNG"""
    # Prepare table data
    table_data = []
    headers = ['Name', 'Overall Rank', 'Nationality', 'Apparatus', 'Total Score']
    
    # Add headers
    table_data.append(headers)
    
    # Add data rows
    for row in data:
        table_data.append([
            row['Name'],
            row['Overall Rank'],
            row['Nationality'],
            row['Apparatus'],
            row['Total Score']
        ])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, max(8, len(data) * 0.15)))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=table_data, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    
    # Style header row
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_height(0.05)
    
    # Style data rows (alternating colors)
    for i in range(1, len(table_data)):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#D9E1F2')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
            table[(i, j)].set_height(0.03)
    
    plt.title('Men\'s Gymnastics World Championships', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Table saved as {output_filename}")

def main():
    """Main function"""
    input_file = 'Lab3_worldchamps_mens_gymnastics.csv'
    output_file = 'table.png'
    
    print(f"Reading data from {input_file}...")
    data = read_gymnastics_data(input_file)
    print(f"Loaded {len(data)} rows")
    
    print(f"Creating table visualization...")
    create_table(data, output_file)
    print("Done!")

if __name__ == "__main__":
    main()
