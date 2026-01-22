#!/usr/bin/env python3
"""
Lab 3 - Task B: Create stacked bar chart visualization
Creates a stacked bar chart from summer_olympics.csv showing:
- Years: 1992-2012 (inclusive)
- Countries: France, Germany, United Kingdom, United States, China
- Sports: Aquatics, Athletics, Football, Gymnastics, Rowing
- Stacked by medal type (Bronze, Silver, Gold)
- Two bars per country (Men, Women)
- Saves as barchart.png
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Filter criteria
TARGET_YEARS = list(range(1992, 2013))  # 1992-2012 inclusive
TARGET_COUNTRIES = ['France', 'Germany', 'United Kingdom', 'United States', 'China']
TARGET_SPORTS = ['Aquatics', 'Athletics', 'Football', 'Gymnastics', 'Rowing']
MEDAL_ORDER = ['Gold', 'Silver', 'Bronze']  # Stacking order (top to bottom)

def read_olympics_data(filename):
    """Read the Olympics CSV and filter data according to requirements"""
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            year = int(row['Year'])
            country = row['Country']
            sport = row['Sport']
            
            # Apply filters
            if year in TARGET_YEARS and country in TARGET_COUNTRIES and sport in TARGET_SPORTS:
                data.append({
                    'Year': year,
                    'Country': country,
                    'Sport': sport,
                    'Gender': row['Gender'],
                    'Medal': row['Medal']
                })
    return data

def count_medals(data):
    """Count medals by Country, Gender, and Medal type"""
    counts = defaultdict(int)
    
    for row in data:
        key = (row['Country'], row['Gender'], row['Medal'])
        counts[key] += 1
    
    return counts

def prepare_chart_data(counts):
    """Prepare data for stacked bar chart"""
    # Organize data: {Country: {Gender: {Medal: count}}}
    organized = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    for (country, gender, medal), count in counts.items():
        organized[country][gender][medal] = count
    
    # Prepare arrays for plotting
    countries = TARGET_COUNTRIES  # Maintain order
    genders = ['Men', 'Women']
    
    # Initialize data arrays (countries x genders x medals)
    gold_data = []
    silver_data = []
    bronze_data = []
    
    x_positions = []
    x_labels = []
    
    x_pos = 0
    for country in countries:
        for gender in genders:
            gold_count = organized[country][gender]['Gold']
            silver_count = organized[country][gender]['Silver']
            bronze_count = organized[country][gender]['Bronze']
            
            gold_data.append(gold_count)
            silver_data.append(silver_count)
            bronze_data.append(bronze_count)
            
            x_positions.append(x_pos)
            x_labels.append(f"{country}\n{gender}")
            x_pos += 1
        
        # Add spacing between countries
        x_pos += 0.5
    
    return gold_data, silver_data, bronze_data, x_positions, x_labels, countries

def create_stacked_bar_chart(counts, output_filename):
    """Create stacked bar chart and save as PNG"""
    gold_data, silver_data, bronze_data, x_positions, x_labels, countries = prepare_chart_data(counts)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Convert to numpy arrays
    gold_array = np.array(gold_data)
    silver_array = np.array(silver_data)
    bronze_array = np.array(bronze_data)
    
    # Create stacked bars
    width = 0.7
    bottom_gold_silver = bronze_array + silver_array
    
    bars_bronze = ax.bar(x_positions, bronze_array, width, label='Bronze', color='#CD7F32')
    bars_silver = ax.bar(x_positions, silver_array, width, bottom=bronze_array, label='Silver', color='#C0C0C0')
    bars_gold = ax.bar(x_positions, gold_array, width, bottom=bottom_gold_silver, label='Gold', color='#FFD700')
    
    # Set x-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=9)
    
    # Add labels and title
    ax.set_xlabel('Country and Gender', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count of Medals', fontsize=12, fontweight='bold')
    ax.set_title('Olympic Medals by Country and Gender (1992-2012)\nSports: Aquatics, Athletics, Football, Gymnastics, Rowing', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    ax.legend(title='Medal Type', title_fontsize=10, fontsize=10, loc='upper right')
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add vertical lines to separate countries
    country_separators = []
    for i in range(1, len(countries)):
        # Each country has 2 bars (Men, Women) + 0.5 spacing
        separator_pos = i * 2.5 - 0.25
        country_separators.append(separator_pos)
        ax.axvline(x=separator_pos, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Chart saved as {output_filename}")

def main():
    """Main function"""
    input_file = 'Lab3_summer_olympics.csv'
    output_file = 'barchart.png'
    
    print(f"Reading data from {input_file}...")
    data = read_olympics_data(input_file)
    print(f"Loaded {len(data)} rows after filtering")
    
    print("Counting medals...")
    counts = count_medals(data)
    print(f"Found {len(counts)} unique combinations")
    
    print("Creating stacked bar chart...")
    create_stacked_bar_chart(counts, output_file)
    print("Done!")

if __name__ == "__main__":
    main()
