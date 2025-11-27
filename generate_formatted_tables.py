#!/usr/bin/env python3
"""
Generate formatted tables showing Current Downloads and Target Downloads
for Top 50, Top 20, Top 10, and Top 5 categories.
"""

import pandas as pd

def generate_table(platform="android"):
    """Generate formatted table for a platform"""
    file_path = f"remitly_category_comparison_{platform}_aligned_with_top20.csv"
    df = pd.read_csv(file_path)
    
    # Filter for the categories we need
    categories = ['Top 50', 'Top 20', 'Top 10', 'Top 5']
    filtered_df = df[df['category'].isin(categories)].copy()
    
    # Create a pivot table
    pivot = filtered_df.pivot_table(
        index=['country', 'country_code'],
        columns='category',
        values='threshold_downloads',
        aggfunc='first'
    ).reset_index()
    
    # Get current downloads (should be same for all categories for a country)
    current_downloads = {}
    for country_code in df['country_code'].unique():
        country_data = df[df['country_code'] == country_code]
        if len(country_data) > 0:
            current_downloads[country_code] = country_data.iloc[0]['current_downloads']
    
    # Add current downloads column
    pivot['current_downloads'] = pivot['country_code'].map(current_downloads)
    
    # Reorder columns
    pivot = pivot[['country', 'current_downloads', 'Top 50', 'Top 20', 'Top 10', 'Top 5']]
    
    # Round to integers
    pivot['current_downloads'] = pivot['current_downloads'].round().astype(int)
    pivot['Top 50'] = pivot['Top 50'].round().astype(int)
    pivot['Top 20'] = pivot['Top 20'].round().astype(int)
    pivot['Top 10'] = pivot['Top 10'].round().astype(int)
    pivot['Top 5'] = pivot['Top 5'].round().astype(int)
    
    # Sort by country name
    pivot = pivot.sort_values('country').reset_index(drop=True)
    
    # Rename columns for display
    pivot.columns = ['Country', 'Current Downloads', 'Top 50 Target', 'Top 20 Target', 'Top 10 Target', 'Top 5 Target']
    
    return pivot

def format_table_for_display(df):
    """Format table as a markdown/plain text table"""
    lines = []
    
    # Header
    header = "\t".join(df.columns)
    lines.append(header)
    lines.append("")
    
    # Rows
    for _, row in df.iterrows():
        row_str = "\t".join([str(val) for val in row.values])
        lines.append(row_str)
    
    return "\n".join(lines)

def main():
    print("=" * 80)
    print("Generating Formatted Tables")
    print("=" * 80)
    
    for platform in ["android", "iphone"]:
        print(f"\n{platform.upper()}:")
        print("-" * 80)
        
        df = generate_table(platform)
        
        # Print formatted table
        print(format_table_for_display(df))
        
        # Save to file
        output_file = f"formatted_table_{platform}.txt"
        with open(output_file, 'w') as f:
            f.write(format_table_for_display(df))
        print(f"\n✅ Saved to: {output_file}")
        
        # Also save as CSV
        csv_file = f"formatted_table_{platform}.csv"
        df.to_csv(csv_file, index=False)
        print(f"✅ Saved CSV to: {csv_file}")

if __name__ == "__main__":
    main()

