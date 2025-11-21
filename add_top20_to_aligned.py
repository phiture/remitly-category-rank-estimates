#!/usr/bin/env python3
"""
Add Top 20 estimates to the aligned category comparison files by interpolating
between Top 10 and Top 50 using logarithmic interpolation.
"""

import pandas as pd
import numpy as np

def estimate_top20(top10, top50):
    """
    Estimate Top 20 downloads by logarithmic interpolation between Top 10 and Top 50.
    Since rank follows a logarithmic relationship with downloads, we interpolate in log space.
    Top 20 is 1/4 of the way from Top 10 to Top 50 in rank space: (20-10)/(50-10) = 0.25
    """
    if pd.isna(top10) or pd.isna(top50) or top10 <= 0 or top50 <= 0:
        return np.nan
    
    # Convert to log space
    log_top10 = np.log(top10)
    log_top50 = np.log(top50)
    
    # Interpolate: Top 20 is 1/4 of the way from Top 10 to Top 50
    log_top20 = log_top10 + 0.25 * (log_top50 - log_top10)
    
    # Convert back
    top20 = np.exp(log_top20)
    return top20

def add_top20_to_aligned_comparison(platform="android"):
    """Add Top 20 estimates to the aligned comparison file"""
    input_file = f"remitly_category_comparison_{platform}_aligned.csv"
    output_file = f"remitly_category_comparison_{platform}_aligned_with_top20.csv"
    
    print(f"\nProcessing {platform} aligned file...")
    df = pd.read_csv(input_file)
    
    # Get Top 10 and Top 50 data for each market
    top10_data = df[df['category'] == 'Top 10'].set_index('country_code')
    top50_data = df[df['category'] == 'Top 50'].set_index('country_code')
    
    # Create Top 20 rows
    top20_rows = []
    for country_code in df['country_code'].unique():
        country_name = df[df['country_code'] == country_code]['country'].iloc[0]
        current_downloads = df[df['country_code'] == country_code]['current_downloads'].iloc[0]
        
        if country_code in top10_data.index and country_code in top50_data.index:
            top10_threshold = top10_data.loc[country_code, 'threshold_downloads']
            top50_threshold = top50_data.loc[country_code, 'threshold_downloads']
            
            top20_threshold = estimate_top20(top10_threshold, top50_threshold)
            
            if not pd.isna(top20_threshold):
                downloads_needed = max(0, top20_threshold - current_downloads)
                already_achieved = current_downloads >= top20_threshold
                
                top20_rows.append({
                    'country': country_name,
                    'country_code': country_code,
                    'category': 'Top 20',
                    'threshold_downloads': top20_threshold,
                    'current_downloads': current_downloads,
                    'downloads_needed': downloads_needed,
                    'already_achieved': already_achieved
                })
    
    # Add Top 20 rows to dataframe
    top20_df = pd.DataFrame(top20_rows)
    
    # Insert Top 20 rows after Top 10 and before Top 50
    # Reorder: Top 1, Top 2, Top 3, Top 5, Top 10, Top 20, Top 50, Top 100
    category_order = ['Top 1', 'Top 2', 'Top 3', 'Top 5', 'Top 10', 'Top 20', 'Top 50', 'Top 100']
    
    # Combine and sort
    combined_df = pd.concat([df, top20_df], ignore_index=True)
    combined_df['category'] = pd.Categorical(combined_df['category'], categories=category_order, ordered=True)
    combined_df = combined_df.sort_values(['country_code', 'category'])
    
    # Save
    combined_df.to_csv(output_file, index=False)
    print(f"  ✅ Saved: {output_file}")
    
    # Print summary
    print(f"\n  Top 20 Summary for {platform} (aligned):")
    achieved = top20_df[top20_df['already_achieved']]
    if len(achieved) > 0:
        print(f"    Markets that achieved Top 20: {len(achieved)}")
        for _, row in achieved.iterrows():
            print(f"      {row['country']} ({row['country_code']}): {row['current_downloads']:,.0f} downloads (threshold: {row['threshold_downloads']:,.0f})")
    else:
        print(f"    No markets have achieved Top 20 yet")
    
    not_achieved = top20_df[~top20_df['already_achieved']]
    if len(not_achieved) > 0:
        print(f"\n    Markets closest to Top 20:")
        closest = not_achieved.nsmallest(5, 'downloads_needed')
        for _, row in closest.iterrows():
            print(f"      {row['country']} ({row['country_code']}): need {row['downloads_needed']:,.0f} more (current: {row['current_downloads']:,.0f}, threshold: {row['threshold_downloads']:,.0f})")
    
    return combined_df

if __name__ == "__main__":
    print("=" * 80)
    print("Adding Top 20 Estimates to Aligned Files")
    print("=" * 80)
    
    android_df = add_top20_to_aligned_comparison("android")
    ios_df = add_top20_to_aligned_comparison("iphone")
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print("\nFiles created:")
    print("  - remitly_category_comparison_android_aligned_with_top20.csv")
    print("  - remitly_category_comparison_iphone_aligned_with_top20.csv")

