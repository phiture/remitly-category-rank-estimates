#!/usr/bin/env python3
"""
Compare Remitly's current downloads to category thresholds and calculate
how many more downloads are needed to reach each category range.
Supports both Android and iOS.
"""

import pandas as pd
import numpy as np
import os

# Country name mapping (from category files to country codes)
COUNTRY_MAPPING = {
    "United States": "us",
    "Mexico": "mx",
    "United Kingdom (UK)": "gb",
    "Germany": "de",
    "France": "fr",
    "Spain": "es",
    "Canada": "ca",
    "Austria": "at",
    "Finland": "fi",
    "Denmark": "dk",
    "Brazil": "br",
    "Italy": "it",
    "Australia": "au",
    "Romania": "ro",
    "United Arab Emirates": "ae",
    "Portugal": "pt",
    "Sweden": "se",
    "Ireland": "ie",
    "Norway": "no",
    "New Zealand": "nz"
}

def load_category_thresholds(platform="android", use_adjusted=True):
    """Load category threshold files for the specified platform and combine them"""
    # Map platform names to file suffixes
    platform_map = {
        "android": "android",
        "iphone": "ios",
        "ios": "ios"
    }
    file_suffix = platform_map.get(platform, platform)
    
    # Use adjusted files if available, otherwise use original
    if use_adjusted:
        file1 = f"market_downloads_category_1_{platform}_adjusted.csv"
        file2 = f"market_downloads_category_2_{platform}_adjusted.csv"
        import os
        if not (os.path.exists(file1) and os.path.exists(file2)):
            # Fall back to original files
            file1 = f"market_downloads_category_1_{file_suffix}.csv"
            file2 = f"market_downloads_category_2_{file_suffix}.csv"
    else:
        file1 = f"market_downloads_category_1_{file_suffix}.csv"
        file2 = f"market_downloads_category_2_{file_suffix}.csv"
    
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Combine the two dataframes
    combined = df1[["Top"]].copy()
    
    # Add all columns from df1 (except Top)
    for col in df1.columns:
        if col != "Top":
            combined[col] = df1[col]
    
    # Add all columns from df2 (except Top)
    for col in df2.columns:
        if col != "Top":
            combined[col] = df2[col]
    
    return combined

def get_remitly_current_downloads(platform="android", days_to_avg=90, use_actual=True):
    """
    Get Remitly's current downloads per market.
    If use_actual=True, uses actual console downloads, otherwise uses AppTweak estimates.
    
    Args:
        platform: 'android' or 'iphone'
        days_to_avg: Number of recent days to average (default 7)
        use_actual: If True, use actual console downloads; if False, use AppTweak estimates
    
    Returns:
        Dictionary mapping country_code -> average downloads
    """
    if use_actual:
        # Use actual console downloads
        platform_file_map = {
            "android": "android",
            "iphone": "ios",
            "ios": "ios"
        }
        file_suffix = platform_file_map.get(platform, platform)
        # Check in console_downloads_data directory first, then root
        file_path = f"console_downloads_data/remitly_actual_downloads_{file_suffix}.csv"
        if not os.path.exists(file_path):
            file_path = f"remitly_actual_downloads_{file_suffix}.csv"
        
        print(f"  Loading actual console downloads from {file_path}...")
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Get most recent N days
        most_recent_date = df['date'].max()
        print(f"  Most recent date: {most_recent_date.date()}")
        cutoff_date = most_recent_date - pd.Timedelta(days=days_to_avg)
        recent_df = df[df['date'] >= cutoff_date].copy()
        
        # Find downloads column
        downloads_col = None
        for col in ['total_downloads', 'search_explore_downloads', 'downloads']:
            if col in recent_df.columns:
                downloads_col = col
                break
        
        if downloads_col is None:
            print(f"  Warning: Could not find downloads column")
            return {}
        
        recent_df = recent_df[recent_df['country_code'].notna()].copy()
        recent_df = recent_df[recent_df[downloads_col] > 0]
        
        # Country code mapping
        COUNTRY_CODE_MAP = {
            'US': 'us', 'MX': 'mx', 'GB': 'gb', 'DE': 'de', 'FR': 'fr', 'ES': 'es',
            'CA': 'ca', 'AT': 'at', 'FI': 'fi', 'DK': 'dk', 'BR': 'br', 'IT': 'it',
            'AU': 'au', 'RO': 'ro', 'AE': 'ae', 'PT': 'pt', 'SE': 'se', 'IE': 'ie',
            'NO': 'no', 'NZ': 'nz'
        }
        
        remitly_downloads = {}
        for country_code_upper in recent_df['country_code'].unique():
            if pd.isna(country_code_upper):
                continue
            country_code_upper = str(country_code_upper).strip()
            country_code_lower = COUNTRY_CODE_MAP.get(country_code_upper, country_code_upper.lower())
            country_data = recent_df[recent_df['country_code'] == country_code_upper][downloads_col]
            if len(country_data) > 0:
                # Use mean (90-day window for stability)
                avg_downloads = country_data.mean()
                remitly_downloads[country_code_lower] = float(avg_downloads)
                print(f"    {country_code_lower}: {avg_downloads:,.0f} downloads (mean of {len(country_data)} data points)")
        
        return remitly_downloads
    else:
        # Use AppTweak estimates from app_rank_downloads.csv
        print(f"  Loading app_rank_downloads.csv...")
        df = pd.read_csv("app_rank_downloads.csv")
        
        # Filter for Remitly
        remitly_df = df[df['app_name'].str.contains('Remitly', case=False, na=False)].copy()
        
        # Filter for platform
        device_map = {"android": "android", "iphone": "iphone"}
        if platform not in device_map:
            raise ValueError(f"Platform must be 'android' or 'iphone', got '{platform}'")
        
        remitly_df = remitly_df[remitly_df['device'] == device_map[platform]].copy()
        
        # Convert date to datetime
        remitly_df['date'] = pd.to_datetime(remitly_df['date'])
        
        # Get most recent date
        most_recent_date = remitly_df['date'].max()
        print(f"  Most recent date: {most_recent_date.date()}")
        
        # Get data from last N days
        cutoff_date = most_recent_date - pd.Timedelta(days=days_to_avg)
        recent_df = remitly_df[recent_df['date'] >= cutoff_date].copy()
        
        # Filter out zero downloads (they might be missing data)
        recent_df = recent_df[recent_df['downloads'] > 0]
        
        # Calculate average downloads per country
        remitly_downloads = {}
        for country_code in recent_df['country'].unique():
            country_data = recent_df[recent_df['country'] == country_code]['downloads']
            if len(country_data) > 0:
                avg_downloads = country_data.mean()
                remitly_downloads[country_code] = float(avg_downloads)
                print(f"    {country_code}: {avg_downloads:,.0f} downloads (avg of {len(country_data)} data points)")
        
        return remitly_downloads

def calculate_downloads_needed(remitly_current, category_thresholds):
    """
    Calculate how many more downloads Remitly needs for each category in each market.
    """
    results = []
    
    # Process each market (country)
    for country_name, country_code in COUNTRY_MAPPING.items():
        if country_name not in category_thresholds.columns:
            continue
        
        current = remitly_current.get(country_code, 0)
        
        # Process each category (Top 1, Top 2, etc.)
        for _, row in category_thresholds.iterrows():
            category = row["Top"]
            threshold = row[country_name]
            
            if pd.isna(threshold):
                continue
            
            threshold = float(threshold)
            downloads_needed = max(0, threshold - current)
            
            results.append({
                "country": country_name,
                "country_code": country_code,
                "category": category,
                "threshold_downloads": threshold,
                "current_downloads": current,
                "downloads_needed": downloads_needed,
                "already_achieved": current >= threshold
            })
    
    return pd.DataFrame(results)

def run_comparison(platform="android"):
    """Run the comparison for a specific platform"""
    print("\n" + "=" * 80)
    print(f"Remitly Category Comparison Analysis - {platform.upper()}")
    print("=" * 80)
    
    # Load category thresholds (use adjusted if available)
    print(f"\nLoading {platform} category thresholds (using adjusted files if available)...")
    category_thresholds = load_category_thresholds(platform, use_adjusted=True)
    print(f"Loaded thresholds for {len(category_thresholds)} categories")
    print(f"Markets: {[col for col in category_thresholds.columns if col != 'Top']}")
    
    # Get Remitly's current downloads (use actual console downloads)
    print(f"\nGetting Remitly's current {platform} downloads from actual console data...")
    remitly_current = get_remitly_current_downloads(platform, use_actual=True)
    
    if not remitly_current:
        print(f"\n⚠️  WARNING: No Remitly {platform} download data found in app_rank_downloads.csv")
        return None
    
    print(f"\nFound Remitly {platform} downloads for {len(remitly_current)} market(s):")
    for country, downloads in sorted(remitly_current.items()):
        print(f"  {country}: {downloads:,.0f} downloads")
    
    # Calculate downloads needed
    print("\nCalculating downloads needed for each category...")
    results_df = calculate_downloads_needed(remitly_current, category_thresholds)
    
    # Save results
    output_file = f"remitly_category_comparison_{platform}_aligned.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Summary by country
    print("\nBy Country (showing categories already achieved and next targets):")
    for country_code in sorted(results_df['country_code'].unique()):
        country_data = results_df[results_df['country_code'] == country_code].copy()
        country_name = country_data['country'].iloc[0]
        current = country_data['current_downloads'].iloc[0]
        
        achieved = country_data[country_data['already_achieved']]
        not_achieved = country_data[~country_data['already_achieved']]
        
        print(f"\n{country_name} ({country_code}):")
        print(f"  Current downloads: {current:,.0f}")
        
        if len(achieved) > 0:
            best_category = achieved.sort_values('threshold_downloads', ascending=False).iloc[0]
            print(f"  ✅ Best category achieved: {best_category['category']} ({best_category['threshold_downloads']:,.0f} downloads)")
        
        if len(not_achieved) > 0:
            next_target = not_achieved.sort_values('threshold_downloads').iloc[0]
            print(f"  🎯 Next target: {next_target['category']} - need {next_target['downloads_needed']:,.0f} more downloads")
    
    # Summary by category
    print("\n\nBy Category (showing how many markets need more downloads):")
    for category in category_thresholds['Top']:
        cat_data = results_df[results_df['category'] == category]
        total_markets = len(cat_data)
        achieved_count = cat_data['already_achieved'].sum()
        need_more = total_markets - achieved_count
        
        if need_more > 0:
            avg_needed = cat_data[~cat_data['already_achieved']]['downloads_needed'].mean()
            print(f"  {category}: {achieved_count}/{total_markets} markets achieved, "
                  f"{need_more} need more (avg: {avg_needed:,.0f} downloads)")
        else:
            print(f"  {category}: {achieved_count}/{total_markets} markets achieved ✅")
    
    return results_df

def main():
    """Run comparisons for both Android and iOS"""
    print("=" * 80)
    print("Remitly Category Comparison Analysis")
    print("=" * 80)
    
    # Run for Android
    android_results = run_comparison("android")
    
    # Run for iOS
    ios_results = run_comparison("iphone")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nFiles created:")
    print("  - remitly_category_comparison_android.csv")
    print("  - remitly_category_comparison_iphone.csv")

if __name__ == "__main__":
    main()
