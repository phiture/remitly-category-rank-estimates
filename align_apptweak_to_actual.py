#!/usr/bin/env python3
"""
Compare AppTweak estimates vs actual console downloads for Remitly,
calculate adjustment factors, and apply them to category thresholds.
"""

import pandas as pd
import numpy as np
import os

# Country code mapping (uppercase to lowercase)
COUNTRY_CODE_MAP = {
    'US': 'us', 'MX': 'mx', 'GB': 'gb', 'DE': 'de', 'FR': 'fr', 'ES': 'es',
    'CA': 'ca', 'AT': 'at', 'FI': 'fi', 'DK': 'dk', 'BR': 'br', 'IT': 'it',
    'AU': 'au', 'RO': 'ro', 'AE': 'ae', 'PT': 'pt', 'SE': 'se', 'IE': 'ie',
    'NO': 'no', 'NZ': 'nz'
}

def get_apptweak_downloads(platform="android", days=90, use_world_file=True):
    """
    Get AppTweak download estimates.
    If use_world_file=True, uses the new world-download-estimates files (October totals / 31).
    Otherwise, uses app_rank_downloads.csv.
    """
    if use_world_file:
        # Use the new world download estimates files
        # Check in apptweak_data directory first, then root
        file_paths_to_try = []
        if platform == "android":
            # Android has a space in the filename
            file_paths_to_try = [
                "apptweak_data/world-download-estimates _android.csv",
                "world-download-estimates _android.csv",
                "apptweak_data/world-download-estimates_android.csv",
                "world-download-estimates_android.csv"
            ]
        else:
            # iOS - handle both "iphone" and "ios" platform names
            ios_platform = "ios" if platform in ["iphone", "ios"] else platform
            file_paths_to_try = [
                f"apptweak_data/world-download-estimates_{ios_platform}.csv",
                f"world-download-estimates_{ios_platform}.csv"
            ]
        
        file_path = None
        for path in file_paths_to_try:
            if os.path.exists(path):
                file_path = path
                break
        
        if file_path and os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            # Country name to code mapping
            COUNTRY_NAME_TO_CODE = {
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
            
            apptweak_downloads = {}
            for _, row in df.iterrows():
                country_name = row['Country']
                total_90days = row['Worldwide download estimates']
                
                # Convert to daily average (90-day period)
                daily_avg = total_90days / 90.0
                
                # Map country name to code
                country_code = COUNTRY_NAME_TO_CODE.get(country_name)
                if country_code:
                    apptweak_downloads[country_code] = float(daily_avg)
            
            return apptweak_downloads
        else:
            # File not found, print error and return empty dict
            print(f"   ⚠️  Warning: Could not find world-download-estimates file for {platform}")
            print(f"   Tried paths: {file_paths_to_try}")
            return {}
    
    # Fall back to app_rank_downloads.csv
    df = pd.read_csv("app_rank_downloads.csv")
    
    # Filter for Remitly
    remitly_df = df[df['app_name'].str.contains('Remitly', case=False, na=False)].copy()
    
    # Filter for platform
    device_map = {"android": "android", "iphone": "iphone"}
    remitly_df = remitly_df[remitly_df['device'] == device_map[platform]].copy()
    
    # Convert date
    remitly_df['date'] = pd.to_datetime(remitly_df['date'])
    
    # Get most recent N days
    most_recent_date = remitly_df['date'].max()
    cutoff_date = most_recent_date - pd.Timedelta(days=days)
    recent_df = remitly_df[remitly_df['date'] >= cutoff_date].copy()
    recent_df = recent_df[recent_df['downloads'] > 0]
    
    # Average by country
    apptweak_downloads = {}
    for country_code in recent_df['country'].unique():
        country_data = recent_df[recent_df['country'] == country_code]['downloads']
        if len(country_data) > 0:
            apptweak_downloads[country_code.lower()] = float(country_data.mean())
    
    return apptweak_downloads

def get_actual_downloads(platform="android", days=90, use_median=False, remove_outliers=True):
    """
    Get actual console downloads.
    Uses mean by default (90-day window for stability).
    Uses last 90 days by default for more stable estimates.
    
    Args:
        platform: 'android' or 'iphone'
        days: Number of recent days to include (default 90)
        use_median: If True, use median instead of mean
        remove_outliers: If True, remove outliers using IQR method before calculating statistics
    """
    # Map platform names to file names
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
    df = pd.read_csv(file_path)
    
    # Convert date
    df['date'] = pd.to_datetime(df['date'])
    
    # Get most recent N days
    most_recent_date = df['date'].max()
    cutoff_date = most_recent_date - pd.Timedelta(days=days)
    recent_df = df[df['date'] >= cutoff_date].copy()
    
    # Find the downloads column (could be 'total_downloads' or 'search_explore_downloads' or similar)
    downloads_col = None
    for col in ['total_downloads', 'search_explore_downloads', 'downloads']:
        if col in recent_df.columns:
            downloads_col = col
            break
    
    if downloads_col is None:
        print(f"   Warning: Could not find downloads column in {file_path}")
        return {}
    
    # Filter out NaN country codes and zero downloads
    recent_df = recent_df[recent_df['country_code'].notna()].copy()
    recent_df = recent_df[recent_df[downloads_col] > 0].copy()
    
    # Convert country codes to lowercase and use median (or mean)
    actual_downloads = {}
    for country_code_upper in recent_df['country_code'].unique():
        if pd.isna(country_code_upper):
            continue
        country_code_upper = str(country_code_upper).strip()
        country_code_lower = COUNTRY_CODE_MAP.get(country_code_upper, country_code_upper.lower())
        country_data = recent_df[recent_df['country_code'] == country_code_upper][downloads_col]
        
        if len(country_data) > 0:
            # Remove outliers if requested
            if remove_outliers and len(country_data) > 3:  # Need at least 4 points for IQR
                original_count = len(country_data)
                q1 = country_data.quantile(0.25)
                q3 = country_data.quantile(0.75)
                iqr = q3 - q1
                
                # Define outlier bounds (using 1.5 * IQR rule)
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Filter out outliers
                country_data_cleaned = country_data[
                    (country_data >= lower_bound) & (country_data <= upper_bound)
                ]
                
                # Only use cleaned data if we still have enough points
                if len(country_data_cleaned) >= max(3, original_count * 0.5):  # Keep at least 50% of data
                    outliers_removed = original_count - len(country_data_cleaned)
                    country_data = country_data_cleaned
                    if outliers_removed > 0:
                        print(f"   {country_code_lower}: Removed {outliers_removed} outlier(s) (out of {original_count} total)")
            
            if use_median:
                actual_downloads[country_code_lower] = float(country_data.median())
            else:
                actual_downloads[country_code_lower] = float(country_data.mean())
    
    return actual_downloads

def calculate_adjustment_factors(apptweak_downloads, actual_downloads):
    """Calculate adjustment factors (ratio of actual to apptweak)"""
    factors = {}
    comparison = []
    
    for country in set(list(apptweak_downloads.keys()) + list(actual_downloads.keys())):
        apptweak = apptweak_downloads.get(country, None)
        actual = actual_downloads.get(country, None)
        
        if apptweak is not None and actual is not None and apptweak > 0:
            factor = actual / apptweak
            factors[country] = factor
            comparison.append({
                'country_code': country,
                'apptweak_downloads': apptweak,
                'actual_downloads': actual,
                'adjustment_factor': factor,
                'difference': actual - apptweak,
                'difference_pct': ((actual - apptweak) / apptweak) * 100
            })
    
    return factors, pd.DataFrame(comparison)

def apply_adjustment_to_categories(platform="android", adjustment_factors=None):
    """Apply adjustment factors to category thresholds"""
    # Load category thresholds
    platform_map = {"android": "android", "iphone": "ios", "ios": "ios"}
    file_suffix = platform_map.get(platform, platform)
    
    # Check in apptweak_data directory first, then root
    file1 = f"apptweak_data/market_downloads_category_1_{file_suffix}.csv"
    file2 = f"apptweak_data/market_downloads_category_2_{file_suffix}.csv"
    if not os.path.exists(file1):
        file1 = f"market_downloads_category_1_{file_suffix}.csv"
    if not os.path.exists(file2):
        file2 = f"market_downloads_category_2_{file_suffix}.csv"
    
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Combine
    combined = df1[["Top"]].copy()
    for col in df1.columns:
        if col != "Top":
            combined[col] = df1[col]
    for col in df2.columns:
        if col != "Top":
            combined[col] = df2[col]
    
    # Country name to code mapping
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
    
    # Apply adjustment factors
    adjusted_df = combined.copy()
    
    if adjustment_factors:
        # Calculate average factor for markets without data
        available_factors = [f for f in adjustment_factors.values() if f > 0]
        avg_factor = np.mean(available_factors) if available_factors else 1.0
        
        for country_name, country_code in COUNTRY_MAPPING.items():
            if country_name in adjusted_df.columns:
                factor = adjustment_factors.get(country_code, avg_factor)
                adjusted_df[country_name] = adjusted_df[country_name] * factor
                print(f"  Adjusted {country_name} ({country_code}): factor = {factor:.3f}")
    
    return adjusted_df

def main():
    print("=" * 80)
    print("Aligning AppTweak Estimates to Actual Console Downloads")
    print("=" * 80)
    
    for platform in ["android", "iphone"]:
        print(f"\n{'='*80}")
        print(f"Processing {platform.upper()}")
        print(f"{'='*80}")
        
        # Get downloads
        print(f"\n1. Getting AppTweak downloads from world-download-estimates files...")
        apptweak_downloads = get_apptweak_downloads(platform, use_world_file=True)
        print(f"   Found {len(apptweak_downloads)} markets")
        if apptweak_downloads:
            print(f"   Sample: {dict(list(apptweak_downloads.items())[:5])}")
        
        print(f"\n2. Getting actual console downloads (90-day mean, outliers removed)...")
        actual_downloads = get_actual_downloads(platform, use_median=False, remove_outliers=True)
        print(f"   Found {len(actual_downloads)} markets")
        
        # Calculate adjustment factors
        print(f"\n3. Calculating adjustment factors...")
        adjustment_factors, comparison_df = calculate_adjustment_factors(
            apptweak_downloads, actual_downloads
        )
        
        if len(comparison_df) > 0:
            print(f"\n   Comparison (AppTweak vs Actual):")
            print(comparison_df.sort_values('country_code').to_string(index=False))
            
            print(f"\n   Adjustment factors:")
            for country, factor in sorted(adjustment_factors.items()):
                print(f"     {country}: {factor:.3f}x")
            
            # Save comparison
            comparison_file = f"apptweak_vs_actual_comparison_{platform}.csv"
            comparison_df.to_csv(comparison_file, index=False)
            print(f"\n   ✅ Saved comparison to: {comparison_file}")
            
            # Apply to category thresholds
            print(f"\n4. Applying adjustment factors to category thresholds...")
            adjusted_categories = apply_adjustment_to_categories(platform, adjustment_factors)
            
            # Save adjusted categories
            output_file1 = f"market_downloads_category_1_{platform}_adjusted.csv"
            output_file2 = f"market_downloads_category_2_{platform}_adjusted.csv"
            
            # Split back into two files
            platform_map = {"android": "android", "iphone": "ios", "ios": "ios"}
            file_suffix = platform_map.get(platform, platform)
            
            # Check in apptweak_data directory first, then root
            orig_file1 = f"apptweak_data/market_downloads_category_1_{file_suffix}.csv"
            orig_file2 = f"apptweak_data/market_downloads_category_2_{file_suffix}.csv"
            if not os.path.exists(orig_file1):
                orig_file1 = f"market_downloads_category_1_{file_suffix}.csv"
            if not os.path.exists(orig_file2):
                orig_file2 = f"market_downloads_category_2_{file_suffix}.csv"
            
            original_df1 = pd.read_csv(orig_file1)
            original_df2 = pd.read_csv(orig_file2)
            
            df1_cols = ["Top"] + [col for col in adjusted_categories.columns 
                                 if col in original_df1.columns]
            df2_cols = ["Top"] + [col for col in adjusted_categories.columns 
                                 if col in original_df2.columns]
            
            adjusted_categories[df1_cols].to_csv(output_file1, index=False)
            adjusted_categories[df2_cols].to_csv(output_file2, index=False)
            
            print(f"   ✅ Saved adjusted categories:")
            print(f"      - {output_file1}")
            print(f"      - {output_file2}")
        else:
            print(f"\n   ⚠️  No overlapping markets found between AppTweak and actual data")
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

