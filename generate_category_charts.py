#!/usr/bin/env python3
"""
Generate bar charts showing downloads needed for Top 5, Top 10, and Top 20,
and current vs target downloads for each category.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

def load_comparison_data(platform="android"):
    """Load the aligned comparison data"""
    file_path = f"remitly_category_comparison_{platform}_aligned_with_top20.csv"
    df = pd.read_csv(file_path)
    return df

def create_bar_chart(df, platform="android", output_dir="charts", country_group=None):
    """
    Create a bar chart showing downloads needed for Top 5, Top 10, Top 20.
    If country_group is provided, filter to only show those countries.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Filter for Top 5, Top 10, and Top 20
    categories = ['Top 5', 'Top 10', 'Top 20']
    filtered_df = df[df['category'].isin(categories)].copy()
    
    # Filter by country group if specified
    if country_group is not None:
        filtered_df = filtered_df[filtered_df['country'].isin(country_group)].copy()
    
    # Create a pivot table: country_code x category -> downloads_needed
    pivot_df = filtered_df.pivot_table(
        index=['country_code', 'country'],
        columns='category',
        values='downloads_needed',
        aggfunc='first'
    ).reset_index()
    
    # Get current downloads (should be the same for all categories for a country)
    current_downloads_dict = {}
    for country in pivot_df['country'].values:
        country_data = filtered_df[filtered_df['country'] == country]
        if len(country_data) > 0:
            current_downloads_dict[country] = country_data.iloc[0]['current_downloads']
    
    # Sort by country name for consistent ordering
    pivot_df = pivot_df.sort_values('country').reset_index(drop=True)
    
    # Determine group suffix for filename
    group1_countries = ["United States", "United Kingdom (UK)", "Germany", "Spain", 
                       "France", "Australia", "Brazil", "Canada"]
    if country_group == group1_countries:
        group_suffix = "_group1"
    elif country_group is not None:
        group_suffix = "_group2"
    else:
        group_suffix = ""
    
    # Prepare data
    countries = pivot_df['country'].values
    top5 = pivot_df['Top 5'].fillna(0).values
    top10 = pivot_df['Top 10'].fillna(0).values
    top20 = pivot_df['Top 20'].fillna(0).values
    current = np.array([current_downloads_dict.get(country, 0) for country in countries])
    
    # Set up the plot
    x = np.arange(len(countries))
    width = 0.2  # Slightly narrower to fit 4 bars
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Create bars - use max(0, value) to ensure no negative heights
    # Current downloads first (as a reference)
    bars_current = ax.bar(x - width * 1.5, np.maximum(current, 0), width, 
                         label='Current Downloads', color='#9E9E9E', alpha=0.6, 
                         edgecolor='#616161', linewidth=1.5, linestyle='--')
    bars1 = ax.bar(x - width * 0.5, np.maximum(top20, 0), width, label='Top 20', color='#4CAF50', alpha=0.8)
    bars2 = ax.bar(x + width * 0.5, np.maximum(top10, 0), width, label='Top 10', color='#2196F3', alpha=0.8)
    bars3 = ax.bar(x + width * 1.5, np.maximum(top5, 0), width, label='Top 5', color='#FF9800', alpha=0.8)
    
    # Customize axes first
    ax.set_xlabel('Market', fontsize=12, fontweight='bold')
    ax.set_ylabel('Downloads', fontsize=12, fontweight='bold')
    ax.set_title(f'Current Downloads vs Downloads Needed for Category Milestones - {platform.upper()}', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(countries, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Use log scale if there's a large range (but only if we have positive values)
    # Also ensure y-axis starts at 0 to show all bars
    positive_values = np.concatenate([top5[top5 > 0], top10[top10 > 0], top20[top20 > 0], current[current > 0]])
    if len(positive_values) > 0:
        max_val = max(np.max(top5[top5 > 0]) if np.any(top5 > 0) else 0,
                     np.max(top10[top10 > 0]) if np.any(top10 > 0) else 0,
                     np.max(top20[top20 > 0]) if np.any(top20 > 0) else 0)
        min_val = np.min(positive_values)
        if max_val / min_val > 100:
            ax.set_yscale('log')
            ax.set_ylabel('Downloads (Log Scale)', fontsize=12, fontweight='bold')
            # For log scale, set a small minimum to show all bars
            ax.set_ylim(bottom=max(0.1, min_val * 0.1))
        else:
            # Set y-axis to start at 0 to ensure all bars are visible
            ax.set_ylim(bottom=0, top=max_val * 1.1)
    else:
        # If all are achieved, set a default range
        ax.set_ylim(bottom=0, top=100)
    
    # Now add "ACHIEVED" labels after y-axis is set
    y_max = ax.get_ylim()[1]
    label_y = y_max * 0.01  # Just above x-axis (1% of y-range)
    
    for i, country in enumerate(countries):
        if top20[i] <= 0:
            ax.text(x[i] - width * 0.5, label_y, 
                   'ACHIEVED', fontsize=8, ha='center', va='bottom', 
                   color='#4CAF50', fontweight='bold', rotation=0)
        if top10[i] <= 0:
            ax.text(x[i] + width * 0.5, label_y, 
                   'ACHIEVED', fontsize=8, ha='center', va='bottom', 
                   color='#2196F3', fontweight='bold', rotation=0)
        if top5[i] <= 0:
            ax.text(x[i] + width * 1.5, label_y, 
                   'ACHIEVED', fontsize=8, ha='center', va='bottom', 
                   color='#FF9800', fontweight='bold', rotation=0)
    
    # Add value labels on bars (only if not too crowded)
    if len(countries) <= 20:
        max_display = max(np.max(top5[top5 > 0]) if np.any(top5 > 0) else 0,
                         np.max(top10[top10 > 0]) if np.any(top10 > 0) else 0,
                         np.max(top20[top20 > 0]) if np.any(top20 > 0) else 0,
                         np.max(current[current > 0]) if np.any(current > 0) else 0)
        
        for bars in [bars_current, bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    if height < max_display * 0.9:  # Don't label if too close to top
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height):,}',
                               ha='center', va='bottom', fontsize=9, rotation=0, alpha=0.8)
    
    plt.tight_layout()
    
    # Save
    if group_suffix:
        output_path = f"{output_dir}/downloads_needed_bar_chart_{platform}{group_suffix}.png"
    else:
        output_path = f"{output_dir}/downloads_needed_bar_chart_{platform}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {output_path}")
    plt.close()

def create_total_downloads_chart(df, platform="android", output_dir="charts", country_group=None):
    """
    Create a bar chart showing total downloads needed (current + downloads needed) for Top 5, Top 10, Top 20.
    If country_group is provided, filter to only show those countries.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Filter for Top 5, Top 10, and Top 20
    categories = ['Top 5', 'Top 10', 'Top 20']
    filtered_df = df[df['category'].isin(categories)].copy()
    
    # Filter by country group if specified
    if country_group is not None:
        filtered_df = filtered_df[filtered_df['country'].isin(country_group)].copy()
    
    # Create a pivot table: country_code x category -> downloads_needed
    pivot_df = filtered_df.pivot_table(
        index=['country_code', 'country'],
        columns='category',
        values='downloads_needed',
        aggfunc='first'
    ).reset_index()
    
    # Get current downloads (should be the same for all categories for a country)
    current_downloads_dict = {}
    for country in pivot_df['country'].values:
        country_data = filtered_df[filtered_df['country'] == country]
        if len(country_data) > 0:
            current_downloads_dict[country] = country_data.iloc[0]['current_downloads']
    
    # Sort by country name for consistent ordering
    pivot_df = pivot_df.sort_values('country').reset_index(drop=True)
    
    # Determine group suffix for filename
    group1_countries = ["United States", "United Kingdom (UK)", "Germany", "Spain", 
                       "France", "Australia", "Brazil", "Canada"]
    if country_group == group1_countries:
        group_suffix = "_group1"
    elif country_group is not None:
        group_suffix = "_group2"
    else:
        group_suffix = ""
    
    # Prepare data
    countries = pivot_df['country'].values
    top5_needed = pivot_df['Top 5'].fillna(0).values
    top10_needed = pivot_df['Top 10'].fillna(0).values
    top20_needed = pivot_df['Top 20'].fillna(0).values
    current = np.array([current_downloads_dict.get(country, 0) for country in countries])
    
    # Get threshold values for each category
    threshold_dict = {}
    for country in countries:
        country_data = filtered_df[filtered_df['country'] == country]
        thresholds = {}
        for cat in ['Top 5', 'Top 10', 'Top 20']:
            cat_data = country_data[country_data['category'] == cat]
            if len(cat_data) > 0:
                thresholds[cat] = cat_data.iloc[0]['threshold_downloads']
        threshold_dict[country] = thresholds
    
    # Calculate total downloads
    # If already achieved (needed <= 0), use threshold value; otherwise use current + needed
    top20_total = np.array([
        threshold_dict[country].get('Top 20', current[i] + max(top20_needed[i], 0))
        if top20_needed[i] <= 0 else current[i] + top20_needed[i]
        for i, country in enumerate(countries)
    ])
    top10_total = np.array([
        threshold_dict[country].get('Top 10', current[i] + max(top10_needed[i], 0))
        if top10_needed[i] <= 0 else current[i] + top10_needed[i]
        for i, country in enumerate(countries)
    ])
    top5_total = np.array([
        threshold_dict[country].get('Top 5', current[i] + max(top5_needed[i], 0))
        if top5_needed[i] <= 0 else current[i] + top5_needed[i]
        for i, country in enumerate(countries)
    ])
    
    # Set up the plot
    x = np.arange(len(countries))
    width = 0.2  # Same width as other chart
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Create bars - showing total downloads needed
    bars_current = ax.bar(x - width * 1.5, np.maximum(current, 0), width, 
                         label='Current Downloads', color='#9E9E9E', alpha=0.6, 
                         edgecolor='#616161', linewidth=1.5, linestyle='--')
    bars1 = ax.bar(x - width * 0.5, np.maximum(top20_total, 0), width, label='Top 20 Total', color='#4CAF50', alpha=0.8)
    bars2 = ax.bar(x + width * 0.5, np.maximum(top10_total, 0), width, label='Top 10 Total', color='#2196F3', alpha=0.8)
    bars3 = ax.bar(x + width * 1.5, np.maximum(top5_total, 0), width, label='Top 5 Total', color='#FF9800', alpha=0.8)
    
    # Customize axes first
    ax.set_xlabel('Market', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Downloads', fontsize=12, fontweight='bold')
    ax.set_title(f'Total Downloads Needed for Category Milestones (Current + Additional) - {platform.upper()}', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(countries, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Use log scale if there's a large range (but only if we have positive values)
    # Also ensure y-axis starts at 0 to show all bars
    positive_values = np.concatenate([top5_total[top5_total > 0], top10_total[top10_total > 0], 
                                     top20_total[top20_total > 0], current[current > 0]])
    if len(positive_values) > 0:
        max_val = max(np.max(top5_total[top5_total > 0]) if np.any(top5_total > 0) else 0,
                     np.max(top10_total[top10_total > 0]) if np.any(top10_total > 0) else 0,
                     np.max(top20_total[top20_total > 0]) if np.any(top20_total > 0) else 0,
                     np.max(current[current > 0]) if np.any(current > 0) else 0)
        min_val = np.min(positive_values)
        if max_val / min_val > 100:
            ax.set_yscale('log')
            ax.set_ylabel('Total Downloads (Log Scale)', fontsize=12, fontweight='bold')
            # For log scale, set a small minimum to show all bars
            ax.set_ylim(bottom=max(0.1, min_val * 0.1))
        else:
            # Set y-axis to start at 0 to ensure all bars are visible
            ax.set_ylim(bottom=0, top=max_val * 1.1)
    else:
        # If all are achieved, set a default range
        ax.set_ylim(bottom=0, top=100)
    
    # Now add "ACHIEVED" labels after y-axis is set (for categories already achieved)
    y_max = ax.get_ylim()[1]
    label_y = y_max * 0.01  # Just above x-axis (1% of y-range)
    
    for i, country in enumerate(countries):
        if top20_needed[i] <= 0:
            ax.text(x[i] - width * 0.5, label_y, 
                   'ACHIEVED', fontsize=8, ha='center', va='bottom', 
                   color='#4CAF50', fontweight='bold', rotation=0)
        if top10_needed[i] <= 0:
            ax.text(x[i] + width * 0.5, label_y, 
                   'ACHIEVED', fontsize=8, ha='center', va='bottom', 
                   color='#2196F3', fontweight='bold', rotation=0)
        if top5_needed[i] <= 0:
            ax.text(x[i] + width * 1.5, label_y, 
                   'ACHIEVED', fontsize=8, ha='center', va='bottom', 
                   color='#FF9800', fontweight='bold', rotation=0)
    
    # Add value labels on bars (only if not too crowded)
    if len(countries) <= 20:
        max_display = max(np.max(top5_total[top5_total > 0]) if np.any(top5_total > 0) else 0,
                         np.max(top10_total[top10_total > 0]) if np.any(top10_total > 0) else 0,
                         np.max(top20_total[top20_total > 0]) if np.any(top20_total > 0) else 0,
                         np.max(current[current > 0]) if np.any(current > 0) else 0)
        
        for bars in [bars_current, bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    if height < max_display * 0.9:  # Don't label if too close to top
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height):,}',
                               ha='center', va='bottom', fontsize=9, rotation=0, alpha=0.8)
    
    plt.tight_layout()
    
    # Save
    if group_suffix:
        output_path = f"{output_dir}/total_downloads_bar_chart_{platform}{group_suffix}.png"
    else:
        output_path = f"{output_dir}/total_downloads_bar_chart_{platform}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {output_path}")
    plt.close()

def create_current_vs_target_chart(df, platform="android", output_dir="charts", country_group=None):
    """
    Create a chart showing current downloads vs target downloads for each category.
    If country_group is provided, filter to only show those countries.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Get unique countries (filtered by group if specified)
    if country_group is not None:
        countries = sorted([c for c in df['country'].unique() if c in country_group])
    else:
        countries = sorted(df['country'].unique())
    
    # Only show Top 50, Top 20, Top 10, Top 5
    categories = ['Top 50', 'Top 20', 'Top 10', 'Top 5']
    
    # Filter to only include categories that exist in the data
    available_categories = [cat for cat in categories if cat in df['category'].values]
    
    # Create subplots - one for each category
    n_categories = len(available_categories)
    n_cols = 4
    n_rows = (n_categories + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_categories == 1 else axes
    
    # Color palette
    colors = plt.cm.Set2(np.linspace(0, 1, len(countries)))
    
    for idx, category in enumerate(available_categories):
        ax = axes[idx]
        
        current_values = []
        target_values = []
        country_labels = []
        
        for country in countries:
            country_data = df[(df['country'] == country) & (df['category'] == category)]
            if len(country_data) > 0:
                current = country_data.iloc[0]['current_downloads']
                target = country_data.iloc[0]['threshold_downloads']
                current_values.append(current)
                target_values.append(target)
                country_labels.append(country)
        
        if len(current_values) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(category, fontsize=11, fontweight='bold')
            continue
        
        x = np.arange(len(country_labels))
        width = 0.35
        
        # Plot current and target as grouped bars
        bars1 = ax.bar(x - width/2, current_values, width, label='Current', 
                      color='#2196F3', alpha=0.8)
        bars2 = ax.bar(x + width/2, target_values, width, label='Target', 
                      color='#FF9800', alpha=0.8)
        
        # Don't change color for achieved - keep blue and orange only
        
        ax.set_title(category, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(country_labels, rotation=45, ha='right', fontsize=8)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Use log scale if needed
        all_vals = current_values + target_values
        all_vals = [v for v in all_vals if v > 0]
        use_log_scale = False
        if len(all_vals) > 0:
            max_val = max(all_vals)
            min_val = min(all_vals)
            if max_val / min_val > 100:
                ax.set_yscale('log')
                use_log_scale = True
        
        # Add value labels on bars (after scale is set)
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            
            # Label current (blue) bar
            if height1 > 0:
                label_y = height1 * 1.1 if use_log_scale else height1
                ax.text(bar1.get_x() + bar1.get_width()/2., label_y,
                       f'{int(height1):,}',
                       ha='center', va='bottom', fontsize=9, rotation=0, alpha=0.8)
            
            # Label target (orange) bar
            if height2 > 0:
                label_y = height2 * 1.1 if use_log_scale else height2
                ax.text(bar2.get_x() + bar2.get_width()/2., label_y,
                       f'{int(height2):,}',
                       ha='center', va='bottom', fontsize=9, rotation=0, alpha=0.8)
    
    # Hide unused subplots
    for idx in range(n_categories, len(axes)):
        axes[idx].axis('off')
    
    # Determine group suffix for filename
    group1_countries = ["United States", "United Kingdom (UK)", "Germany", "Spain", 
                       "France", "Australia", "Brazil", "Canada"]
    if country_group == group1_countries:
        group_suffix = "_group1"
        group_label = " (US, UK, DE, ES, FR, AU, BR, CA)"
    elif country_group is not None:
        group_suffix = "_group2"
        group_label = " (Other countries)"
    else:
        group_suffix = ""
        group_label = ""
    
    fig.suptitle(f'Current Downloads vs Target Downloads by Category - {platform.upper()}{group_label}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save
    output_path = f"{output_dir}/current_vs_target_downloads_{platform}{group_suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {output_path}")
    plt.close()

def main():
    print("=" * 80)
    print("Generating Category Comparison Charts")
    print("=" * 80)
    
    for platform in ["android", "iphone"]:
        print(f"\n{'='*80}")
        print(f"Processing {platform.upper()}")
        print(f"{'='*80}")
        
        # Load data
        print(f"\n1. Loading comparison data...")
        df = load_comparison_data(platform)
        print(f"   Found {len(df)} rows across {len(df['country'].unique())} markets")
        
        # Create bar charts - split into two groups
        print(f"\n2. Creating downloads needed bar charts (split by country groups)...")
        
        # Group 1: US, UK, DE, ES, FR, AU, BR, CA
        group1_countries = ["United States", "United Kingdom (UK)", "Germany", "Spain", 
                           "France", "Australia", "Brazil", "Canada"]
        
        # Get all countries and determine group 2
        all_countries = sorted(df['country'].unique())
        group2_countries = [c for c in all_countries if c not in group1_countries]
        
        print(f"   Group 1: {len(group1_countries)} countries")
        create_bar_chart(df, platform, country_group=group1_countries)
        
        print(f"   Group 2: {len(group2_countries)} countries")
        create_bar_chart(df, platform, country_group=group2_countries)
        
        # Create total downloads charts - split into two groups
        print(f"\n2b. Creating total downloads bar charts (split by country groups)...")
        print(f"   Group 1: {len(group1_countries)} countries")
        create_total_downloads_chart(df, platform, country_group=group1_countries)
        
        print(f"   Group 2: {len(group2_countries)} countries")
        create_total_downloads_chart(df, platform, country_group=group2_countries)
        
        # Create current vs target charts - split into two groups
        print(f"\n3. Creating current vs target downloads charts (split by country groups)...")
        print(f"   Group 1: {len(group1_countries)} countries")
        create_current_vs_target_chart(df, platform, country_group=group1_countries)
        
        print(f"   Group 2: {len(group2_countries)} countries")
        create_current_vs_target_chart(df, platform, country_group=group2_countries)
        
        # Print summary
        print(f"\n4. Summary for {platform.upper()}:")
        markets = df['country'].unique()
        print(f"   Markets with data: {len(markets)}")
        
        for category in ['Top 20', 'Top 10', 'Top 5']:
            achieved = len(df[(df['category'] == category) & (df['downloads_needed'] <= 0)])
            print(f"   Markets that achieved {category}: {achieved}")
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print("\nCharts saved to: charts/")
    print("  Downloads Needed Bar Charts (by country group):")
    print("    - downloads_needed_bar_chart_android_group1.png (US, UK, DE, ES, FR, AU, BR, CA)")
    print("    - downloads_needed_bar_chart_android_group2.png (Other countries)")
    print("    - downloads_needed_bar_chart_iphone_group1.png (US, UK, DE, ES, FR, AU, BR, CA)")
    print("    - downloads_needed_bar_chart_iphone_group2.png (Other countries)")
    print("  Total Downloads Bar Charts (Current + Needed, by country group):")
    print("    - total_downloads_bar_chart_android_group1.png (US, UK, DE, ES, FR, AU, BR, CA)")
    print("    - total_downloads_bar_chart_android_group2.png (Other countries)")
    print("    - total_downloads_bar_chart_iphone_group1.png (US, UK, DE, ES, FR, AU, BR, CA)")
    print("    - total_downloads_bar_chart_iphone_group2.png (Other countries)")
    print("  Current vs Target Downloads (Top 50, Top 20, Top 10, Top 5, by country group):")
    print("    - current_vs_target_downloads_android_group1.png (US, UK, DE, ES, FR, AU, BR, CA)")
    print("    - current_vs_target_downloads_android_group2.png (Other countries)")
    print("    - current_vs_target_downloads_iphone_group1.png (US, UK, DE, ES, FR, AU, BR, CA)")
    print("    - current_vs_target_downloads_iphone_group2.png (Other countries)")

if __name__ == "__main__":
    main()
