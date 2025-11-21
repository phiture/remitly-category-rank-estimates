# Executive Summary: Category Ranking Download Targets

## Overview
This analysis compares Remitly's current app downloads against category ranking thresholds to determine how many additional downloads are needed to reach Top 5, Top 10, Top 20, Top 50, and Top 100 rankings in each market. The analysis accounts for discrepancies between AppTweak estimates and actual console download data.

## Data Sources

### 1. Category Thresholds (AppTweak Estimates)
- **Source**: `market_downloads_category_1_{platform}.csv` and `market_downloads_category_2_{platform}.csv`
- **Content**: Download thresholds required to achieve each category rank (Top 1, Top 2, Top 3, Top 5, Top 10, Top 50, Top 100) for each market
- **Platform**: Separate files for Android and iOS
- **Markets**: 20 markets (US, UK, DE, ES, FR, AU, BR, CA, AE, AT, FI, DK, IT, RO, PT, SE, IE, NO, NZ, MX)

### 2. AppTweak Download Estimates
- **Source**: `world-download-estimates_{platform}.csv`
- **Content**: 90-day total download estimates from AppTweak
- **Calculation**: Total downloads ÷ 90 = daily average
- **Purpose**: Used to compare against actual console downloads

### 3. Actual Console Downloads
- **Source**: `remitly_actual_downloads_{platform}.csv`
- **Content**: Actual daily downloads from app store consoles
- **Calculation**: Mean of downloads over the last 90 days (rolling window)
- **Purpose**: Ground truth for Remitly's actual download performance

## Methodology: Aligning AppTweak to Actual Downloads

### Step 1: Calculate Adjustment Factors
For each market and platform, we calculated the ratio of actual downloads to AppTweak estimates:

```
Adjustment Factor = Actual Downloads / AppTweak Downloads
```

**Example (US iPhone):**
- AppTweak estimate: 5,939 downloads/day
- Actual downloads: 7,549 downloads/day
- Adjustment factor: 7,549 / 5,939 = **1.271x**

This means actual downloads are 27% higher than AppTweak estimates for US iPhone.

### Step 2: Apply Adjustment to Category Thresholds
Since category thresholds are based on AppTweak estimates, we adjusted them to match the actual download scale:

```
Adjusted Threshold = Original Threshold × Adjustment Factor
```

**Example (US iPhone Top 10):**
- Original threshold: 14,906 downloads
- Adjustment factor: 1.271x
- Adjusted threshold: 14,906 × 1.271 = **18,947 downloads**

### Step 3: Calculate Downloads Needed
For each market and category, we calculated:

```
Downloads Needed = max(0, Adjusted Threshold - Current Downloads)
```

If `Downloads Needed ≤ 0`, the market has already achieved that category rank.

## Key Findings

### Adjustment Factors by Platform

**Android:**
- **18 of 19 markets** show AppTweak **overestimating** downloads (factors < 1.0)
- Average factor: **0.324x** (actual is 32% of AppTweak on average)
- Median factor: **0.263x** (actual is 26% of AppTweak at median)
- Range: 0.115x to 1.192x
- Exception: Mexico (1.19x - AppTweak underestimates)

**iPhone:**
- **13 of 20 markets** show AppTweak **underestimating** downloads (factors > 1.0)
- Average factor: **2.940x** (actual is 294% of AppTweak on average)
- Median factor: **1.316x** (actual is 132% of AppTweak at median)
- Range: 0.793x to 17.047x
- Note: High average driven by outliers (IE: 17.0x, BR: 9.5x, NZ: 9.1x)
- Some European markets (AT, DE, DK, RO) closer to 1.0x or below

### Market Performance

**Android:**
- 5 markets achieved Top 20 (AE, ES, CA, IE, NZ)
- 1 market achieved Top 10 (AE)
- 0 markets achieved Top 5

**iPhone:**
- 3 markets achieved Top 20 (CA, AE, NZ)
- 1 market achieved Top 10 (AE)
- 0 markets achieved Top 5

## Outputs Generated

### 1. Comparison Files
- `apptweak_vs_actual_comparison_{platform}.csv`: Adjustment factors for each market
- `remitly_category_comparison_{platform}_aligned_with_top20.csv`: Downloads needed for each category

### 2. Adjusted Category Thresholds
- `market_downloads_category_1_{platform}_adjusted.csv`
- `market_downloads_category_2_{platform}_adjusted.csv`
- These files contain the adjusted thresholds used for all calculations

### 3. Visualizations
- **Downloads Needed Bar Charts**: Current downloads + downloads needed for Top 20, Top 10, Top 5
- **Total Downloads Bar Charts**: Total downloads required (current + needed) for each category
- **Current vs Target Charts**: Side-by-side comparison of current vs target downloads

### 4. Data Tables (for Google Sheets)
- Downloads needed tables (by country group)
- Total downloads tables (by country group)
- Current vs target tables (by country group)

## Why This Matters

1. **Accurate Targets**: By aligning AppTweak estimates to actual console data, we ensure category targets reflect real-world download scales
2. **Platform Differences**: The analysis reveals significant differences between Android and iPhone download patterns and AppTweak accuracy
3. **Actionable Insights**: Clear visibility into which markets are closest to achieving higher category ranks and how many additional downloads are needed

## Technical Notes

- **Time Period**: All calculations use 90-day rolling averages for stability
- **Top 20 Estimation**: Top 20 thresholds were estimated using logarithmic interpolation between Top 10 and Top 50 thresholds
- **Country Groups**: Markets split into Group 1 (US, UK, DE, ES, FR, AU, BR, CA) and Group 2 (others) for better chart readability

