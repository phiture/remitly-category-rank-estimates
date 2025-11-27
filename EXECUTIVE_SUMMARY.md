# Executive Summary: Category Ranking Download Targets

## Overview
This analysis compares Remitly's current app downloads against category ranking thresholds to determine how many additional downloads are needed to reach Top 5, Top 10, Top 20, Top 50, and Top 100 rankings in each market. The analysis accounts for discrepancies between AppTweak estimates and actual console download data. **Outlier removal** has been applied to actual download data to prevent large download spikes from skewing the results.

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
- **Calculation**: Mean of downloads over the last 90 days (rolling window), with outliers removed using IQR method (1.5 × IQR rule)
- **Purpose**: Ground truth for Remitly's actual download performance
- **Outlier Removal**: Large download spikes (likely from promotional campaigns or data anomalies) are filtered out to prevent skewing the analysis

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
- Average factor: **0.308x** (actual is 31% of AppTweak on average)
- Median factor: **0.255x** (actual is 26% of AppTweak at median)
- Range: 0.115x to 1.188x
- Exception: Mexico (1.19x - AppTweak underestimates)

**iPhone:**
- **15 of 20 markets** show AppTweak **overestimating** downloads (factors < 1.0)
- **5 of 20 markets** show AppTweak **underestimating** downloads (factors > 1.0)
- Average factor: **0.907x** (actual is 91% of AppTweak on average)
- Median factor: **0.947x** (actual is 95% of AppTweak at median)
- Range: 0.387x to 1.155x
- Note: Outlier removal significantly improved accuracy (previously had extreme outliers like IE: 17.0x, BR: 9.5x, NZ: 9.1x)
- Markets with AppTweak underestimating: US (1.15x), MX (1.15x), ES (1.08x), FR (1.10x), SE (1.07x)

### Market Performance

**Android:**
- 5 markets achieved Top 20 (AE, CA, ES, IE, NZ)
- 1 market achieved Top 10 (AE)
- 0 markets achieved Top 5

**iPhone:**
- 3 markets achieved Top 20 (AE, CA, NZ)
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
- **Outlier Removal**: Download data is cleaned using the IQR (Interquartile Range) method, removing values beyond Q3 + 1.5×IQR or below Q1 - 1.5×IQR. This prevents large download spikes from skewing the analysis.
- **Top 20 Estimation**: Top 20 thresholds were estimated using logarithmic interpolation between Top 10 and Top 50 thresholds
- **Country Groups**: Markets split into Group 1 (US, UK, DE, ES, FR, AU, BR, CA) and Group 2 (others) for better chart readability

