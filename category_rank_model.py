import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from pygam import LinearGAM, s

# ============================================================================
# Data Loading and Merging
# ============================================================================

print("=" * 80)
print("Loading and merging data files...")
print("=" * 80)

# Load existing app_rank_downloads.csv
print("Loading app_rank_downloads.csv...")
df_existing = pd.read_csv("app_rank_downloads.csv")
print(f"  Loaded {len(df_existing)} rows from existing file")

# Load and process Android ranking history
print("Loading Android ranking history...")
df_android_rank = pd.read_csv("category-ranking-history_android.csv")
date_col = df_android_rank.columns[0]  # First column is date
# Exclude "Annotations" column if it exists
cols_to_melt = [c for c in df_android_rank.columns if c != 'Annotations' and c != date_col]
df_android_rank = df_android_rank[[date_col] + cols_to_melt]
df_android_rank[date_col] = pd.to_datetime(df_android_rank[date_col])
# Melt to long format
df_android_rank = df_android_rank.melt(
    id_vars=[date_col],
    var_name='app_name',
    value_name='rank'
)
df_android_rank = df_android_rank.rename(columns={date_col: 'date'})
df_android_rank['device'] = 'android'
df_android_rank['country'] = 'us'
# Remove rows with missing ranks
df_android_rank = df_android_rank[df_android_rank['rank'].notna()]
print(f"  Loaded {len(df_android_rank)} rows from Android rankings")

# Load and process iOS ranking history
print("Loading iOS ranking history...")
df_ios_rank = pd.read_csv("category-ranking-history_ios.csv")
date_col = df_ios_rank.columns[0]  # First column is date
# Exclude "Annotations" column if it exists
cols_to_melt = [c for c in df_ios_rank.columns if c != 'Annotations' and c != date_col]
df_ios_rank = df_ios_rank[[date_col] + cols_to_melt]
df_ios_rank[date_col] = pd.to_datetime(df_ios_rank[date_col])
# Melt to long format
df_ios_rank = df_ios_rank.melt(
    id_vars=[date_col],
    var_name='app_name',
    value_name='rank'
)
df_ios_rank = df_ios_rank.rename(columns={date_col: 'date'})
df_ios_rank['device'] = 'iphone'
df_ios_rank['country'] = 'us'
# Remove rows with missing ranks
df_ios_rank = df_ios_rank[df_ios_rank['rank'].notna()]
print(f"  Loaded {len(df_ios_rank)} rows from iOS rankings")

# Load and process Android download estimates
print("Loading Android download estimates...")
df_android_dl = pd.read_csv("download-estimates_android.csv")
date_col = df_android_dl.columns[0]  # First column is date
# Exclude "Annotations" column if it exists
cols_to_melt = [c for c in df_android_dl.columns if c != 'Annotations' and c != date_col]
df_android_dl = df_android_dl[[date_col] + cols_to_melt]
df_android_dl[date_col] = pd.to_datetime(df_android_dl[date_col])
# Melt to long format
df_android_dl = df_android_dl.melt(
    id_vars=[date_col],
    var_name='app_name',
    value_name='downloads'
)
df_android_dl = df_android_dl.rename(columns={date_col: 'date'})
df_android_dl['device'] = 'android'
df_android_dl['country'] = 'us'
# Remove rows with missing downloads
df_android_dl = df_android_dl[df_android_dl['downloads'].notna()]
print(f"  Loaded {len(df_android_dl)} rows from Android downloads")

# Load and process iOS download estimates
print("Loading iOS download estimates...")
df_ios_dl = pd.read_csv("download-estimates_ios.csv")
date_col = df_ios_dl.columns[0]  # First column is date
# Exclude "Annotations" column if it exists
cols_to_melt = [c for c in df_ios_dl.columns if c != 'Annotations' and c != date_col]
df_ios_dl = df_ios_dl[[date_col] + cols_to_melt]
df_ios_dl[date_col] = pd.to_datetime(df_ios_dl[date_col])
# Melt to long format
df_ios_dl = df_ios_dl.melt(
    id_vars=[date_col],
    var_name='app_name',
    value_name='downloads'
)
df_ios_dl = df_ios_dl.rename(columns={date_col: 'date'})
df_ios_dl['device'] = 'iphone'
df_ios_dl['country'] = 'us'
# Remove rows with missing downloads
df_ios_dl = df_ios_dl[df_ios_dl['downloads'].notna()]
print(f"  Loaded {len(df_ios_dl)} rows from iOS downloads")

# Merge ranking and download data for Android
print("Merging Android data...")
df_android = pd.merge(
    df_android_rank,
    df_android_dl,
    on=['date', 'app_name', 'device', 'country'],
    how='inner'
)
print(f"  Merged Android data: {len(df_android)} rows")

# Merge ranking and download data for iOS
print("Merging iOS data...")
df_ios = pd.merge(
    df_ios_rank,
    df_ios_dl,
    on=['date', 'app_name', 'device', 'country'],
    how='inner'
)
print(f"  Merged iOS data: {len(df_ios)} rows")

# Combine Android and iOS data
df_us = pd.concat([df_android, df_ios], ignore_index=True)

# Add category information (assuming Finance category)
df_us['category'] = 6015  # Finance category code
df_us['category_name'] = 'Finance'

# Add app_id if needed (we'll use app_name as identifier for now)
# You may want to add a mapping if you have app_ids
df_us['app_id'] = None

# Combine with existing data (optional - you may want to keep them separate)
# For now, we'll use only the US data
print(f"\nCombined US data: {len(df_us)} rows")
print(f"  Android: {len(df_android)} rows")
print(f"  iOS: {len(df_ios)} rows")

# ============================================================================
# Data Cleaning
# ============================================================================

print("=" * 80)
print("Cleaning data...")
print("=" * 80)

# Use US data
df = df_us.copy()

# Basic cleaning
df['date'] = pd.to_datetime(df['date'])
df = df[df['downloads'].notna()]

# Filter out zero or negative downloads (can't take log of zero/negative)
df = df[df['downloads'] > 0]

# Filter for Finance category and US (already filtered, but keeping for consistency)
df = df[df['category_name'] == "Finance"]
df = df[df['country'] == "us"]

# Log-transform (rank responds nonlinearly to installs)
df['log_downloads'] = np.log(df['downloads'])

# Additional cleaning: remove any infinite or NaN values that might have been created
df = df[np.isfinite(df['log_downloads'])]
df = df[df['rank'].notna()]

print(f"Final data: {len(df)} rows")
print(f"  Downloads range: {df['downloads'].min():.0f} - {df['downloads'].max():.0f}")
print(f"  Rank range: {df['rank'].min():.0f} - {df['rank'].max():.0f}")
print(f"  Devices: {df['device'].unique()}")
print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
print()


# ============================================================================
# Tier definitions
# ============================================================================

# Tiers defined as inclusive ranges on rank
TIERS = {
    "1-10":  (1, 10),
    "11-20": (11, 20),
    "21-30": (21, 30),
    "30+":  (31, np.inf)   # 30 or worse (higher rank number)
}


# ============================================================================
# Model 1: Log-Linear Regression (OLS)
# ============================================================================

print("=" * 80)
print("MODEL 1: Log-Linear Regression (OLS)")
print("=" * 80)

countries = df['country'].unique()
country_models = {}

for c in countries:
    df_c = df[(df['country'] == c)]
    df_c = df_c[df_c['rank'] <= 300]  # optional: focus on ranks up to 300

    if len(df_c) < 500:
        print(f"Skipping {c}, not enough data ({len(df_c)} rows)")
        continue

    ols_model = smf.ols("rank ~ log_downloads", data=df_c).fit()
    country_models[c] = ols_model
    print(f"{c}: n={len(df_c)}, R^2={ols_model.rsquared:.3f}")


# Extract parameters
alpha = ols_model.params['Intercept']
beta = ols_model.params['log_downloads']

print(f"Model equation: rank = {alpha:.2f} + {beta:.2f} * log(downloads)")
print()

# Function to calculate downloads needed for target rank
def downloads_needed_ols(target_rank):
    return float(np.exp((target_rank - alpha) / beta))

# Store OLS results for CSV
ols_results = []
print("Downloads needed for target ranks (OLS model):")
for r in [50, 75, 100]:
    downloads = downloads_needed_ols(r)
    ols_results.append({
        'model': 'OLS',
        'target_rank': r,
        'downloads_needed': downloads
    })
    print(f"  Rank {r:3d}: {downloads:8.0f} downloads")
print()

# ============================================================================
# Model 2: Generalized Additive Model (GAM)
# ============================================================================

print("=" * 80)
print("MODEL 2: Generalized Additive Model (LinearGAM)")
print("=" * 80)

countries = df['country'].unique()
country_models = {}

for c in countries:
    df_c = df[(df['country'] == c)]
    df_c = df_c[df_c['rank'] <= 300]  # optional: focus on ranks up to 300

    if len(df_c) < 500:
        print(f"Skipping {c}, not enough data ({len(df_c)} rows)")
        continue

    gam = LinearGAM(s(0)).fit(df_c['log_downloads'].values, df_c['rank'].values)
    country_models[c] = gam
    print(f"{c}: n={len(df_c)}, R^2={gam.statistics_['pseudo_r2']['explained_deviance']:.3f}")


# Function to invert GAM for target rank
def invert_gam(target_rank, search_grid=np.linspace(20, 150000, 5000)):
    preds = gam.predict(np.log(search_grid).reshape(-1, 1))
    idx = (np.abs(preds - target_rank)).argmin()
    return search_grid[idx]

# Store GAM results for CSV
gam_results = []
print("Downloads needed for target ranks (GAM model):")
for r in [10, 20, 50, 75, 100]:
    downloads = invert_gam(r)
    gam_results.append({
        'model': 'GAM',
        'target_rank': r,
        'downloads_needed': downloads
    })
    print(f"  Rank {r:3d}: {downloads:8.0f} downloads")
print()

# Add top 10 and top 20 to OLS results
print("Downloads needed for top ranks (OLS model):")
for r in [10, 20]:
    downloads = downloads_needed_ols(r)
    ols_results.append({
        'model': 'OLS',
        'target_rank': r,
        'downloads_needed': downloads
    })
    print(f"  Rank {r:3d}: {downloads:8.0f} downloads")
print()

# ============================================================================
# Visualization: Comparison of Both Models
# ============================================================================

print("=" * 80)
print("Generating plots...")
print("=" * 80)

def make_country_simulator(model):
    resid_std = model.resid.std()

    def simulate_rank(downloads, n_sim=5000):
        log_d = np.log(downloads)
        preds = model.predict(pd.DataFrame({"log_downloads": [log_d] * n_sim}))
        noise = np.random.normal(0, resid_std, n_sim)
        return preds + noise

    return simulate_rank





# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Prepare data for plotting
x_plot = np.linspace(df['downloads'].min(), df['downloads'].max(), 200)
x_log = np.log(x_plot)

# Plot 1: OLS Model
ax1 = axes[0]
ax1.scatter(df['downloads'], df['rank'], alpha=0.4, s=20, label='Data')
y_hat_ols = ols_model.predict(pd.DataFrame({'log_downloads': x_log}))
ax1.plot(x_plot, y_hat_ols, 'r-', linewidth=2, label='OLS Model')
ax1.set_xlabel("Downloads", fontsize=12)
ax1.set_ylabel("Rank", fontsize=12)
ax1.set_title("Log-Linear Regression (OLS)", fontsize=14, fontweight='bold')
ax1.invert_yaxis()  # rank 1 is top
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: GAM Model
ax2 = axes[1]
ax2.scatter(df['downloads'], df['rank'], alpha=0.4, s=20, label='Data')
y_hat_gam = gam.predict(x_log.reshape(-1, 1))
ax2.plot(x_plot, y_hat_gam, 'g-', linewidth=2, label='GAM Model')
ax2.set_xlabel("Downloads", fontsize=12)
ax2.set_ylabel("Rank", fontsize=12)
ax2.set_title("Generalized Additive Model (GAM)", fontsize=14, fontweight='bold')
ax2.invert_yaxis()  # rank 1 is top
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: model_comparison.png")
plt.show()

# Combined plot showing both models
plt.figure(figsize=(10, 6))
plt.scatter(df['downloads'], df['rank'], alpha=0.3, s=15, label='Data', color='gray')
plt.plot(x_plot, y_hat_ols, 'r-', linewidth=2, label='OLS Model', alpha=0.8)
plt.plot(x_plot, y_hat_gam, 'g-', linewidth=2, label='GAM Model', alpha=0.8)
plt.xlabel("Downloads", fontsize=12)
plt.ylabel("Rank", fontsize=12)
plt.title("Model Comparison: OLS vs GAM", fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()  # rank 1 is top
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('models_overlaid.png', dpi=150, bbox_inches='tight')
print("Saved: models_overlaid.png")
plt.show()

# ============================================================================
# Simulation: Rank Distribution for Given Downloads
# ============================================================================

print("=" * 80)
print("Simulation Analysis")
print("=" * 80)

import numpy as np

def make_country_simulator(model):
    resid_std = model.resid.std()

    def simulate_rank(downloads, n_sim=5000):
        log_d = np.log(downloads)
        preds = model.predict(pd.DataFrame({"log_downloads": [log_d] * n_sim}))
        noise = np.random.normal(0, resid_std, n_sim)
        return preds + noise

    return simulate_rank


# Probability of landing in a given tier at a given download level
def prob_in_tier(simulate_rank_fn, downloads, tier, n_sim=5000):
    low, high = TIERS[tier]
    sim = simulate_rank_fn(downloads, n_sim=n_sim)
    return np.mean((sim >= low) & (sim <= high))


def find_download_target_tier(simulate_rank_fn, tier, target_prob=0.7,
                              search=np.arange(50, 10000, 25),
                              n_sim=2000):
    for d in search:
        p = prob_in_tier(simulate_rank_fn, d, tier, n_sim=n_sim)
        if p >= target_prob:
            return d
    return None

# Find downloads needed for specific rank targets
def find_download_target_rank(model, target_rank, target_prob=0.7, 
                              search=np.arange(50, 150000, 100),
                              n_sim=3000):
    """Find minimum downloads needed to achieve target rank with target probability"""
    def simulate_rank_ols(downloads, n_sim=n_sim):
        log_d = np.log(downloads)
        preds = model.predict(pd.DataFrame({"log_downloads": [log_d] * n_sim}))
        noise = np.random.normal(0, model.resid.std(), n_sim)
        return preds + noise
    
    for d in search:
        sims = simulate_rank_ols(d, n_sim=n_sim)
        if np.mean(sims <= target_rank) >= target_prob:
            return d
    return None


target_probs = [0.5, 0.7, 0.9]

for c, m in country_models.items():
    print("=" * 80)
    print(f"Country: {c}")
    print("=" * 80)

    sim_fn = make_country_simulator(m)

    for tier in TIERS.keys():
        for p in target_probs:
            d_needed = find_download_target_tier(
                simulate_rank_fn=sim_fn,
                tier=tier,
                target_prob=p,
                search=np.arange(50, 10000, 25),
                n_sim=2000
            )
            print(f"  Tier {tier}, P(in tier) ≥ {int(p*100):2d}% -> "
                  f"{d_needed if d_needed is not None else 'None in search range'} downloads")
    print()


# Store simulation results for CSV (will be populated if needed)
simulation_results = []

# ============================================================================
# Top 10 and Top 20 Analysis (Combined Android + iOS)
# ============================================================================

print("=" * 80)
print("Top 10 and Top 20 Download Requirements (Combined)")
print("=" * 80)

target_probs = [0.5, 0.7, 0.9]
rank_target_results = []

# Use the OLS model from country_models (assuming 'us' is the country)
if 'us' in country_models:
    us_model = country_models['us']
    for target_rank in [10, 20]:
        print(f"\nRank ≤ {target_rank} analysis (Combined):")
        for target_prob in target_probs:
            downloads_needed = find_download_target_rank(
                model=us_model,
                target_rank=target_rank,
                target_prob=target_prob,
                search=np.arange(50, 150000, 200),
                n_sim=2000
            )
            rank_target_results.append({
                'device': 'combined',
                'target_rank': target_rank,
                'target_probability': target_prob,
                'downloads_needed': downloads_needed if downloads_needed else None
            })
            if downloads_needed:
                print(f"  P(rank ≤ {target_rank}) ≥ {int(target_prob*100):2d}%: {downloads_needed:,.0f} downloads")
            else:
                print(f"  P(rank ≤ {target_rank}) ≥ {int(target_prob*100):2d}%: Not found in search range")
else:
    print("Warning: 'us' country model not found, skipping combined analysis")
print()

# ============================================================================
# Device-Specific Analysis: Android and iOS
# ============================================================================

print("=" * 80)
print("Device-Specific Analysis: Android and iOS")
print("=" * 80)

# Split data by device
df_android_clean = df[df['device'] == 'android'].copy()
df_ios_clean = df[df['device'] == 'iphone'].copy()

print(f"\nAndroid data: {len(df_android_clean)} rows")
print(f"  Downloads range: {df_android_clean['downloads'].min():.0f} - {df_android_clean['downloads'].max():.0f}")
print(f"  Rank range: {df_android_clean['rank'].min():.0f} - {df_android_clean['rank'].max():.0f}")

print(f"\niOS data: {len(df_ios_clean)} rows")
print(f"  Downloads range: {df_ios_clean['downloads'].min():.0f} - {df_ios_clean['downloads'].max():.0f}")
print(f"  Rank range: {df_ios_clean['rank'].min():.0f} - {df_ios_clean['rank'].max():.0f}")

# Function to run analysis for a specific device
def run_device_analysis(df_device, device_name):
    """Run full analysis for a specific device"""
    print(f"\n{'='*80}")
    print(f"Analysis for {device_name.upper()}")
    print(f"{'='*80}")
    
    # Fit OLS model
    ols_device = smf.ols("rank ~ log_downloads", data=df_device).fit()
    alpha_d = ols_device.params['Intercept']
    beta_d = ols_device.params['log_downloads']
    
    print(f"\nOLS Model for {device_name}:")
    print(f"  R² = {ols_device.rsquared:.4f}")
    print(f"  Equation: rank = {alpha_d:.2f} + {beta_d:.2f} * log(downloads)")
    
    def downloads_needed_device(target_rank):
        return float(np.exp((target_rank - alpha_d) / beta_d))
    
    def simulate_rank_device(downloads, n_sim=5000):
        log_d = np.log(downloads)
        preds = ols_device.predict(pd.DataFrame({"log_downloads": [log_d] * n_sim}))
        noise = np.random.normal(0, ols_device.resid.std(), n_sim)
        return preds + noise
    
    def find_download_target_device(target_rank, target_prob=0.7, 
                                    search=np.arange(50, 150000, 200),
                                    n_sim=2000):
        for d in search:
            sims = simulate_rank_device(d, n_sim=n_sim)
            if np.mean(sims <= target_rank) >= target_prob:
                return d
        return None
    
    # Top 10 and Top 20 analysis
    print(f"\nTop 10 and Top 20 Download Requirements for {device_name}:")
    device_results = []
    
    for target_rank in [10, 20]:
        print(f"\n  Rank ≤ {target_rank}:")
        # Direct model prediction
        direct_pred = downloads_needed_device(target_rank)
        print(f"    Model prediction: {direct_pred:,.0f} downloads")
        
        # Simulation-based with probabilities
        for target_prob in [0.5, 0.7, 0.9]:
            downloads_needed = find_download_target_device(
                target_rank=target_rank,
                target_prob=target_prob,
                search=np.arange(50, 150000, 200),
                n_sim=2000
            )
            device_results.append({
                'device': device_name,
                'target_rank': target_rank,
                'target_probability': target_prob,
                'downloads_needed': downloads_needed if downloads_needed else None
            })
            if downloads_needed:
                print(f"    P(rank ≤ {target_rank}) ≥ {int(target_prob*100):2d}%: {downloads_needed:,.0f} downloads")
            else:
                print(f"    P(rank ≤ {target_rank}) ≥ {int(target_prob*100):2d}%: Not found")
    
    return device_results

# Run analysis for Android
android_results = run_device_analysis(df_android_clean, 'android')

# Run analysis for iOS  
ios_results = run_device_analysis(df_ios_clean, 'ios')

# Combine all device results
rank_target_results.extend(android_results)
rank_target_results.extend(ios_results)

print("=" * 80)
print("Tier-level download targets from OLS + simulation")
print("=" * 80)

target_probs = [0.5, 0.7, 0.9]  # median, 70%, 90% probability

results = []
# Use the OLS model from country_models (assuming 'us' is the country)
if 'us' in country_models:
    us_model = country_models['us']
    sim_fn_us = make_country_simulator(us_model)
    
    for tier_name in TIERS.keys():
        for p in target_probs:
            d_needed = find_download_target_tier(
                simulate_rank_fn=sim_fn_us,
                tier=tier_name,
                target_prob=p,
                search=np.arange(50, 150000, 200),
                n_sim=2000
            )
            results.append({
                "tier": tier_name,
                "target_prob": p,
                "downloads_needed": d_needed
            })
            if d_needed:
                print(f"Tier {tier_name}, P(in tier) ≥ {int(p*100):2d}% -> {d_needed:,.0f} downloads")
            else:
                print(f"Tier {tier_name}, P(in tier) ≥ {int(p*100):2d}% -> None in search range")
else:
    print("Warning: 'us' country model not found, skipping tier analysis")
print()

# ============================================================================
# Confidence Intervals
# ============================================================================

def bootstrap_download_target_tier(df, tier_name, target_prob=0.7,
                                   search=np.arange(50, 150000, 500),
                                   n_boot=200, n_sim=1000):
    """
    Bootstrap CI for downloads needed to reach a given tier with target_prob.
    Returns (median, lower, upper) for downloads.
    """
    boot_targets = []

    for b in range(n_boot):
        sample = df.sample(frac=1.0, replace=True)

        # Refit OLS on bootstrap sample
        m = smf.ols("rank ~ log_downloads", data=sample).fit()

        # Local simulate function using this bootstrap model
        def simulate_rank_boot(downloads, n_sim=n_sim):
            log_d = np.log(downloads)
            preds = m.predict(pd.DataFrame({"log_downloads": [log_d]*n_sim}))
            noise = np.random.normal(0, m.resid.std(), n_sim)
            return preds + noise

        def prob_in_tier_boot(downloads, tier_name, n_sim=n_sim):
            low, high = TIERS[tier_name]
            sim = simulate_rank_boot(downloads, n_sim=n_sim)
            if np.isinf(high):
                return np.mean(sim >= low)
            else:
                return np.mean((sim >= low) & (sim <= high))

        # Find downloads for this bootstrap model
        target = None
        for d in search:
            p = prob_in_tier_boot(d, tier_name, n_sim=n_sim)
            if p >= target_prob:
                target = d
                break

        if target is not None:
            boot_targets.append(target)

    if len(boot_targets) == 0:
        return None, None, None

    lower, median, upper = np.percentile(boot_targets, [2.5, 50, 97.5])
    return median, lower, upper

print("=" * 80)
print("Bootstrap CIs for tier-level download targets (OLS)")
print("=" * 80)

# Store bootstrap results for CSV
bootstrap_results = []
for tier_name in TIERS.keys():
    median, lower, upper = bootstrap_download_target_tier(
        df, tier_name, target_prob=0.7,
        search=np.arange(50, 150000, 500),
        n_boot=200, n_sim=800
    )
    bootstrap_results.append({
        'tier': tier_name,
        'target_probability': 0.7,
        'median_downloads': median if median is not None else None,
        'ci_lower': lower if lower is not None else None,
        'ci_upper': upper if upper is not None else None
    })
    print(f"Tier {tier_name}, P(in tier) ≥ 70%:")
    if median is not None:
        print(f"  median: {median:.0f}, 95% CI: [{lower:.0f}, {upper:.0f}]")
    else:
        print(f"  Could not find solution in search range")
print()


# ============================================================================
# Probability curves by tier
# ============================================================================

print("=" * 80)
print("Probability curves by tier")
print("=" * 80)

import matplotlib.pyplot as plt

download_grid = np.arange(50, 10000, 50)

for c, m in country_models.items():
    sim_fn = make_country_simulator(m)
    plt.figure(figsize=(10, 6))

    for tier in TIERS.keys():
        probs = [prob_in_tier(sim_fn, d, tier, n_sim=2000) for d in download_grid]
        plt.plot(download_grid, probs, label=f"Tier {tier}")

    plt.xlabel("Downloads")
    plt.ylabel("P(rank in tier)")
    plt.title(f"Tier probability vs downloads – {c.upper()} (Finance, iPhone)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'tier_probability_curves_{c}.png', dpi=150, bbox_inches='tight')
    print(f"Saved: tier_probability_curves_{c}.png")
    plt.show()

# ============================================================================
# Tier Download Targets by Country
# ============================================================================

print("=" * 80)
print("Tier Download Targets by Country")
print("=" * 80)

target_probs = [0.5, 0.7, 0.9]  # median, 70%, 90% probability
rows = []

for c, m in country_models.items():
    sim_fn = make_country_simulator(m)
    
    for tier in TIERS.keys():
        for p in target_probs:
            d_needed = find_download_target_tier(
                simulate_rank_fn=sim_fn,
                tier=tier,
                target_prob=p,
                search=np.arange(50, 150000, 200),
                n_sim=2000
            )
            rows.append({
                "country": c,
                "tier": tier,
                "target_prob": p,
                "downloads_needed": d_needed
            })

targets_df = pd.DataFrame(rows)
targets_df.to_csv("tier_download_targets_by_country.csv", index=False)
print("Saved: tier_download_targets_by_country.csv")
print(f"\nSummary: {len(targets_df)} rows saved")
print(targets_df.head(20))
print()

# ============================================================================
# Save Results to CSV Files
# ============================================================================

print("=" * 80)
print("Saving results to CSV files...")
print("=" * 80)

# 1. Model Summary Statistics
model_summary = pd.DataFrame([{
    'model': 'OLS',
    'r_squared': ols_model.rsquared,
    'adj_r_squared': ols_model.rsquared_adj,
    'intercept': alpha,
    'log_downloads_coef': beta,
    'n_observations': len(df),
    'aic': ols_model.aic,
    'bic': ols_model.bic
}, {
    'model': 'GAM',
    'r_squared': gam.statistics_['pseudo_r2']['explained_deviance'],
    'adj_r_squared': None,  # GAM doesn't have adj R² in same way
    'intercept': None,
    'log_downloads_coef': None,
    'n_observations': len(df),
    'aic': gam.statistics_['AIC'],
    'bic': None
}])
model_summary.to_csv('model_summary.csv', index=False)
print("Saved: model_summary.csv")

# 2. Target Rank Downloads (both models)
target_rank_downloads = pd.DataFrame(ols_results + gam_results)
target_rank_downloads.to_csv('target_rank_downloads.csv', index=False)
print("Saved: target_rank_downloads.csv")

# 3. Tier-level Download Targets
tier_targets_df = pd.DataFrame(results)
tier_targets_df.to_csv('tier_download_targets.csv', index=False)
print("Saved: tier_download_targets.csv")

# 4. Bootstrap Confidence Intervals
bootstrap_df = pd.DataFrame(bootstrap_results)
bootstrap_df.to_csv('bootstrap_confidence_intervals.csv', index=False)
print("Saved: bootstrap_confidence_intervals.csv")

# 5. Simulation Results
simulation_df = pd.DataFrame(simulation_results)
simulation_df.to_csv('simulation_results.csv', index=False)
print("Saved: simulation_results.csv")

# 6. Rank Target Results
rank_target_df = pd.DataFrame(rank_target_results)
rank_target_df.to_csv('rank_target_results.csv', index=False)
print("Saved: rank_target_results.csv")

# 7. Tier Probability Curves (detailed grid)
tier_prob_curves = {}
for c, m in country_models.items():
    sim_fn = make_country_simulator(m)
    for tier in TIERS.keys():
        probs = [prob_in_tier(sim_fn, d, tier, n_sim=2000) for d in download_grid]
        tier_prob_curves[f'{c}_{tier}'] = probs

# Create DataFrame with downloads and tier probabilities
tier_prob_data = {'downloads': download_grid}
for tier in TIERS.keys():
    # Use the first country's probabilities (or average if multiple countries)
    if country_models:
        first_country = list(country_models.keys())[0]
        sim_fn = make_country_simulator(country_models[first_country])
        probs = [prob_in_tier(sim_fn, d, tier, n_sim=2000) for d in download_grid]
        tier_prob_data[f'tier_{tier}_probability'] = probs

tier_prob_df = pd.DataFrame(tier_prob_data)
tier_prob_df.to_csv('tier_probability_curves_data.csv', index=False)
print("Saved: tier_probability_curves_data.csv")

print()

# ============================================================================
# Analysis complete!
# ============================================================================

print("=" * 80)
print("Analysis complete!")
print("=" * 80)
print("\nAll results have been saved to CSV files:")
print("  - model_summary.csv")
print("  - target_rank_downloads.csv")
print("  - tier_download_targets.csv")
print("  - tier_download_targets_by_country.csv")
print("  - bootstrap_confidence_intervals.csv")
print("  - simulation_results.csv")
print("  - rank_target_results.csv")
print("  - tier_probability_curves_data.csv")
print()
