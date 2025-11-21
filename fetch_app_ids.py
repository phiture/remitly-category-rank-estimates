import pandas as pd
import requests
from google_play_scraper import search
import time
import urllib.parse
import re

INPUT_CSV = "competitor_names/Remitly - AppTweak Data - app names.csv"
OUTPUT_CSV = "competitor_names/app_ids_output.csv"

# ------------- iOS LOOKUP (Apple Search API) ------------- #

def get_ios_app_id(app_name, country="us", max_results=5):
    """
    Use the iTunes Search API to find an iOS app ID for a given app name.
    Returns the first matching trackId or None.
    """
    base_url = "https://itunes.apple.com/search"
    params = {
        "term": app_name,
        "entity": "software",
        "limit": max_results,
        "country": country
    }

    try:
        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            return None

        # Very simple heuristic: take the first result
        # You can add stricter matching if needed
        return results[0].get("trackId")
    except Exception as e:
        print(f"[iOS] Error looking up '{app_name}': {e}")
        return None

# ------------- ANDROID LOOKUP (Google Play Scraper) ------------- #

def normalize_app_name(app_name):
    """Create search variations of the app name"""
    variations = []
    
    # Original name
    variations.append(app_name)
    
    # Remove common suffixes
    suffixes_to_remove = [
        " - Mobile Banking", ": Mobile Banking", " Mobile Banking",
        " – Mobile Banking", " - Bank & Invest", ": Bank & Invest",
        " - Pay, Send, Save", ": Pay, Send, Save",
        " - Pay & Send Money", ": Pay & Send Money",
        " - Pay, Send, Save", ": Pay, Send, Save",
        " - Send Money & Transfer", ": Send Money & Transfer",
        " - Send, spend and save", ": Send, spend and save",
        " - Global Money", ": Global Money",
        " - Mobile app", ": Mobile app",
        " - Mobile", ": Mobile", " Mobile",
        " App", "®", " –", " -"
    ]
    
    cleaned = app_name
    for suffix in suffixes_to_remove:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)].strip()
            variations.append(cleaned)
    
    # Remove everything after colon or dash (take first part)
    if ":" in app_name:
        variations.append(app_name.split(":")[0].strip())
    if " – " in app_name:
        variations.append(app_name.split(" – ")[0].strip())
    if " - " in app_name:
        variations.append(app_name.split(" - ")[0].strip())
    
    # Remove special characters but keep spaces
    cleaned_special = re.sub(r'[^\w\s]', ' ', app_name).strip()
    if cleaned_special != app_name:
        variations.append(cleaned_special)
    
    # Deduplicate while preserving order
    seen = set()
    unique_variations = []
    for v in variations:
        if v and v not in seen:
            seen.add(v)
            unique_variations.append(v)
    
    return unique_variations

def score_match(app_name, result_title, result_app_id):
    """Score how well a search result matches the app name"""
    if not result_title or not result_app_id:
        return 0
    
    app_lower = app_name.lower()
    title_lower = result_title.lower()
    
    # Exact match (after normalization)
    if app_lower == title_lower:
        return 100
    
    # App name is in title
    if app_lower in title_lower:
        return 80
    
    # Key words match (split and check)
    app_words = set(app_lower.split())
    title_words = set(title_lower.split())
    common_words = app_words.intersection(title_words)
    
    if len(common_words) >= 2:
        return 60 + (len(common_words) * 5)
    
    # At least one word matches
    if common_words:
        return 30
    
    return 0

def get_android_app_id(app_name, lang="en", countries=["us", "gb", "au", "ca"], n_results=20):
    """
    Use google-play-scraper to search Google Play for the app and return
    the best matching appId (package name), or None.
    
    Tries multiple search variations and countries for better matching.
    """
    search_variations = normalize_app_name(app_name)
    
    best_match = None
    best_score = 0
    
    for search_term in search_variations[:3]:  # Try up to 3 variations
        for country in countries:
            try:
                results = search(
                    search_term,
                    lang=lang,
                    country=country,
                    n_hits=n_results
                )

                if not results or len(results) == 0:
                    continue
                
                # Score each result and find the best match
                for result in results:
                    if not result:
                        continue
                    
                    try:
                        result_title = result.get("title", "")
                        result_app_id = result.get("appId")
                        
                        if not result_app_id:
                            continue
                        
                        score = score_match(app_name, result_title, result_app_id)
                        
                        if score > best_score:
                            best_score = score
                            best_match = result_app_id
                            
                            # If we have a very good match, return early
                            if score >= 80:
                                return best_match
                    except (KeyError, TypeError, AttributeError):
                        continue
                        
            except Exception as e:
                # Continue to next variation/country on error
                continue
    
    # Return best match if we found something reasonable
    if best_score >= 30:
        return best_match
    
    return None

# ------------- MAIN PIPELINE ------------- #

def main():
    # Check if output file exists - if so, load it and only update missing Android IDs
    existing_data = None
    try:
        existing_df = pd.read_csv(OUTPUT_CSV)
        existing_data = existing_df.set_index("app_name").to_dict("index")
        print(f"Found existing output file with {len(existing_df)} apps")
        print(f"Apps missing Android IDs: {existing_df['android_app_id'].isna().sum()}")
    except FileNotFoundError:
        print("No existing output file found, starting fresh")
    
    # Read your CSV
    df = pd.read_csv(INPUT_CSV)

    # Use the first column (index 0)
    # The CSV appears to have app names in the first column
    app_names_series = df.iloc[:, 0]

    # Clean + deduplicate names
    app_names = (
        app_names_series
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    print(f"Found {len(app_names)} unique app names in column B.")
    
    # If we have existing data, only process apps missing Android IDs
    if existing_data:
        apps_to_process = []
        for app_name in app_names:
            if app_name not in existing_data or pd.isna(existing_data[app_name].get("android_app_id")):
                apps_to_process.append(app_name)
        print(f"Re-running Android lookup for {len(apps_to_process)} apps missing Android IDs...")
    else:
        apps_to_process = app_names
        print(f"Processing all {len(app_names)} app names...")

    rows = []

    for i, app_name in enumerate(apps_to_process, start=1):
        print(f"\n[{i}/{len(apps_to_process)}] Looking up: {app_name}")

        # If we have existing data, use it; otherwise fetch iOS ID
        if existing_data and app_name in existing_data:
            ios_id = existing_data[app_name].get("ios_app_id")
            if pd.isna(ios_id):
                ios_id = get_ios_app_id(app_name)
                print(f"  iOS app ID: {ios_id}")
            else:
                print(f"  iOS app ID: {ios_id} (from existing)")
        else:
            ios_id = get_ios_app_id(app_name)
            print(f"  iOS app ID: {ios_id}")

        # Always try to get Android ID (or re-try if missing)
        android_id = get_android_app_id(app_name)
        print(f"  Android package: {android_id}")

        rows.append({
            "app_name": app_name,
            "ios_app_id": ios_id,
            "android_app_id": android_id,
        })

        # Be nice to the APIs / avoid rate limits
        time.sleep(0.5)

    # Merge with existing data if it exists
    if existing_data:
        # Update existing data with new results
        for row in rows:
            existing_data[row["app_name"]] = {
                "ios_app_id": row["ios_app_id"],
                "android_app_id": row["android_app_id"]
            }
        
        # Rebuild full list from all app names
        all_rows = []
        for app_name in app_names:
            if app_name in existing_data:
                all_rows.append({
                    "app_name": app_name,
                    "ios_app_id": existing_data[app_name].get("ios_app_id"),
                    "android_app_id": existing_data[app_name].get("android_app_id")
                })
            else:
                # Shouldn't happen, but handle it
                all_rows.append({
                    "app_name": app_name,
                    "ios_app_id": None,
                    "android_app_id": None
                })
        out_df = pd.DataFrame(all_rows, columns=["app_name", "ios_app_id", "android_app_id"])
    else:
        out_df = pd.DataFrame(rows, columns=["app_name", "ios_app_id", "android_app_id"])

    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\nDone. Results written to: {OUTPUT_CSV}")
    print(f"Total apps: {len(out_df)}")
    print(f"Apps with iOS ID: {out_df['ios_app_id'].notna().sum()}")
    print(f"Apps with Android ID: {out_df['android_app_id'].notna().sum()}")
    print(f"Apps with both: {(out_df['ios_app_id'].notna() & out_df['android_app_id'].notna()).sum()}")

if __name__ == "__main__":
    main()
