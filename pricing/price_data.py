"""
Price database built from scraped Love Luxury listings.
CSV columns: model, size, leather, color, hardware, condition, price_eur
"""

import pandas as pd
import os

PRICE_DB_PATH = os.path.join(os.path.dirname(__file__), "price_database.csv")

# Condition multipliers derived from actual data (baseline: Box Fresh)
CONDITION_MULTIPLIERS = {
    "Box Fresh":  1.00,
    "Unworn":     1.00,
    "Excellent":  1.15,
    "Pre-Loved":  0.74,
    "Good":       0.65,
    "Fair":       0.55,
}

# Hardware multipliers - only Rose Gold has meaningful difference in data
# Gold vs Palladium are identical (both 1.00x baseline)
HARDWARE_MULTIPLIERS = {
    "Rose Gold":     1.40,  # n=3, use with caution
    "Brushed Gold":  1.02,  # n=3, effectively neutral
    "Gold":          1.00,  # n=37, baseline
    "Palladium":     1.00,  # n=40, baseline
    "Permabrass":    0.92,  # n=3, slightly lower
    "Ruthenium":     0.95,  # estimated, no data
}


def load_price_db():
    if not os.path.exists(PRICE_DB_PATH):
        raise FileNotFoundError(
            f"Price database not found at {PRICE_DB_PATH}. "
            "Run price_fetcher.py first to generate it."
        )
    df = pd.read_csv(PRICE_DB_PATH)
    df.columns = df.columns.str.strip().str.lower()
    return df


def estimate_price(model, size, leather, color, hardware=None, condition=None):
    """
    Returns a price range based on comparable listings.

    Strategy:
    1. Look up comparable listings (progressively relaxing filters)
    2. Apply hardware multiplier if exact match not available
    3. Apply condition multiplier on top
    """
    df = load_price_db()
    matches = df.copy()

    # Step 1: Lookup – progressively relax filters
    matches = matches[matches["model"].str.lower() == model.lower()]
    if len(matches) == 0:
        return {"error": f"No listings found for model '{model}'", "comparables": 0}

    for col, val in [("size", str(size)), ("leather", leather), ("color", color)]:
        subset = matches[matches[col].astype(str).str.lower() == val.lower()]
        if len(subset) >= 3:
            matches = subset

    if len(matches) == 0:
        return {"error": "No comparable listings found", "comparables": 0}

    # Step 2: Hardware multiplier
    hw_multiplier = 1.0
    hw_note = None
    if hardware:
        exact_hw = matches[matches["hardware"].str.lower() == hardware.lower()]
        if len(exact_hw) >= 3:
            matches = exact_hw
        else:
            dominant_hw = matches["hardware"].mode()[0] if not matches.empty else None
            if dominant_hw and dominant_hw.lower() != hardware.lower():
                hw_multiplier = HARDWARE_MULTIPLIERS.get(hardware, 1.0) / HARDWARE_MULTIPLIERS.get(dominant_hw, 1.0)
                hw_note = f"Hardware adjusted: {dominant_hw} → {hardware} (×{hw_multiplier:.2f})"

    # Step 3: Condition multiplier
    cond_multiplier = 1.0
    cond_note = None
    if condition:
        exact_cond = matches[matches["condition"].str.lower() == condition.lower()]
        if len(exact_cond) >= 3:
            matches = exact_cond
        else:
            dominant_cond = matches["condition"].mode()[0] if not matches.empty else None
            if dominant_cond and dominant_cond.lower() != condition.lower():
                cond_multiplier = CONDITION_MULTIPLIERS.get(condition, 1.0) / CONDITION_MULTIPLIERS.get(dominant_cond, 1.0)
                cond_note = f"Condition adjusted: {dominant_cond} → {condition} (×{cond_multiplier:.2f})"

    # Step 4: Apply multipliers
    total = hw_multiplier * cond_multiplier
    prices = matches["price_eur"] * total

    return {
        "low":         int(prices.quantile(0.25)),
        "mid":         int(prices.median()),
        "high":        int(prices.quantile(0.75)),
        "mean":        int(prices.mean()),
        "comparables": len(matches),
        "multiplier":  round(total, 3),
        "notes":       [n for n in [hw_note, cond_note] if n],
    }