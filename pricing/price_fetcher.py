"""
price_fetcher.py
Fetches price + condition from Love Luxury and extracts all attributes from filename.
All attribute lists are derived directly from the actual filenames in the dataset.

Output: price_database.csv with columns:
    filename, model, size, leather, color, hardware, condition, price_eur

SETUP:   pip install selenium webdriver-manager pandas
USAGE:   1. Set IMAGE_FOLDER below  2. python price_fetcher.py
"""

import os, re, time
import pandas as pd
from pathlib import Path

IMAGE_FOLDER = os.path.expanduser("~/Downloads/birkin_images")  # ← CHANGE THIS
OUTPUT_CSV   = os.path.join(IMAGE_FOLDER, "price_database.csv")
BASE_URL     = "https://loveluxury.co.uk/shop/"
DELAY        = 1.5
GBP_TO_EUR   = 1.17

# ── All values derived directly from actual filenames ─────────────────────────

MODELS = ["Micro Birkin", "Birkin Shoulder", "HAC", "Birkin"]
# Order matters: check longer/specific names first

SIZES = ["20", "25", "29", "30", "35", "39", "40"]

# Ordered longest-match first to avoid partial matches
LEATHERS = [
    "Evercolour", "Alligator", "Crocodile", "Niloticus",
    "Clemence", "Volupto", "Gulliver", "Madame", "Ardennes",
    "Barenia", "Lizard", "Novilo", "Chevre", "Ostrich", "Epsom",
    "Swift", "Togo", "Toile", "Fjord",
]

# Ordered longest/most-specific first
COLORS = [
    "Mauve Sylvester", "Bleu Celeste", "Bleu Zanzibar", "Bleu Tempete", "Bleu Marine",
    "Bleu Royal", "Bleu Jean", "Bleu Saphir", "Bleu Glacier",
    "Bleu Agate", "Gris Etain", "Gris Pantin", "Gris Perle", "Gris Plomb",
    "Rose Scheherazade", "Rose Azalee", "Rose Tyrien", "Rose Fuchsia",
    "Rose Mexico", "Rose Sakura", "Rouge Radieux", "Rouge De Coeur",
    "Rouge H", "Vert Amande", "Vert Anglais", "Vert Criquet",
    "Vert Mangrove", "Vert Verone", "Vert Vertigo", "Vert Deau",
    "Vert Yucca", "Orange H", "Candy Kiwi", "Blue De France",
    "Bougainvillea", "Sunset Rainbo", "Himalaya", "Poussiere",
    "Biscuit", "Cabane", "Caramel", "Graphite", "Fuchsia", "Geranium",
    "Anemone", "Bordeaux", "Brique", "Ebene", "Etoupe", "Ghillies",
    "Havane", "Mushroom", "Nata", "Noir", "Plomb", "Raisin",
    "Rouge", "Rubis", "Safran", "Soleil", "Tosca", "Trench",
    "Vanille", "Volynka", "Argile", "Craie", "Blanc", "White",
    "Lime", "Kiwi", "Gold", "Blue", "Red",
]

# Hardware - check compound first
HARDWARE_PATTERNS = [
    ("Rose Gold",    r"rose[- ]gold"),
    ("Brushed Gold", r"brushed[- ]gold"),
    ("Permabrass",   r"permabrass"),
    ("Palladium",    r"palladium"),
    ("Ruthenium",    r"ruthenium"),
    ("Gold",         r"gold[- ]hardware"),
]

CONDITIONS = [
    "Box Fresh", "Pre-Loved", "Preloved", "Excellent",
    "Very Good", "Good", "Fair", "Unworn", "Like New",
]

# ── Special slug fixes for known typos/oddities in filenames ─────────────────
# Maps a filename substring → correct URL slug override
SLUG_OVERRIDES = {
    "bikrin":     lambda s: s.replace("bikrin", "birkin"),
    "bariena":    lambda s: s.replace("bariena", "barenia"),
    "otsrich":    lambda s: s.replace("otsrich", "ostrich"),
    "fushsia":    lambda s: s.replace("fushsia", "fuchsia"),
    "missipian":  lambda s: s.replace("missipian", "missippian"),
    "retoure":    lambda s: s.replace("retoure", "retourne"),
    "novilo":     lambda s: s.replace("novilo", "novillo"),
    "fridatogo":  lambda s: s.replace("fridatogo", "frida-togo"),
}

# ── Helper functions ──────────────────────────────────────────────────────────

def filename_to_slug(filename):
    name = Path(filename).stem.lower()
    name = re.sub(r'-\d+x\d+$', '', name)
    name = re.sub(r'-updated-color.*$', '', name)
    name = re.sub(r'-updated-\d+$', '', name)
    name = re.sub(r'-updated$', '', name)
    name = re.sub(r'-new.*$', '', name)
    name = re.sub(r'-pre-loved.*$', '', name)
    name = re.sub(r'-preloved.*$', '', name)
    name = re.sub(r'-\d{4}-\d{4}.*$', '', name)
    name = re.sub(r'-\d{4}(?:-\d+)*$', '', name)
    name = re.sub(r'-\d+$', '', name)
    # Apply typo fixes
    for typo, fix in SLUG_OVERRIDES.items():
        if typo in name:
            name = fix(name)
    return name


def extract_model(filename):
    for model in MODELS:
        pattern = model.replace(' ', '[- ]')
        if re.search(pattern, filename, re.IGNORECASE):
            return model
    return "Birkin"


def extract_size(filename):
    m = re.search(r'(?:Birkin|Bikrin|Shoulder|HAC|Micro)[- ](\d+)', filename, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def extract_leather(filename):
    for leather in LEATHERS:
        if re.search(leather, filename, re.IGNORECASE):
            return leather
    return None


def extract_color(filename):
    for color in COLORS:
        pattern = color.replace(' ', '[- ]')
        if re.search(pattern, filename, re.IGNORECASE):
            return color
    return None


def extract_hardware(filename):
    for hw_name, pattern in HARDWARE_PATTERNS:
        if re.search(pattern, filename, re.IGNORECASE):
            return hw_name
    return None


def extract_condition_from_text(text):
    for cond in CONDITIONS:
        if cond.lower() in text.lower():
            return "Pre-Loved" if cond.lower() == "preloved" else cond
    return None


def fmt(price):
    return f"£{price:,.0f}" if price is not None else "NOT FOUND"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    exts = {'.jpg', '.jpeg', '.png', '.webp'}
    images = sorted([f for f in os.listdir(IMAGE_FOLDER)
                     if Path(f).suffix.lower() in exts])

    if not images:
        print(f"❌ No images found in: {IMAGE_FOLDER}")
        return

    print(f"✅ {len(images)} images found")
    print("🌐 Starting browser...")

    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.chrome.service import Service

    options = webdriver.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=options)

    results = []
    cache = {}

    for i, filename in enumerate(images, 1):
        slug = filename_to_slug(filename)

        model    = extract_model(filename)
        size     = extract_size(filename)
        leather  = extract_leather(filename)
        color    = extract_color(filename)
        hardware = extract_hardware(filename)

        if slug in cache:
            price_gbp = cache[slug]["price"]
            condition = cache[slug]["condition"]
            print(f"[{i}/{len(images)}] {filename[:50]} → {fmt(price_gbp)} | {condition or 'N/A'} (cached)")
        else:
            url = BASE_URL + slug + "/"
            try:
                driver.get(url)
                time.sleep(DELAY)
                price_els = driver.find_elements(By.CSS_SELECTOR, ".price .woocommerce-Price-amount")
                price_gbp = float(re.sub(r'[£,]', '', price_els[0].text.strip())) if price_els else None
                desc_els  = driver.find_elements(By.CSS_SELECTOR, ".woocommerce-product-details__short-description")
                condition = extract_condition_from_text(desc_els[0].text) if desc_els else None
            except Exception as e:
                print(f"  ERROR: {e}")
                price_gbp = None
                condition = None
            cache[slug] = {"price": price_gbp, "condition": condition}
            print(f"[{i}/{len(images)}] {filename[:50]} → {fmt(price_gbp)} | {condition or 'N/A'}")

        price_eur = round(price_gbp * GBP_TO_EUR) if price_gbp else None

        results.append({
            "filename": filename,
            "model":    model,
            "size":     size,
            "leather":  leather,
            "color":    color,
            "hardware": hardware,
            "condition": condition,
            "price_eur": price_eur,
        })

    driver.quit()

    df = pd.DataFrame(results)[[
        "filename", "model", "size", "leather",
        "color", "hardware", "condition", "price_eur"
    ]]
    df.to_csv(OUTPUT_CSV, index=False)

    found = df["price_eur"].notna().sum()
    print(f"\n✅ Done! {found}/{len(df)} prices found")
    print(f"📁 Saved: {OUTPUT_CSV}")
    print(f"\nNull values per column:")
    print(df.isnull().sum().to_string())
    print(f"\nSample:")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()