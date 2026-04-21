"""
price_fetcher.py
Fetches prices from Love Luxury for all images in a folder.
Output: birkin_prices.csv with two columns: filename | price_gbp

SETUP:   pip install selenium webdriver-manager pandas
USAGE:   1. Set IMAGE_FOLDER below  2. python price_fetcher.py
"""

import os, re, time
import pandas as pd
from pathlib import Path

IMAGE_FOLDER = os.path.expanduser("~/Downloads/birkin_images")  # ← CHANGE THIS
OUTPUT_CSV   = os.path.join(IMAGE_FOLDER, "birkin_prices.csv")
BASE_URL     = "https://loveluxury.co.uk/shop/"
DELAY        = 1.5


def filename_to_slug(filename):
    name = Path(filename).stem.lower()
    name = re.sub(r'-\d+x\d+$', '', name)
    name = re.sub(r'-updated-\d+$', '', name)
    name = re.sub(r'-updated$', '', name)
    name = re.sub(r'-pre-loved.*$', '', name)
    name = re.sub(r'-\d{4}(?:-\d+)*$', '', name)
    name = re.sub(r'-\d+$', '', name)
    return name


def fmt(price):
    return f"£{price:,.0f}" if price is not None else "NOT FOUND"


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

        if slug in cache:
            price = cache[slug]
            print(f"[{i}/{len(images)}] {filename[:60]} → {fmt(price)} (cached)")
        else:
            url = BASE_URL + slug + "/"
            try:
                driver.get(url)
                time.sleep(DELAY)
                els = driver.find_elements(By.CSS_SELECTOR, ".price .woocommerce-Price-amount")
                price = float(re.sub(r'[£,]', '', els[0].text.strip())) if els else None
            except Exception as e:
                print(f"  ERROR: {e}")
                price = None
            cache[slug] = price
            print(f"[{i}/{len(images)}] {filename[:60]} → {fmt(price)}")

        results.append({"filename": filename, "price_gbp": price})

    driver.quit()

    df = pd.DataFrame(results)[["filename", "price_gbp"]]
    df.to_csv(OUTPUT_CSV, index=False)

    found = df["price_gbp"].notna().sum()
    not_found = df["price_gbp"].isna().sum()
    print(f"\n✅ Done! {found}/{len(df)} prices found")
    print(f"❌ Not found: {not_found}")
    print(f"📁 Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()