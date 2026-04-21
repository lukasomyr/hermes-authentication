"""
Preprocessing: background removal with rembg + train/val/test split.
"""

import os
import shutil
from pathlib import Path

from PIL import Image
from rembg import remove
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config


def remove_background(input_path: str, output_path: str) -> bool:
    try:
        img = Image.open(input_path).convert("RGBA")
        result = remove(img)
        white_bg = Image.new("RGBA", result.size, config.BACKGROUND_COLOR + (255,))
        white_bg.paste(result, mask=result.split()[3])
        final = white_bg.convert("RGB")
        final = final.resize((config.IMG_SIZE, config.IMG_SIZE), Image.LANCZOS)
        final.save(output_path, "JPEG", quality=95)
        return True
    except Exception as e:
        print(f"  [SKIP] {input_path}: {e}")
        return False


def process_class(class_name, raw_dir, temp_dir):
    input_dir = os.path.join(raw_dir, class_name)
    output_dir = os.path.join(temp_dir, class_name)
    os.makedirs(output_dir, exist_ok=True)

    valid_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = [f for f in os.listdir(input_dir) if Path(f).suffix.lower() in valid_ext]

    if not image_files:
        print(f"  [WARNING] No images in {input_dir}")
        return []

    processed = []
    print(f"\nProcessing '{class_name}' ({len(image_files)} images)...")
    for fname in tqdm(image_files, desc=f"  {class_name}"):
        out_fname = Path(fname).stem + ".jpg"
        out_path = os.path.join(output_dir, out_fname)
        if remove_background(os.path.join(input_dir, fname), out_path):
            processed.append(out_path)

    print(f"  Done: {len(processed)} / {len(image_files)}")
    return processed


def split_and_organise(temp_dir):
    print("\n--- Splitting into train / val / test ---")
    for class_name in config.CLASS_NAMES:
        class_dir = os.path.join(temp_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        images = sorted(os.listdir(class_dir))
        print(f"  '{class_name}': {len(images)} images")

        train_imgs, val_test_imgs = train_test_split(
            images, test_size=(config.VAL_RATIO + config.TEST_RATIO), random_state=config.SEED)
        relative_test = config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
        val_imgs, test_imgs = train_test_split(
            val_test_imgs, test_size=relative_test, random_state=config.SEED)

        print(f"    Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")

        for split_name, file_list in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            dest = os.path.join(config.PROCESSED_DATA_DIR, split_name, class_name)
            os.makedirs(dest, exist_ok=True)
            for fname in file_list:
                shutil.copy2(os.path.join(class_dir, fname), os.path.join(dest, fname))


def main():
    config.seed_everything()
    print("=" * 60)
    print("  Hermes Authenticator - Preprocessing")
    print("=" * 60)

    for cn in config.CLASS_NAMES:
        p = os.path.join(config.RAW_DATA_DIR, cn)
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Expected '{p}'. Check RAW_DATA_DIR in config.py")

    temp_dir = os.path.join(config.PROCESSED_DATA_DIR, "_temp")
    os.makedirs(temp_dir, exist_ok=True)

    for cn in config.CLASS_NAMES:
        process_class(cn, config.RAW_DATA_DIR, temp_dir)

    split_and_organise(temp_dir)
    shutil.rmtree(temp_dir)
    print("\n Done! Data saved to:", config.PROCESSED_DATA_DIR)

if __name__ == "__main__":
    main()
