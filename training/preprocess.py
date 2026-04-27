"""
Preprocessing: background removal with rembg (BiRefNet) + train/val/test split.
Only processes NEW images not already in the processed cache.
"""
import os
import shutil
from pathlib import Path
from PIL import Image
from rembg import remove, new_session
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import config

# Persistent cache of processed images (survives re-runs)
CACHE_DIR = os.path.join(config.PROCESSED_DATA_DIR, "_cache")


def remove_background(input_path, output_path, session) -> bool:
    try:
        img = Image.open(input_path).convert("RGBA")
        result = remove(img, session=session)
        white_bg = Image.new("RGBA", result.size, config.BACKGROUND_COLOR + (255,))
        white_bg.paste(result, mask=result.split()[3])
        final = white_bg.convert("RGB")
        final = final.resize((config.IMG_SIZE, config.IMG_SIZE), Image.LANCZOS)
        final.save(output_path, "JPEG", quality=95)
        return True
    except Exception as e:
        print(f"  [SKIP] {input_path}: {e}")
        return False


def process_class(class_name, raw_dir, session):
    input_dir = os.path.join(raw_dir, class_name)
    cache_class_dir = os.path.join(CACHE_DIR, class_name)
    os.makedirs(cache_class_dir, exist_ok=True)

    valid_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = [f for f in os.listdir(input_dir) if Path(f).suffix.lower() in valid_ext]

    if not image_files:
        print(f"  [WARNING] No images in {input_dir}")
        return

    # Check which images are already processed
    already_done = set(os.listdir(cache_class_dir))
    new_files = [f for f in image_files if (Path(f).stem + ".jpg") not in already_done]
    skipped = len(image_files) - len(new_files)

    print(f"\n'{class_name}': {len(image_files)} total | {skipped} cached | {len(new_files)} new to process")

    if not new_files:
        print(f"  All images already processed, skipping!")
        return

    for fname in tqdm(new_files, desc=f"  {class_name}"):
        in_path = os.path.join(input_dir, fname)
        out_fname = Path(fname).stem + ".jpg"
        out_path = os.path.join(cache_class_dir, out_fname)
        remove_background(in_path, out_path, session)

    total = len(os.listdir(cache_class_dir))
    print(f"  Done! Cache now has {total} images for '{class_name}'")


def split_and_organise():
    print("\n--- Splitting into train / val / test ---")

    # Clear old splits but keep cache
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(config.PROCESSED_DATA_DIR, split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)

    for class_name in config.CLASS_NAMES:
        cache_class_dir = os.path.join(CACHE_DIR, class_name)
        if not os.path.exists(cache_class_dir):
            continue
        images = sorted(os.listdir(cache_class_dir))
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
                shutil.copy2(os.path.join(cache_class_dir, fname), os.path.join(dest, fname))


def main():
    config.seed_everything()
    print("=" * 60)
    print("  Hermes Authenticator - Preprocessing")
    print("=" * 60)

    for cn in config.CLASS_NAMES:
        p = os.path.join(config.RAW_DATA_DIR, cn)
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Expected '{p}'. Check RAW_DATA_DIR in config.py")

    session = new_session("birefnet-general")
    for cn in config.CLASS_NAMES:
        process_class(cn, config.RAW_DATA_DIR, session)

    split_and_organise()

    print("\n Done! Data saved to:", config.PROCESSED_DATA_DIR)
    print(" Cache kept at:", CACHE_DIR)


if __name__ == "__main__":
    main()
