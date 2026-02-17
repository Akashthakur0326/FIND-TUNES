import zipfile
import os
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from find_tunes.core.config import NOISE_DIR, ensure_directories

def seed_noise_dataset():
    zip_path = Path(r"C:\Users\Admin\Desktop\FIND TUNES\data\noise_data_16k (1).zip")
    
    if not zip_path.exists():
        logger.error(f"❌ Zip file not found at {zip_path}")
        return

    ensure_directories()
    logger.info(f"📦 Preparing to unzip 10GB dataset to: {NOISE_DIR}")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        files = zip_ref.namelist()
        # 🌟 PROGRESS BAR: Since it's 10GB, we need to see progress
        for file in tqdm(files, desc="Unzipping noise data", unit="file"):
            zip_ref.extract(file, NOISE_DIR)

    logger.success("✅ Extraction complete!")
    
    # 🌟 PRINT FILE STRUCTURE
    logger.info("📂 Final Noise Directory Structure:")
    print_directory_tree(NOISE_DIR)

def print_directory_tree(path: Path, indent=""):
    """Helper to visualize the resulting structure in terminal."""
    # Limit depth so 10GB of files doesn't flood the terminal
    items = list(path.iterdir())
    for i, item in enumerate(items[:20]): # Show first 20 items only
        if item.is_dir():
            print(f"{indent}└── 📁 {item.name}/")
            # Recurse one level down if it's a small subdir
            # print_directory_tree(item, indent + "    ")
        else:
            print(f"{indent}├── 📄 {item.name}")
    
    if len(items) > 20:
        print(f"{indent}... and {len(items) - 20} more files.")

if __name__ == "__main__":
    seed_noise_dataset()