"""
download_models.py - Download SkillSight trained models

This script downloads the pre-trained models from cloud storage.
Run this after cloning the repository to get the model files.

Usage:
    python download_models.py
    python download_models.py --model deberta
    python download_models.py --model roberta
    python download_models.py --model all
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve
from tqdm import tqdm

# ============================================
# CONFIGURATION - Update these URLs after uploading models
# ============================================

MODEL_URLS = {
    "deberta": {
        "url": "PLACEHOLDER_URL",  # TODO: Replace with actual URL after upload
        "filename": "deberta_v3_base.zip",
        "target_dir": "models/deberta_v3_base",
        "size_mb": 720,
    },
    "roberta": {
        "url": "PLACEHOLDER_URL",  # TODO: Replace with actual URL after upload
        "filename": "roberta_base.zip",
        "target_dir": "models/roberta_base",
        "size_mb": 480,
    },
}

# Hosting options (instructions below):
# 1. Google Drive (free, 15GB)
# 2. Hugging Face Hub (free, unlimited for models)
# 3. OneDrive (free, 5GB)
# 4. Dropbox (free, 2GB)

REPO_ROOT = Path(__file__).resolve().parent


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path) -> bool:
    """Download a file with progress bar."""
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
            urlretrieve(url, output_path, reporthook=t.update_to)
        return True
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return False


def extract_zip(zip_path: Path, target_dir: Path) -> bool:
    """Extract a zip file to target directory."""
    try:
        print(f"[INFO] Extracting to {target_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(target_dir.parent)
        zip_path.unlink()  # Delete zip after extraction
        print(f"[OK] Extracted successfully")
        return True
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        return False


def download_model(model_name: str) -> bool:
    """Download and extract a single model."""
    if model_name not in MODEL_URLS:
        print(f"[ERROR] Unknown model: {model_name}")
        return False
    
    config = MODEL_URLS[model_name]
    target_dir = REPO_ROOT / config["target_dir"]
    
    # Check if already exists
    if (target_dir / "model.safetensors").exists():
        print(f"[SKIP] {model_name} already exists at {target_dir}")
        return True
    
    # Check if URL is configured
    if config["url"] == "PLACEHOLDER_URL":
        print(f"\n[ERROR] Model URL not configured for {model_name}!")
        print(f"        Please update MODEL_URLS in download_models.py")
        print(f"\n        Or download manually from the shared link and extract to:")
        print(f"        {target_dir}")
        return False
    
    print(f"\n[INFO] Downloading {model_name} (~{config['size_mb']}MB)...")
    
    # Create temp directory
    temp_dir = REPO_ROOT / "temp_download"
    temp_dir.mkdir(exist_ok=True)
    zip_path = temp_dir / config["filename"]
    
    # Download
    if not download_file(config["url"], zip_path):
        return False
    
    # Extract
    target_dir.mkdir(parents=True, exist_ok=True)
    if not extract_zip(zip_path, target_dir):
        return False
    
    # Cleanup temp
    temp_dir.rmdir()
    
    print(f"[OK] {model_name} ready at {target_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download SkillSight models")
    parser.add_argument("--model", type=str, default="all", 
                       choices=["deberta", "roberta", "all"],
                       help="Which model to download (default: all)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("SkillSight Model Downloader")
    print("=" * 60)
    
    models_to_download = list(MODEL_URLS.keys()) if args.model == "all" else [args.model]
    
    success = True
    for model in models_to_download:
        if not download_model(model):
            success = False
    
    print("\n" + "=" * 60)
    if success:
        print("[OK] All models downloaded successfully!")
        print("\nYou can now run:")
        print('  python analyze_resume.py --text "your text" --model deberta')
    else:
        print("[WARN] Some models failed to download.")
        print("       See instructions above for manual download.")
    print("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
