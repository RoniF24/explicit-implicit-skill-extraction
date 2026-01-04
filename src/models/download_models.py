"""
download_models.py - Download SkillSight trained models from Hugging Face

This script downloads the pre-trained models from Hugging Face Hub.
Run this after cloning the repository to get the model files.

Usage:
    python download_models.py
    python download_models.py --model deberta
    python download_models.py --model roberta
    python download_models.py --model all
"""

import argparse
import sys
from pathlib import Path

# ============================================
# CONFIGURATION - Hugging Face Model Repositories
# ============================================

MODELS = {
    "deberta": {
        "repo_id": "YonatanEl/skillsight-deberta-v3",
        "target_dir": "src/models/trained_models/deberta_v3_base",
        "size_mb": 740,
    },
    "roberta": {
        "repo_id": "YonatanEl/skillsight-roberta-base",
        "target_dir": "src/models/trained_models/roberta_base",
        "size_mb": 500,
    },
    "deberta-onepass": {
        "repo_id": "YonatanEl/skillsight-deberta-v3-onepass",
        "target_dir": "src/models/trained_models/deberta_v3_onepass",
        "size_mb": 740,
    },
}

REPO_ROOT = Path(__file__).resolve().parents[2]  # src/models -> src -> SkillSight


def download_model(model_name: str) -> bool:
    """Download a model from Hugging Face Hub."""
    if model_name not in MODELS:
        print(f"[ERROR] Unknown model: {model_name}")
        return False
    
    config = MODELS[model_name]
    target_dir = REPO_ROOT / config["target_dir"]
    
    # Check if already exists
    if (target_dir / "model.safetensors").exists():
        print(f"[SKIP] {model_name} already exists at {target_dir}")
        return True
    
    print(f"\n[INFO] Downloading {model_name} (~{config['size_mb']}MB)...")
    print(f"       From: https://huggingface.co/{config['repo_id']}")
    
    try:
        from huggingface_hub import snapshot_download
        
        snapshot_download(
            repo_id=config["repo_id"],
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )
        
        print(f"[OK] {model_name} ready at {target_dir}")
        return True
        
    except ImportError:
        print("[ERROR] huggingface_hub not installed!")
        print("        Run: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download SkillSight models from Hugging Face")
    parser.add_argument("--model", type=str, default="all", 
                       choices=["deberta", "roberta", "deberta-onepass", "all"],
                       help="Which model to download (default: all)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("SkillSight Model Downloader (Hugging Face)")
    print("=" * 60)
    
    models_to_download = list(MODELS.keys()) if args.model == "all" else [args.model]
    
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
        print("       Check your internet connection and try again.")
    print("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
