import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints" / "clip"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# Redirect HF cache into project before any transformers import
os.environ["HF_HOME"] = str(PROJECT_ROOT / "checkpoints" / "hf_cache")

from transformers import CLIPModel, CLIPProcessor

# HuggingFace save_pretrained writes a directory, not a single file
DST = CHECKPOINTS_DIR / "hf_vitb16_openai"

if DST.exists() and (DST / "config.json").exists():
    print(f"Already downloaded: {DST}")
    print("Delete the folder manually if you want to re-download.")
else:
    print("Downloading openai/clip-vit-base-patch16 from HuggingFace ...")
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    model.save_pretrained(DST)
    processor.save_pretrained(DST)
    print(f"Saved to {DST}")

print("Done.")