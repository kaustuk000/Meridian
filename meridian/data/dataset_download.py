import argparse
import subprocess
from pathlib import Path


DATA_ROOT = Path("data")


def run(cmd):
    print("\n>>>", cmd)
    subprocess.run(cmd, shell=True, check=True)


# ============================================================
# CC3M
# ============================================================

def download_cc3m():
    root = DATA_ROOT / "cc3m"
    root.mkdir(parents=True, exist_ok=True)

    print(
        "\nPlace cc3m.tsv inside:\n"
        f"  {root}\n"
        "before running."
    )

    cmd = f"""
img2dataset ^
  --url_list "{root / 'cc3m.tsv'}" ^
  --input_format tsv ^
  --url_col url ^
  --caption_col caption ^
  --output_format webdataset ^
  --output_folder "{root / 'webdataset'}" ^
  --processes_count 16 ^
  --thread_count 64 ^
  --image_size 256
"""
    print(cmd)


# ============================================================
# COCO
# ============================================================

def download_coco():
    root = DATA_ROOT / "coco"
    root.mkdir(parents=True, exist_ok=True)

    print(
        """
Download manually:

Images:
https://cocodataset.org/#download

Required:
  train2017.zip
  val2017.zip

Annotations:
  annotations_trainval2017.zip

Expected layout:

data/coco/
├── train2017/
├── val2017/
└── annotations/
"""
    )


# ============================================================
# Flickr30K
# ============================================================

def download_flickr30k():
    root = DATA_ROOT / "flickr30k"
    root.mkdir(parents=True, exist_ok=True)

    print(
        """
Download:

Images:
https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset

Karpathy split:
https://cs.stanford.edu/people/karpathy/deepimagesent/

Expected:

data/flickr30k/
├── flickr30k_images/
└── dataset_flickr30k.json
"""
    )


# ============================================================
# Visual Genome
# ============================================================

def download_visual_genome():
    root = DATA_ROOT / "visual_genome"
    root.mkdir(parents=True, exist_ok=True)

    print(
        """
Download from:

https://visualgenome.org/api/v0/api_home.html

Required:

images.zip
images2.zip
region_descriptions.json

Expected:

data/visual_genome/
├── VG_100K/
├── VG_100K_2/
└── region_descriptions.json
"""
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["cc3m", "coco", "flickr30k", "vg", "all"],
        required=True,
    )

    args = parser.parse_args()

    if args.dataset in ["cc3m", "all"]:
        download_cc3m()

    if args.dataset in ["coco", "all"]:
        download_coco()

    if args.dataset in ["flickr30k", "all"]:
        download_flickr30k()

    if args.dataset in ["vg", "all"]:
        download_visual_genome()


if __name__ == "__main__":
    main()