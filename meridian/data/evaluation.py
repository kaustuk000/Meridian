# Portions of this file are adapted from MERU
# (Meta AI Research, https://github.com/facebookresearch/meru)
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Original source licensed under the license in LICENSES/MERU_LICENSE.
#
# Modified for Meridian:
#   - CocoCaptions      
#   - Flickr30kCaptions


"""
Hierarchy supervision
---------------------
  VisualGenomeDataset   — full images + region crops for hyperbolic tree loss

Retrieval evaluation
--------------------
  CocoCaptions          — MS-COCO 2017 Karpathy split, adapted from MERU
  Flickr30kCaptions     — Flickr30K Karpathy split, adapted from MERU

Smoke-test / local dev
----------------------
  ImageTextDataset      — flat JSON/TSV/CSV dataset, no TAR dependency
"""
from __future__ import annotations

import copy
import glob
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from meridian.data.transforms import (
    build_eval_transform,
    build_region_transform,
)

MAX_SEQ_LEN: int = 77  # CLIP text transformer context window

class VisualGenomeDataset(Dataset):
    """
    Visual Genome dataset for hyperbolic hierarchy supervision.

    Each sample is a (full-image, region-crop, region-phrase) triple.

    Why this structure?
    In Meridian's hyperbolic space, the full image is the *parent* node —
    general, closer to the Lorentz origin, wide entailment cone.  A region
    crop is a *child* node — specific, further from origin, narrow cone.
    ``spatial_norm(region) > spatial_norm(full_image)`` must hold after
    training.  This dataset provides the supervision pairs that enforce that.

    Important: two separate transforms are applied —
      ``image_transform``  (``build_train_transform``)  for the full image.
      ``region_transform`` (``build_region_transform``) for the region crop.
    Using ``build_train_transform`` on a region would apply a second
    RandomResizedCrop *on top of* the already-extracted bbox crop, which
    would destroy the exact content the hierarchy loss depends on.

    Expected directory layout::

        root/
          VG_100K/                 # images, part 1  (IDs 1–107,077)
          VG_100K_2/               # images, part 2  (IDs 107,078–108,077)
          region_descriptions.json # ~1.5 GB; downloaded from visualgenome.org

    Args:
        root:               VG root directory.
        split:              ``"train"`` | ``"val"``.  A deterministic 90/10
                            image-level split is applied (no official VG split).
        image_transform:    Transform for the full image.
                            Defaults to ``build_train_transform()`` / ``build_eval_transform()``.
        region_transform:   Transform for the region crop.
                            Defaults to ``build_region_transform()`` / ``build_eval_transform()``.
        min_region_size:    Minimum width AND height in pixels to accept a region.
                            Regions smaller than this produce uninformative crops.
        max_regions_per_image: Sample at most this many regions per image when
                            building the flat sample list.  Prevents images with
                            hundreds of regions from dominating the dataset.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_transform: Optional[Callable] = None,
        region_transform: Optional[Callable] = None,
        min_region_size: int = 32,
        max_regions_per_image: int = 5,
    ):
        super().__init__()
        self.root  = Path(root)
        self.split = split
        is_train   = split == "train"

        self.image_transform = image_transform or (
            build_train_transform() if is_train else build_eval_transform()
        )
        self.region_transform = region_transform or (
            build_region_transform() if is_train else build_eval_transform()
        )

        self.samples = self._build_samples(min_region_size, max_regions_per_image)

        n_images = len({s["image_id"] for s in self.samples})
        logger.info(
            f"VisualGenomeDataset [{split}]: {len(self.samples):,} region samples "
            f"from {n_images:,} images."
        )


    def _get_image_path(self, image_id: int) -> Path:
        """
        VG images are spread across two directories.  Try VG_100K first,
        then VG_100K_2 — mirrors the official distribution structure.
        """
        p1 = self.root / "VG_100K"   / f"{image_id}.jpg"
        p2 = self.root / "VG_100K_2" / f"{image_id}.jpg"
        return p1 if p1.exists() else p2

    def _in_split(self, image_id: int) -> bool:
        """
        Deterministic 90/10 train/val split by image ID.
        image_id % 10 == 0  → val (10 %)
        else                → train (90 %)
        Stable across runs; no randomness needed.
        """
        in_val = (image_id % 10) == 0
        return (self.split == "val") == in_val

    def _build_samples(
        self,
        min_size: int,
        max_per_image: int,
    ) -> list[dict]:
        """
        Flatten ``region_descriptions.json`` into a list of per-region records.

        Filters:
          1. Wrong split (image ID)
          2. Regions smaller than ``min_size`` in either dimension
          3. Empty or whitespace-only phrases
          4. More than ``max_per_image`` regions per image (random subsample)

        Each record is a lightweight dict — no images are loaded here.
        """
        json_path = self.root / "region_descriptions.json"
        logger.info(f"Loading VG region descriptions from {json_path} …")

        with json_path.open() as f:
            data: list[dict] = json.load(f)

        samples: list[dict] = []

        for entry in data:
            image_id = entry["id"]
            if not self._in_split(image_id):
                continue

            # Filter degenerate / empty regions.
            valid = []
            for r in entry.get("regions", []):
                # VG uses both "width"/"height" and "w"/"h" across versions.
                w = r.get("width",  r.get("w", 0))
                h = r.get("height", r.get("h", 0))
                phrase = r.get("phrase", "").strip()

                if w >= min_size and h >= min_size and phrase:
                    valid.append({
                        "image_id":  image_id,
                        "region_id": r["region_id"],
                        "phrase":    phrase,
                        "bbox":      (r["x"], r["y"], w, h),
                    })

            if not valid:
                continue

            # Subsample to cap per-image contribution.
            if len(valid) > max_per_image:
                valid = random.sample(valid, max_per_image)

            samples.extend(valid)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns::

            {
              "image_id"  : int,
              "region_id" : int,
              "image"     : FloatTensor (3, 224, 224),  # full image
              "region"    : FloatTensor (3, 224, 224),  # region crop
              "text"      : str,                         # region phrase
            }

        The trainer passes ``image`` + ``text`` to the standard contrastive
        loss, and ``image`` + ``region`` to the hierarchy loss that enforces
        ``spatial_norm(region) > spatial_norm(image)``.
        """
        record   = self.samples[idx]
        img_path = self._get_image_path(record["image_id"])

        full_img = Image.open(img_path).convert("RGB")
        iw, ih   = full_img.size

        # Extract region crop — clamp bbox to image bounds because VG
        # annotations occasionally overflow by a few pixels.
        x, y, w, h = record["bbox"]
        x1, y1 = max(0, x),     max(0, y)
        x2, y2 = min(iw, x + w), min(ih, y + h)
        region_crop = full_img.crop((x1, y1, x2, y2))

        return {
            "image_id":  record["image_id"],
            "region_id": record["region_id"],
            "image":     self.image_transform(full_img),      # (3, 224, 224)
            "region":    self.region_transform(region_crop),  # (3, 224, 224)
            "text":      record["phrase"],
        }

class CocoCaptions(Dataset):
    """
    MS-COCO Captions dataset for image-text retrieval evaluation.

    Returns 5 captions per image (Karpathy / official split).

    Expected layout::

        root/
          train2017/  val2017/  test2017/
          annotations/
            captions_train2017.json
            captions_val2017.json

    Args:
        root:      COCO root directory.
        split:     ``"train"`` | ``"val"`` | ``"test"``.
        transform: PIL → Tensor.  Defaults to ``build_eval_transform()``.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.root      = Path(root)
        self.split     = split
        self.transform = transform or build_eval_transform()

        json_path = self.root / "annotations" / f"captions_{split}2017.json"
        coco_json = json.load(open(json_path))

        image_id_to_anns: dict[int, list] = defaultdict(list)
        for ann in coco_json["annotations"]:
            image_id_to_anns[ann["image_id"]].append(ann)

        # Each sample: (image_id, image_path, list[caption_id], list[caption])
        self.samples = [
            (
                image_id,
                self.root / f"{split}2017" / f"{image_id:0>12d}.jpg",
                [ann["id"]      for ann in anns],
                [ann["caption"] for ann in anns],
            )
            for image_id, anns in image_id_to_anns.items()
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        image_id, image_path, caption_ids, captions = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        return {
            "image_id":    image_id,
            "caption_ids": caption_ids,
            "image":       self.transform(image),
            "captions":    captions,
        }


class Flickr30kCaptions(CocoCaptions):
    """
    Flickr30K Captions dataset for image-text retrieval evaluation.
    Uses the Karpathy split JSON.

    Download the Karpathy split from:
    https://cs.stanford.edu/people/karpathy/deepimagesent/

    Expected layout::

        root/
          flickr30k_images/   (~31K images)
          dataset_flickr30k.json

    Inherits ``__len__`` and ``__getitem__`` from ``CocoCaptions`` —
    ``self.samples`` is populated in the same (image_id, path, ids, captions)
    format so all downstream code works identically.

    Args:
        root:      Flickr30K root directory.
        split:     ``"train"`` | ``"val"`` | ``"test"``.
        transform: PIL → Tensor.  Defaults to ``build_eval_transform()``.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Optional[Callable] = None,
    ):
    
        Dataset.__init__(self)
        self.root      = Path(root)
        self.split     = split
        self.transform = transform or build_eval_transform()

        flickr_json = json.load(open(self.root / "dataset_flickr30k.json"))

        self.samples = [
            (
                int(ann["filename"][:-4]),
                self.root / "flickr30k_images" / ann["filename"],
                ann["sentids"],
                [entry["raw"] for entry in ann["sentences"]],
            )
            for ann in flickr_json["images"]
            if ann["split"] == split
        ]


class ImageTextDataset(Dataset):
    """
    Map-style dataset for local smoke tests.

    Reads a flat JSON / TSV / CSV metadata file and a directory of images.
    Use this when you want to verify the training loop on 100–1000 samples
    before committing to a full CC3M run on vast.ai.

    For production training use ``ImageTextWebDataset`` + ``CC3MTarMapper``.

    Args:
        metadata_path: Path to the metadata file (.json / .tsv / .csv).
        image_root:    Root directory for relative image paths.
        split:         ``"train"`` | ``"val"`` | ``"test"``.
        max_samples:   Cap the dataset size (handy for quick iteration).
    """

    def __init__(
        self,
        metadata_path: str | Path,
        image_root:    str | Path,
        split:         str = "train",
        max_samples:   Optional[int] = None,
    ) -> None:
        self.image_root = Path(image_root)
        # Use the new transforms API 
        self.transform  = (
            build_train_transform() if split == "train" else build_eval_transform()
        )
        self.samples: List[Dict[str, str]] = _load_metadata(
            Path(metadata_path), max_samples
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        record = self.samples[idx]
        image  = self._load_image(record["image"])
        return {
            "image": image,
            "text":  record["caption"]
        }

    def _load_image(self, rel_path: str) -> Tensor:
        img = Image.open(self.image_root / rel_path).convert("RGB")
        return self.transform(img)



# Module-level helper — shared by ImageTextDataset and build_metadata_db.py

def _load_metadata(
    path: Path,
    max_samples: Optional[int],
) -> List[Dict[str, str]]:
    """
    Parse a flat metadata file → list of ``{"image": ..., "caption": ...}`` dicts.

    Raises:
        FileNotFoundError: *path* does not exist.
        ValueError: Unsupported extension or missing JSON keys.
    """
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    if path.suffix == ".json":
        with path.open() as f:
            records: List[Dict[str, str]] = json.load(f)
        if records and not {"image", "caption"}.issubset(records[0]):
            raise ValueError(
                f"JSON records must have 'image' and 'caption' keys. "
                f"Got: {list(records[0].keys())}"
            )

    elif path.suffix in (".tsv", ".csv"):
        sep = "\t" if path.suffix == ".tsv" else ","
        records = []
        with path.open() as f:
            for line in f:
                parts = line.rstrip("\n").split(sep, maxsplit=1)
                if len(parts) < 2:
                    continue
                caption, image_name = parts[0].strip(), parts[1].strip()
                records.append({"image": image_name, "caption": caption})
    else:
        raise ValueError(
            f"Unsupported metadata format: '{path.suffix}'. "
            "Expected .json, .tsv, or .csv"
        )

    if max_samples is not None:
        records = records[:max_samples]
    return records


def retrieval_collate_fn(batch):
    """
    Custom collate function for CocoCaptions and Flickr30kCaptions to 
    prevent PyTorch from scrambling the list of 5 captions per image.
    """
    return {
        "image_id": [item["image_id"] for item in batch],
        "caption_ids": [item["caption_ids"] for item in batch],
        "captions": [item["captions"] for item in batch], # Keeps list of lists intact
        "image": torch.stack([item["image"] for item in batch], dim=0),
    }