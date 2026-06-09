# Portions of this file are adapted from MERU
# (Meta AI Research, https://github.com/facebookresearch/meru)
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Original source licensed under the license in LICENSES/MERU_LICENSE.
#
# Modified for Meridian:
#   - ImageTextWebDataset: replaced meru.utils.distributed with inline helpers;
#                          changed buffer_size threshold from >1 to >0


"""
meridian/data/dataset.py
========================
All dataset classes for Meridian, organised by role:

Training (streaming, webdataset TAR shards)
-------------------------------------------
  ImageTextWebDataset   — generic TAR shard streamer, adapted from MERU
  CC3MTarMapper         — per-sample mapper for CC3M img2dataset output

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
import webdataset as wds
from loguru import logger
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset

import functools

from meridian.data.transforms import (
    build_train_transform,
)
from meridian.tokenizer import Tokenizer

MAX_SEQ_LEN: int = 77  # CLIP text transformer context window
# At module level — CLIP vocab always ends with these two special tokens
_SOT_TOKEN: int = 49406
_EOT_TOKEN: int = 49407

# ---------------------------------------------------------------------------
# Distributed helpers — inline to avoid a separate utils module.
# Return safe defaults when running outside a dist process group (single GPU,
# smoke tests, local dev).
# ---------------------------------------------------------------------------

def _rank() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def _world_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


# ===========================================================================
# CC3M training — webdataset streaming
# ===========================================================================

class ImageTextWebDataset(IterableDataset):
    """
    Streaming IterableDataset over webdataset TAR shards.

    Used for CC3M training on vast.ai.  TAR files are produced by img2dataset
    (``--output_format webdataset``).  Each shard is read sequentially and
    samples are shuffled via a rolling buffer to approximate i.i.d. sampling
    without loading the whole dataset into RAM.

    Adapted from MERU (Meta AI Research, Apache-2.0).  Changes vs MERU:
      - ``meru.utils.distributed`` replaced with inline torch.distributed helpers
      - ``wordsegment`` dependency removed (not needed for CC3M)

    Multi-GPU behaviour
    -------------------
    TAR files are sharded across GPU ranks automatically.  Rank 0 gets shards
    [0, world_size, 2*world_size, …], rank 1 gets [1, world_size+1, …] etc.
    Each GPU therefore sees a disjoint subset of the data — no duplicates.

    Args:
        tarfiles:        Path, glob pattern, or list of either pointing at TAR shards.
        mapper:          Callable applied to each decoded webdataset dict.
                         Should return ``{"image": Tensor, "text": str}``.
        buffer_size:     Shuffle buffer depth.  0 disables shuffling.
        infinite_stream: Loop over all shards forever (True for training).
                         Yield one epoch and stop (False for one-pass eval).
        seed:            RNG seed for deterministic shuffling.
    """

    def __init__(
        self,
        tarfiles: str | list[str],
        mapper: Callable,
        buffer_size: int = 5000,
        infinite_stream: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        self.mapper = mapper
        self.buffer_size = buffer_size
        self.infinite_stream = infinite_stream
        self.seed = seed

        if isinstance(tarfiles, str):
            tarfiles = [tarfiles]

        # Expand all glob patterns into individual TAR paths.
        self.tarfiles: list[str] = []
        for _path in tarfiles:
            for _glob in _path.split():
                self.tarfiles.extend(glob.glob(_glob))

        self.tarfiles = sorted(self.tarfiles)
        logger.info(f"{self.__class__.__name__} found {len(self.tarfiles)} TARs.")

        # Shard across GPU ranks so each process sees unique data.
        rank, ws = _rank(), _world_size()
        self.tarfiles = self.tarfiles[rank::ws]
        logger.info(f"Rank {rank}/{ws} will load {len(self.tarfiles)} TARs.")

    def __iter__(self):
        rng = random.Random(self.seed)

        pipeline = wds.DataPipeline(
            wds.SimpleShardList(self.tarfiles, seed=self.seed),
            wds.split_by_worker,
            wds.tarfile_to_samples(),
        )

        if self.buffer_size > 0:
            pipeline.append(
                wds.shuffle(self.buffer_size, initial=self.buffer_size, rng=rng)
            )

        # Decode .jpg keys to PIL.Image before passing to the mapper.
        pipeline.append(wds.decode("pil", handler=wds.warn_and_continue))
        pipeline.append(wds.map(self.mapper))

        if self.infinite_stream:
            # Training loop controls when to stop — yield indefinitely.
            while True:
                yield from copy.deepcopy(pipeline)
        else:
            yield from pipeline


class CC3MTarMapper:
    """
    Per-sample mapper for CC3M webdataset TAR shards.

    img2dataset (``--output_format webdataset``) stores each sample as:
      ``{key}.jpg``   — image
      ``{key}.txt``   — caption (single line)
      ``{key}.json``  — metadata dict (original URL, download status, …)

    After ``wds.decode("pil")`` the webdataset dict has:
      ``"jpg"``   : PIL.Image (already decoded)
      ``"txt"``   : str
      ``"json"``  : dict

    The ``"__key__"`` field is the zero-padded original TSV line number, which
    is what ``build_metadata_db.py`` uses as the SQLite ``idx`` — so the mapping
    between the index and the caption is never broken by dead URLs.

    Returns ``{"__key__": str, "image": FloatTensor (3,224,224), "text": str}``.

    Why no tokenization here?
    Raw text is returned so the trainer's collate function can batch-tokenize
    efficiently.  Tokenizing per-sample inside a worker process is slower than
    a single batched call on the main process.

    Args:
        image_transform: PIL.Image → FloatTensor pipeline.
                         Defaults to ``build_train_transform()``.
    """

    def __init__(self, image_transform: Optional[Callable] = None):
        self.image_transform = image_transform or build_train_transform()

    def __call__(self, sample: dict) -> dict:
        # Caption lives in .txt; fall back to metadata JSON if .txt is empty.
        caption = (sample.get("txt") or "").strip()
        if not caption:
            caption = sample.get("json", {}).get("caption", "")

        return {
            "__key__": sample["__key__"],
            "image":   self.image_transform(sample["jpg"]),
            "text":    caption,
        }


def _collate_cc3m(
    batch: list[dict],
    tokenizer: Tokenizer,
) -> dict[str, Tensor]:
    """
    Collate CC3M samples into a batch for MeridianModel.forward().

    The Meridian Tokenizer returns list[IntTensor] with variable lengths and
    no padding — this function handles truncation, padding, mask construction,
    and EOS index extraction.

    Returns
    -------
    pixel_values   : (B, 3, 224, 224)  float32
    input_ids      : (B, 77)           long
    attention_mask : (B, 77)           long   — 1 for real tokens, 0 for pad
    eos_indices    : (B,)              long   — position of EOT token per row
    __keys__       : list[str]
    """
    images   = torch.stack([s["image"] for s in batch])
    captions = [s["text"] for s in batch]

    # Returns list[IntTensor], each with shape (seq_len_i,) — variable length,
    # already has SOT prepended and EOT appended by the tokenizer.
    token_list: list[Tensor] = tokenizer(captions)

    B = len(token_list)
    input_ids      = torch.zeros(B, MAX_SEQ_LEN, dtype=torch.long)
    attention_mask = torch.zeros(B, MAX_SEQ_LEN, dtype=torch.long)
    eos_indices    = torch.zeros(B, dtype=torch.long)

    for i, tokens in enumerate(token_list):
        # Truncate to MAX_SEQ_LEN; if truncated the EOT gets cut — force it back
        # at the last slot (mirrors official CLIP behaviour).
        seq = tokens[:MAX_SEQ_LEN].long()
        if len(tokens) > MAX_SEQ_LEN:
            seq = seq.clone()
            seq[-1] = _EOT_TOKEN

        n = len(seq)                              # actual length after truncation
        input_ids[i, :n]      = seq
        attention_mask[i, :n] = 1
        eos_indices[i]        = n - 1             # EOT is always the last real token

    return {
        "pixel_values":   images,
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "eos_indices":    eos_indices,
        "__keys__":       [s["__key__"] for s in batch],
    }

def build_cc3m_dataloader(
    tarfiles: str | list[str],
    tokenizer: Tokenizer,
    batch_size: int = 256,
    num_workers: int = 4,
    buffer_size: int = 5000,
    infinite_stream: bool = True,
    seed: int = 42,
    image_transform: Optional[Callable] = None,
) -> torch.utils.data.DataLoader:
    """
    Build a DataLoader over CC3M TAR shards for Meridian training.

    Args:
        tarfiles:        Path glob(s) pointing at img2dataset TAR shards.
        tokenizer:       Meridian Tokenizer — callable that batch-tokenises captions
                         and returns {"input_ids": Tensor, "attention_mask": Tensor}.
        batch_size:      Samples per GPU per step.
        num_workers:     Dataloader worker processes (4-8 is typical on vast.ai).
        buffer_size:     Shuffle buffer depth.  0 disables in-buffer shuffling.
        infinite_stream: Loop forever (True for training) or one pass (False for eval).
        seed:            RNG seed for reproducible shuffling.
        image_transform: Optional override for the default CC3M train transform.

    Returns:
        torch.utils.data.DataLoader wrapping an ImageTextWebDataset.
    """
    mapper = CC3MTarMapper(image_transform = image_transform)
    dataset = ImageTextWebDataset(
        tarfiles = tarfiles,
        mapper = mapper,
        buffer_size = buffer_size,
        infinite_stream = infinite_stream,
        seed = seed,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        collate_fn = functools.partial(_collate_cc3m, tokenizer=tokenizer),
        pin_memory = True,
        drop_last = True,           # constant batch size; avoids partial-batch edge cases in loss
        persistent_workers = num_workers > 0,
    )
    return loader