"""
Image transforms for Meridian.

Three pipelines:
    build_train_transform  - full images during training (random augmentation)
    build_region_transform - region crops during training (resize only, no re-crop)
    build_eval_transform   - both images and regions at eval / inference time

All three normalise with CLIP's ViT-B/16 statistics so features fed to the
frozen encoder are in the distribution it was trained on.

Why separate train vs region transforms?
    The full image goes through RandomResizedCrop because we want the model to
    see the image at varying scales and positions.  The region crop is already
    defined by a specific VG dataset bounding box — applying another random crop on top
    of it would destroy the exact content we're trying to encode as a child node
    in the hierarchy.  So regions only get Resize + optional flip.
"""

from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


# ViT-B/16 (OpenAI) normalisation constants
IMAGE_SIZE = 224
CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


def build_train_transform(image_size: int = IMAGE_SIZE) -> T.Compose:
    """
    Transform for full images during training.

    RandomResizedCrop with a generous scale range (0.5–1.0) ensures the model
    sees the full image at many different zoom levels.  ColorJitter adds
    photometric diversity without breaking semantic content.

    Args:
        image_size: Output spatial size (default 224, CLIP's native resolution).

    Returns:
        A torchvision Compose transform that accepts a PIL Image.
    """
    return T.Compose([
        T.RandomResizedCrop(
            image_size,
            scale=(0.5, 1.0),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def build_region_transform(image_size: int = IMAGE_SIZE) -> T.Compose:
    """
    Transform for VG region crops during training.

    The crop is already extracted from the PIL image using the VG bounding box
    before this transform runs.  We only resize and optionally flip — no
    RandomResizedCrop (would discard the specific region content) and no
    ColorJitter (keeps the region faithful for hierarchy supervision).

    Args:
        image_size: Output spatial size.

    Returns:
        A torchvision Compose transform that accepts a PIL Image.
    """
    return T.Compose([
        T.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def build_eval_transform(image_size: int = IMAGE_SIZE) -> T.Compose:
    """
    Deterministic transform for evaluation and inference.

    Used for both full images and region crops at eval time.  No randomness —
    ensures reproducible embeddings across runs.

    Resize first to image_size + a small margin, then centre-crop to
    image_size.  This avoids distortion from direct resize-to-square on
    non-square inputs and matches CLIP's standard eval preprocessing.

    Args:
        image_size: Output spatial size.

    Returns:
        A torchvision Compose transform that accepts a PIL Image.
    """
    return T.Compose([
        T.Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])