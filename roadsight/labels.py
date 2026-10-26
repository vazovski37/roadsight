"""Reading images and labels from the prepared dataset (numpy/OpenCV only, no torch).

Kept torch-free so evaluation worker processes and the ONNX demo stay lightweight.
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .classes import IGNORE, INS_CAT_TO_DET, STUFF_IDS, det_to_sem


def read_split(root, split):
    return Path(root, "splits", f"{split}.txt").read_text().split()


def load_image(root, name):
    img = cv2.imread(str(Path(root, "images", f"{name}.jpg")), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_sem(root, name):
    return cv2.imread(str(Path(root, "sem", f"{name}.png")), cv2.IMREAD_UNCHANGED)


def load_bitmask(root, name):
    # PIL keeps RGBA channel order (cv2 would return BGRA).
    return np.asarray(Image.open(Path(root, "inst", f"{name}.png")))


def decode_bitmask(bitmask):
    """Split an RGBA bitmask into category (1..8, 0 = none), attribute bits and instance id."""
    category = bitmask[..., 0]
    attributes = bitmask[..., 1]
    instance = (bitmask[..., 2].astype(np.int32) << 8) | bitmask[..., 3].astype(np.int32)
    return category, attributes, instance


def resize_image(img, width, height):
    """The one resize used everywhere (training, evaluation, ONNX demo) so inputs always match."""
    h, w = img.shape[:2]
    if (w, h) == (width, height):
        return img
    interp = cv2.INTER_AREA if width < w else cv2.INTER_LINEAR
    return cv2.resize(img, (width, height), interpolation=interp)


def resize_labels(labels, width, height):
    if labels.shape[1] == width and labels.shape[0] == height:
        return labels
    return cv2.resize(labels, (width, height), interpolation=cv2.INTER_NEAREST)


def gt_panoptic(sem, bitmask):
    """Build panoptic ground truth (Cityscapes convention) from the semantic mask and instance bitmask.

    Returns (cls, inst, crowd): cls is the semantic class per pixel (255 = void), inst is a
    per-image instance index for thing pixels (0 for stuff/void), crowd marks crowd/ignore regions.
    Thing pixels that the semantic mask labels as a thing but no instance covers become void.
    """
    category, attributes, instance = decode_bitmask(bitmask)
    cls = np.full(sem.shape, IGNORE, np.uint8)
    stuff = np.isin(sem, STUFF_IDS)
    cls[stuff] = sem[stuff]

    inst = np.zeros(sem.shape, np.int32)
    things = category > 0
    if things.any():
        for cat in np.unique(category[things]):
            cls[category == cat] = det_to_sem(INS_CAT_TO_DET[int(cat)])
        # re-index BDD annotation ids to 1..n for this image
        _, inverse = np.unique(instance[things], return_inverse=True)
        inst[things] = inverse + 1
    crowd = things & ((attributes & 0b11) > 0)  # crowd bit (2) or ignore bit (1)
    return cls, inst, crowd
