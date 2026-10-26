"""PyTorch dataset for BDD100K 10K: images, 19-class semantic masks and boxes from instance bitmasks.

Layout written by tools/prepare_data.py:
    data/bdd10k/images/<name>.jpg   1280x720 RGB
    data/bdd10k/sem/<name>.png      uint8 train ids, 255 = ignore
    data/bdd10k/inst/<name>.png     BDD100K RGBA bitmask (R=category, G=attributes, B<<8|A=instance id)
    data/bdd10k/annotations.json    per-image boxes/labels/crowd/area derived from the bitmasks
    data/bdd10k/splits/{train,holdout,val}.txt
"""
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .labels import load_image, load_sem, read_split, resize_image, resize_labels

cv2.setNumThreads(1)  # DataLoader workers already run in parallel


class BDDPanoptic(Dataset):
    """Returns resized image, resized semantic mask and boxes for one image.

    Items are either an index or an (index, scale) pair; ScaleBatchSampler uses the pair form so
    every image in a batch is resized to the same random size and batches stack without padding.
    """

    def __init__(self, root, split, size, train=False):
        self.root = Path(root)
        self.names = read_split(root, split)
        self.size = size  # (width, height) at scale 1.0
        self.train = train
        with open(self.root / "annotations.json") as f:
            ann = json.load(f)
        self.ann = {n: ann[n] for n in self.names}

    def __len__(self):
        return len(self.names)

    def __getitem__(self, item):
        index, scale = item if isinstance(item, tuple) else (item, 1.0)
        name = self.names[index]
        img = load_image(self.root, name)
        h0, w0 = img.shape[:2]
        width, height = round(self.size[0] * scale), round(self.size[1] * scale)
        img = resize_image(img, width, height)

        a = self.ann[name]
        boxes = np.asarray(a["boxes"], np.float32).reshape(-1, 4)
        boxes *= np.array([width / w0, height / h0] * 2, np.float32)
        labels = np.asarray(a["labels"], np.int64)
        crowd = np.asarray(a["crowd"], bool)

        sem = None
        if self.train:
            sem = resize_labels(load_sem(self.root, name), width, height)
            if random.random() < 0.5:
                img = img[:, ::-1]
                sem = sem[:, ::-1]
                boxes = boxes[:, [2, 1, 0, 3]] * np.array([-1, 1, -1, 1], np.float32) + np.array([width, 0, width, 0], np.float32)

        out = {
            "name": name,
            "image": torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1),
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(labels),
            "crowd": torch.from_numpy(crowd),
        }
        if sem is not None:
            out["sem"] = torch.from_numpy(np.ascontiguousarray(sem))
        return out


def collate(batch):
    images = torch.stack([b["image"] for b in batch])
    sem = torch.stack([b["sem"] for b in batch]) if "sem" in batch[0] else None
    targets = [{k: b[k] for k in ("boxes", "labels", "crowd")} for b in batch]
    names = [b["name"] for b in batch]
    return images, sem, targets, names


class ScaleBatchSampler(Sampler):
    """Shuffled batches of (index, scale) pairs; one random scale per batch (multi-scale training).

    Scales come from a fixed set of 5 values so cuDNN autotuning sees few distinct input shapes.
    """

    def __init__(self, num_items, batch_size, scale_range, seed=0, num_scales=5):
        self.num_items = num_items
        self.batch_size = batch_size
        self.scales = np.linspace(scale_range[0], scale_range[1], num_scales).tolist()
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_items // self.batch_size

    def __iter__(self):
        rng = random.Random(self.seed * 1000 + self.epoch)
        order = list(range(self.num_items))
        rng.shuffle(order)
        for b in range(len(self)):
            scale = rng.choice(self.scales)
            yield [(i, scale) for i in order[b * self.batch_size:(b + 1) * self.batch_size]]
