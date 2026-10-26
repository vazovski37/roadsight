"""Upper bound of box-guided fusion: feed ground-truth semantics + ground-truth boxes through `fuse`.

Whatever PQ is lost here is lost by the fusion rule itself (overlapping same-class objects inside
each other's boxes), not by the network. Reported overall and on images with / without overlaps.

    python tools/fusion_oracle.py --data-root data/bdd10k --out results/fusion_oracle.json
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roadsight.classes import IGNORE  # noqa: E402
from roadsight.labels import gt_panoptic, load_bitmask, load_sem, read_split  # noqa: E402
from roadsight.metrics import panoptic_metrics, pq_stats  # noqa: E402
from roadsight.postprocess import fuse  # noqa: E402

ORACLE = {"score_thr": 0.0, "fill_thr": 0.0, "stuff_area": 0.0, "thin_stuff_area": 0.0}


def one(args):
    root, name, ann = args
    gt_cls, gt_inst, crowd = gt_panoptic(load_sem(root, name), load_bitmask(root, name))
    keep = ~np.asarray(ann["crowd"], bool)
    boxes = np.asarray(ann["boxes"], np.float32).reshape(-1, 4)[keep]
    labels = np.asarray(ann["labels"])[keep]
    # boxes ordered by area, smallest first (smaller objects tend to be in front); score = order
    order = np.argsort((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    scores = np.zeros(len(boxes), np.float32)
    scores[order] = np.linspace(1.0, 0.5, len(boxes)) if len(boxes) else scores
    sem_in = np.where(gt_cls == IGNORE, 0, gt_cls).astype(np.uint8)
    cls_map, inst_map, _ = fuse(sem_in, boxes, scores, labels, ORACLE)
    overlap = False
    for c in np.unique(labels):
        b = boxes[labels == c]
        if len(b) > 1:
            lt = np.maximum(b[:, None, :2], b[None, :, :2])
            rb = np.minimum(b[:, None, 2:], b[None, :, 2:])
            inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
            area = np.prod(b[:, 2:] - b[:, :2], axis=1)
            iou = inter / (area[:, None] + area[None] - inter)
            overlap |= bool((np.triu(iou, 1) > 0.3).any())
    return pq_stats(cls_map, inst_map, gt_cls, gt_inst, crowd), overlap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/bdd10k")
    ap.add_argument("--split", default="val")
    ap.add_argument("--workers", type=int, default=min(os.cpu_count() or 4, 12))
    ap.add_argument("--out", default="results/fusion_oracle.json")
    a = ap.parse_args()
    ann = json.loads(Path(a.data_root, "annotations.json").read_text())
    names = read_split(a.data_root, a.split)
    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(pool.map(one, [(a.data_root, n, ann[n]) for n in names], chunksize=8))
    pick = lambda m: {k: round(m[k], 2) for k in ("PQ", "PQ_th", "PQ_st", "SQ_th", "RQ_th")}  # noqa: E731
    out = {
        "all": pick(panoptic_metrics(sum(s for s, _ in rows))),
        "with_same_class_overlap": pick(panoptic_metrics(sum(s for s, o in rows if o))),
        "without_same_class_overlap": pick(panoptic_metrics(sum(s for s, o in rows if not o))),
        "images_with_overlap": int(sum(o for _, o in rows)), "images": len(rows),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
