"""Per-image scoring against ground truth (numpy only, runs in worker processes).

Shared by the PyTorch evaluation (engine.evaluate) and the ONNX evaluation (tools/onnx_eval.py),
so both report numbers through exactly the same post-processing and metrics.
"""
from pathlib import Path

import numpy as np

from .labels import gt_panoptic, load_bitmask, load_sem, resize_labels
from .metrics import (coco_box_metrics, coco_ground_truth, confusion, panoptic_metrics, pq_stats,
                      semantic_metrics)
from .postprocess import decode_detections, fuse, scale_boxes


def eval_image(root, name, outputs, size, det_cfg, fusion_cfg, dump_dir=None):
    """Score one image. outputs: model outputs at the model's input size (sem / scores / boxes)."""
    w, h = size
    gt_sem = load_sem(root, name)
    h0, w0 = gt_sem.shape
    res = {}
    if "sem" in outputs:
        res["conf"] = confusion(resize_labels(outputs["sem"], w0, h0), gt_sem)
    if "scores" in outputs:
        boxes, scores, labels = decode_detections(outputs["scores"], outputs["boxes"], (w, h), **det_cfg)
        full = scale_boxes(boxes, (w, h), (w0, h0))
        res["dets"] = [(name, int(l) + 1, [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])], float(s))
                       for b, s, l in zip(full, scores, labels)]
        if "sem" in outputs:
            cls_map, inst_map, _ = fuse(outputs["sem"], boxes, scores, labels, fusion_cfg)
            gt_cls, gt_inst, crowd = gt_panoptic(gt_sem, load_bitmask(root, name))
            res["pq"] = pq_stats(resize_labels(cls_map, w0, h0), resize_labels(inst_map.astype(np.uint16), w0, h0),
                                 gt_cls, gt_inst, crowd)
        if dump_dir is not None:
            np.savez_compressed(Path(dump_dir, f"{name}.npz"), sem=outputs.get("sem", np.zeros(0)), boxes=boxes,
                                scores=scores, labels=labels, size=np.array(size))
    return res


class Accumulator:
    """Sums per-image results and turns them into the final metric dict."""

    def __init__(self):
        self.conf, self.pq, self.dets, self.has_det = None, None, [], False

    def add(self, r):
        if "conf" in r:
            self.conf = r["conf"] if self.conf is None else self.conf + r["conf"]
        if "pq" in r:
            self.pq = r["pq"] if self.pq is None else self.pq + r["pq"]
        if "dets" in r:
            self.has_det = True
            self.dets.extend(r["dets"])

    def metrics(self, names, annotations):
        out = {}
        if self.conf is not None:
            out.update(semantic_metrics(self.conf))
        if self.has_det:
            img_id = {n: i for i, n in enumerate(names)}
            coco_dets = [{"image_id": img_id[n], "category_id": c, "bbox": b, "score": s} for n, c, b, s in self.dets]
            out.update(coco_box_metrics(coco_ground_truth(names, annotations), coco_dets))
        if self.pq is not None:
            out.update(panoptic_metrics(self.pq))
        return out
