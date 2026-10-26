"""Error analysis of one run on val, from the outputs it dumped during evaluation.

    python tools/error_analysis.py runs/r18_lite --out results/analysis/r18_lite [--fusion results/fusion_x.json]

Produces (in --out):
  summary.json          everything below as numbers
  confusion.png         row-normalised 19x19 semantic confusion matrix
  slices                PQ / mIoU / AP on: dark vs bright images (brightness proxy for night: the
                        10K val images have no time-of-day labels), images with vs without
                        overlapping same-class objects (the known weak spot of box-guided fusion)
  detection errors      TIDE-style counts at score >= 0.3: classification, localisation, both,
                        duplicate, background false positives, and missed objects by size
  worst/*.jpg           the 12 images with the lowest image-level PQ: image | ground truth | prediction
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roadsight.classes import DET_CLASSES, NUM_SEM, SEM_CLASSES  # noqa: E402
from roadsight.labels import gt_panoptic, load_bitmask, load_image, load_sem, resize_labels  # noqa: E402
from roadsight.metrics import (coco_box_metrics, coco_ground_truth, confusion, panoptic_metrics, pq_stats,  # noqa: E402
                               semantic_metrics)
from roadsight.postprocess import DEFAULT_FUSION, colorize, fuse, scale_boxes  # noqa: E402

SCORE_THR = 0.3


def box_iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def detection_errors(boxes, scores, labels, gt_boxes, gt_labels, gt_areas):
    """Classify each confident detection and each missed object (TIDE-style, counts only)."""
    errors = dict.fromkeys(["tp", "cls", "loc", "both", "dupe", "bkg"], 0)
    missed = {"small": 0, "medium": 0, "large": 0}
    keep = scores >= SCORE_THR
    boxes, labels = boxes[keep], labels[keep]
    order = np.argsort(-scores[keep])
    iou = box_iou(boxes, gt_boxes)
    matched = np.zeros(len(gt_boxes), bool)
    for i in order:
        same = gt_labels == labels[i]
        best_same = iou[i][same].max() if same.any() else 0.0
        best_any = iou[i].max() if len(gt_boxes) else 0.0
        if best_same >= 0.5:
            j = np.flatnonzero(same)[np.argmax(iou[i][same])]
            if matched[j]:
                errors["dupe"] += 1
            else:
                matched[j] = True
                errors["tp"] += 1
        elif best_any >= 0.5:
            errors["cls"] += 1
        elif best_same >= 0.1:
            errors["loc"] += 1
        elif best_any >= 0.1:
            errors["both"] += 1
        else:
            errors["bkg"] += 1
    for j in np.flatnonzero(~matched):
        a = gt_areas[j]
        missed["small" if a < 32 ** 2 else "medium" if a < 96 ** 2 else "large"] += 1
    return errors, missed


def analyse(args):
    root, npz, fusion, ann = args
    d = np.load(npz)
    name = Path(npz).stem
    w, h = (int(v) for v in d["size"])
    img = load_image(root, name)
    h0, w0 = img.shape[:2]
    sem_gt = load_sem(root, name)
    gt_cls, gt_inst, crowd = gt_panoptic(sem_gt, load_bitmask(root, name))
    cls_map, inst_map, _ = fuse(d["sem"], d["boxes"], d["scores"], d["labels"], fusion)
    cls_full = resize_labels(cls_map, w0, h0)
    inst_full = resize_labels(inst_map.astype(np.uint16), w0, h0)
    stats = pq_stats(cls_full, inst_full, gt_cls, gt_inst, crowd)
    conf = confusion(resize_labels(d["sem"], w0, h0), sem_gt)

    boxes = scale_boxes(d["boxes"], (w, h), (w0, h0))
    keep = ~np.asarray(ann["crowd"], bool)
    gt_boxes = np.asarray(ann["boxes"], np.float32).reshape(-1, 4)[keep]
    gt_labels = np.asarray(ann["labels"])[keep]
    gt_areas = np.asarray(ann["areas"])[keep]
    errors, missed = detection_errors(boxes, d["scores"], d["labels"], gt_boxes, gt_labels, gt_areas)
    overlap = False
    for c in np.unique(gt_labels):
        bc = gt_boxes[gt_labels == c]
        if len(bc) > 1 and (np.triu(box_iou(bc, bc), 1) > 0.3).any():
            overlap = True
            break
    tp, fp, fn, iou = stats.sum(0)
    image_pq = iou / max(tp + 0.5 * fp + 0.5 * fn, 1e-9)
    dets = [(name, int(l) + 1, [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])], float(s))
            for b, s, l in zip(boxes, d["scores"], d["labels"])]
    brightness = float(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).mean())
    return {"name": name, "pq": stats, "conf": conf, "errors": errors, "missed": missed, "overlap": overlap,
            "image_pq": float(image_pq), "brightness": brightness, "dets": dets}


def slice_metrics(rows, names_all, annotations):
    names = [r["name"] for r in rows]
    out = {"images": len(rows)}
    out.update({k: v for k, v in panoptic_metrics(sum(r["pq"] for r in rows)).items() if "/" not in k})
    out.update({k: v for k, v in semantic_metrics(sum(r["conf"] for r in rows)).items() if "/" not in k})
    dets = [d for r in rows for d in r["dets"]]
    img_id = {n: i for i, n in enumerate(names)}
    coco = [{"image_id": img_id[n], "category_id": c, "bbox": b, "score": s} for n, c, b, s in dets]
    out.update({k: v for k, v in coco_box_metrics(coco_ground_truth(names, annotations), coco).items()
                if "/" not in k})
    return out


def render(root, row, fusion, run, split, out_dir):
    d = np.load(Path(run, "dumps", split, f"{row['name']}.npz"))
    img = load_image(root, row["name"])
    h0, w0 = img.shape[:2]
    gt_cls, gt_inst, _ = gt_panoptic(load_sem(root, row["name"]), load_bitmask(root, row["name"]))
    cls_map, inst_map, inst = fuse(d["sem"], d["boxes"], d["scores"], d["labels"], fusion)
    pred = colorize(img, resize_labels(cls_map, w0, h0), resize_labels(inst_map.astype(np.uint16), w0, h0),
                    [{"id": i["id"]} for i in inst])
    gt_instances = [{"id": int(i)} for i in np.unique(gt_inst) if i > 0]
    gt = colorize(img, gt_cls, gt_inst, gt_instances)
    panel = np.concatenate([img, gt, pred], axis=1)
    panel = cv2.resize(panel, (panel.shape[1] // 2, panel.shape[0] // 2), interpolation=cv2.INTER_AREA)
    for x, t in ((10, "image"), (w0 // 2 + 10, "ground truth"), (w0 + 10, f"prediction (image PQ {row['image_pq']:.2f})")):
        cv2.putText(panel, t, (x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(Path(out_dir, f"{row['name']}.jpg")), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--data-root", default="data/bdd10k")
    ap.add_argument("--fusion", default=None, help="tune_fusion.py output; uses its best settings")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=min(os.cpu_count() or 4, 12))
    a = ap.parse_args()
    out = Path(a.out)
    (out / "worst").mkdir(parents=True, exist_ok=True)
    fusion = json.loads(Path(a.fusion).read_text())["best"] if a.fusion else DEFAULT_FUSION
    annotations = json.loads(Path(a.data_root, "annotations.json").read_text())
    files = sorted(Path(a.run, "dumps", a.split).glob("*.npz"))
    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(pool.map(analyse, [(a.data_root, f, fusion, annotations[f.stem]) for f in files], chunksize=8))
    names = [r["name"] for r in rows]

    summary = {"run": a.run, "fusion": fusion, "all": slice_metrics(rows, names, annotations)}
    b = np.array([r["brightness"] for r in rows])
    lo, hi = np.percentile(b, [33.3, 66.7])
    summary["brightness_thresholds"] = [float(lo), float(hi)]
    summary["dark_third"] = slice_metrics([r for r in rows if r["brightness"] < lo], names, annotations)
    summary["bright_third"] = slice_metrics([r for r in rows if r["brightness"] >= hi], names, annotations)
    summary["with_same_class_overlap"] = slice_metrics([r for r in rows if r["overlap"]], names, annotations)
    summary["without_same_class_overlap"] = slice_metrics([r for r in rows if not r["overlap"]], names, annotations)
    errors = {k: sum(r["errors"][k] for r in rows) for k in rows[0]["errors"]}
    missed = {k: sum(r["missed"][k] for r in rows) for k in rows[0]["missed"]}
    summary["detection_errors_at_score_0.3"] = {"detections": errors, "missed_by_size": missed}

    conf = sum(r["conf"] for r in rows).astype(np.float64)
    norm = conf / np.maximum(conf.sum(1, keepdims=True), 1)
    pairs = [(SEM_CLASSES[i], SEM_CLASSES[j], float(norm[i, j])) for i in range(NUM_SEM) for j in range(NUM_SEM)
             if i != j]
    summary["top_confusions"] = sorted(pairs, key=lambda p: -p[2])[:10]
    worst = sorted(rows, key=lambda r: r["image_pq"])[:12]
    summary["worst_images"] = [(r["name"], round(r["image_pq"], 3)) for r in worst]
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_SEM), SEM_CLASSES, rotation=70, fontsize=8)
    ax.set_yticks(range(NUM_SEM), SEM_CLASSES, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.set_title(f"Semantic confusion (row-normalised), {Path(a.run).name}, {a.split}")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "confusion.png", dpi=130)
    for r in worst:
        render(a.data_root, r, fusion, a.run, a.split, out / "worst")
    print(json.dumps({k: summary[k] for k in ("all", "dark_third", "bright_third", "with_same_class_overlap",
                                              "without_same_class_overlap", "detection_errors_at_score_0.3",
                                              "top_confusions")}, indent=1)[:3000])


if __name__ == "__main__":
    main()
