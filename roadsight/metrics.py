"""Metrics: semantic mIoU, panoptic quality (PQ, Kirillov et al. 2019) and COCO box AP."""
import contextlib
import io

import numpy as np

from .classes import DET_CLASSES, IGNORE, NUM_SEM, SEM_CLASSES, STUFF_IDS, THING_IDS, THIN_DET, THIN_SEM


def confusion(pred, gt, n=NUM_SEM):
    valid = gt != IGNORE
    return np.bincount(n * gt[valid].astype(np.int64) + pred[valid], minlength=n * n).reshape(n, n)


def semantic_metrics(conf):
    tp = np.diag(conf).astype(np.float64)
    iou = tp / np.maximum(conf.sum(0) + conf.sum(1) - tp, 1)
    present = conf.sum(1) > 0
    out = {"mIoU": float(iou[present].mean() * 100)}
    out.update({f"IoU/{c}": float(iou[i] * 100) for i, c in enumerate(SEM_CLASSES)})
    out["mIoU_thin"] = float(np.mean([iou[SEM_CLASSES.index(c)] for c in THIN_SEM]) * 100)
    return out


def _segments(cls_map, inst_map):
    """Encode (class, instance) as one id; 0 = void. Stuff has instance 0 (one segment per class)."""
    seg = (cls_map.astype(np.int64) + 1) * 1000 + inst_map
    seg[cls_map == IGNORE] = 0
    return seg


def pq_stats(pred_cls, pred_inst, gt_cls, gt_inst, gt_crowd):
    """Per-class [tp, fp, fn, iou_sum] for one image, following the panopticapi matching rules."""
    stats = np.zeros((NUM_SEM, 4), np.float64)
    gt = _segments(gt_cls, gt_inst)
    # all crowd pixels of a class form one crowd segment
    gt[gt_crowd] = (gt_cls[gt_crowd].astype(np.int64) + 1) * 1000 + 999
    pred = _segments(pred_cls, pred_inst)

    gt_ids, gt_area = np.unique(gt, return_counts=True)
    pred_ids, pred_area = np.unique(pred, return_counts=True)
    gt_area, pred_area = dict(zip(gt_ids, gt_area)), dict(zip(pred_ids, pred_area))
    pair_ids, pair_area = np.unique(gt * 10**7 + pred, return_counts=True)
    inter = {(int(p // 10**7), int(p % 10**7)): int(a) for p, a in zip(pair_ids, pair_area)}

    cls_of = lambda seg: seg // 1000 - 1
    is_crowd = lambda seg: seg % 1000 == 999
    matched_gt, matched_pred = set(), set()
    for (g, p), a in inter.items():
        if g == 0 or p == 0 or is_crowd(g) or cls_of(g) != cls_of(p):
            continue
        union = pred_area[p] + gt_area[g] - a - inter.get((0, p), 0)
        iou = a / union
        if iou > 0.5:
            c = cls_of(g)
            stats[c, 0] += 1
            stats[c, 3] += iou
            matched_gt.add(g)
            matched_pred.add(p)
    for g in gt_area:
        if g != 0 and not is_crowd(g) and g not in matched_gt:
            stats[cls_of(g), 2] += 1
    for p in pred_area:
        if p == 0 or p in matched_pred:
            continue
        c = cls_of(p)
        # predictions lying mostly on void or on a crowd region of their class are not false positives
        ignored = inter.get((0, p), 0) + inter.get(((c + 1) * 1000 + 999, p), 0)
        if ignored / pred_area[p] > 0.5:
            continue
        stats[c, 1] += 1
    return stats


def panoptic_metrics(stats):
    tp, fp, fn, iou = stats.T
    seen = tp + fp + fn > 0
    pq = np.where(seen, iou / np.maximum(tp + 0.5 * fp + 0.5 * fn, 1e-9), 0)
    sq = np.where(tp > 0, iou / np.maximum(tp, 1e-9), 0)
    rq = np.where(seen, tp / np.maximum(tp + 0.5 * fp + 0.5 * fn, 1e-9), 0)

    def avg(arr, ids):
        ids = [i for i in ids if seen[i]]
        return float(np.mean(arr[ids]) * 100) if ids else 0.0

    every = list(range(NUM_SEM))
    out = {"PQ": avg(pq, every), "SQ": avg(sq, every), "RQ": avg(rq, every),
           "PQ_th": avg(pq, THING_IDS), "PQ_st": avg(pq, STUFF_IDS),
           "SQ_th": avg(sq, THING_IDS), "RQ_th": avg(rq, THING_IDS)}
    out.update({f"PQ/{c}": float(pq[i] * 100) for i, c in enumerate(SEM_CLASSES)})
    return out


def coco_ground_truth(names, annotations):
    """COCO-format box ground truth at the original resolution (crowd/ignore -> iscrowd)."""
    images, anns = [], []
    for img_id, name in enumerate(names):
        images.append({"id": img_id, "file_name": name, "width": 1280, "height": 720})
        a = annotations[name]
        for box, lab, crowd, area in zip(a["boxes"], a["labels"], a["crowd"], a["areas"]):
            x1, y1, x2, y2 = box
            anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": lab + 1,
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "area": area, "iscrowd": int(crowd)})
    cats = [{"id": i + 1, "name": c} for i, c in enumerate(DET_CLASSES)]
    return {"images": images, "annotations": anns, "categories": cats}


def coco_box_metrics(gt_dict, detections):
    """detections: list of dicts (image_id, category_id, bbox xywh, score)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    if not detections:
        return {"mAP": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = gt_dict
        gt.createIndex()
        dt = gt.loadRes(detections)
        ev = COCOeval(gt, dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = ev.stats
    out = {"mAP": s[0] * 100, "AP50": s[1] * 100, "AP75": s[2] * 100,
           "AP_S": s[3] * 100, "AP_M": s[4] * 100, "AP_L": s[5] * 100, "AR100": s[8] * 100}
    precision = ev.eval["precision"]  # [iou, recall, class, area, maxdet]
    for k, c in enumerate(DET_CLASSES):
        p = precision[:, :, k, 0, -1]
        out[f"AP/{c}"] = float(p[p > -1].mean() * 100) if (p > -1).any() else 0.0
    out["AP_thin"] = float(np.mean([out[f"AP/{c}"] for c in THIN_DET]))
    return {k: float(v) for k, v in out.items()}
