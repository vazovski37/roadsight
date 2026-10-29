"""Everything after the network, in numpy (shared by evaluation, ONNX benchmark and the demo API):
detection decoding + NMS, box-guided panoptic fusion, the text summary and the colour overlay."""
import numpy as np

from .classes import DET_CLASSES, IGNORE, PALETTE, SEM_CLASSES, STUFF_IDS, det_to_sem

DEFAULT_FUSION = {
    "score_thr": 0.4,      # detections below this do not become instances
    "fill_thr": 0.2,       # min share of the box covered by its class in the semantic map
    "stuff_area": 0.002,   # stuff classes smaller than this share of the image become void
    "thin_stuff_area": 0.0,  # ...except poles / traffic lights / signs (they are small by nature)
}
THIN_STUFF = [SEM_CLASSES.index(c) for c in ("pole", "traffic light", "traffic sign")]


def nms(boxes, scores, iou_thr):
    """Greedy non-maximum suppression; returns kept indices in descending score order."""
    order = np.argsort(-scores)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def decode_detections(scores, boxes, image_size, score_thr=0.05, pre_nms_topk=1000, nms_iou=0.6, max_det=100):
    """scores [L, C], boxes [L, 4] -> (boxes [K, 4], scores [K], labels [K]) after class-aware NMS."""
    w, h = image_size
    loc, cls = np.nonzero(scores > score_thr)
    s = scores[loc, cls]
    if s.size > pre_nms_topk:
        top = np.argpartition(-s, pre_nms_topk)[:pre_nms_topk]
        loc, cls, s = loc[top], cls[top], s[top]
    b = boxes[loc].copy()
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
    ok = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
    b, s, cls = b[ok], s[ok], cls[ok]
    # class-aware NMS in one call: shift each class's boxes to its own region of the plane
    offset = cls[:, None].astype(np.float32) * (max(w, h) + 1)
    keep = nms(b + offset, s, nms_iou)[:max_det]
    return b[keep].astype(np.float32), s[keep].astype(np.float32), cls[keep].astype(np.int64)


def fuse(sem, boxes, scores, labels, cfg=None):
    """Box-guided semantic fusion -> panoptic map.

    sem: [H, W] uint8 semantic prediction (must include thing classes).
    Each detection (highest score first) claims the still-free pixels inside its box whose
    semantic class equals the box class. Returns (cls [H, W] uint8, inst [H, W] int32, instances).
    """
    cfg = {**DEFAULT_FUSION, **(cfg or {})}
    h, w = sem.shape
    cls_map = np.full((h, w), IGNORE, np.uint8)
    inst_map = np.zeros((h, w), np.int32)
    taken = np.zeros((h, w), bool)
    instances = []
    for i in np.argsort(-scores):
        if scores[i] < cfg["score_thr"]:
            break
        x1, y1 = int(np.floor(boxes[i, 0])), int(np.floor(boxes[i, 1]))
        x2, y2 = int(np.ceil(boxes[i, 2])), int(np.ceil(boxes[i, 3]))
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
        if x2 <= x1 or y2 <= y1:
            continue
        c = det_to_sem(int(labels[i]))
        region = (sem[y1:y2, x1:x2] == c) & ~taken[y1:y2, x1:x2]
        n = int(region.sum())
        if n == 0 or n / ((x2 - x1) * (y2 - y1)) < cfg["fill_thr"]:
            continue
        k = len(instances) + 1
        cls_map[y1:y2, x1:x2][region] = c
        inst_map[y1:y2, x1:x2][region] = k
        taken[y1:y2, x1:x2] |= region
        instances.append({"id": k, "label": DET_CLASSES[int(labels[i])], "class_id": c,
                          "score": float(scores[i]), "box": [float(v) for v in boxes[i]], "area": n})
    free = ~taken
    min_area = cfg["stuff_area"] * h * w
    min_thin = cfg["thin_stuff_area"] * h * w
    for c in STUFF_IDS:
        m = (sem == c) & free
        if m.sum() >= (min_thin if c in THIN_STUFF else min_area):
            cls_map[m] = c
    # thing pixels that no detection claimed stay void (IGNORE)
    return cls_map, inst_map, instances


def summarize(cls_map, instances):
    """Rule-based text summary of the scene (a heuristic on top of the model output, not learned)."""
    h, w = cls_map.shape
    counts = {}
    for inst in instances:
        counts[inst["label"]] = counts.get(inst["label"], 0) + 1

    def side(xs):
        x = float(np.mean(xs)) / w
        return "left" if x < 1 / 3 else ("right" if x > 2 / 3 else "centre")

    plural = {"person": "people", "bus": "buses"}
    parts = [f"{n} {plural.get(name, name + 's') if n > 1 else name}"
             for name, n in sorted(counts.items(), key=lambda kv: -kv[1])]
    signals = {}
    for name in ("traffic light", "traffic sign"):
        ys, xs = np.nonzero(cls_map == SEM_CLASSES.index(name))
        if len(xs) > 20:
            signals[name] = side(xs)
            parts.append(f"{name} {signals[name]}")

    # free road: share of road pixels in the lower half, per third of the image
    lower = cls_map[h // 2:]
    thirds = np.array_split(lower == SEM_CLASSES.index("road"), 3, axis=1)
    shares = [float(t.mean()) for t in thirds]
    free_road = None
    if max(shares) > 0.15:
        free_road = ["left", "centre", "right"][int(np.argmax(shares))]
        parts.append(f"most free road {free_road}")
    text = "; ".join(parts) if parts else "empty scene"
    return {"counts": counts, "signals": signals, "free_road": free_road,
            "free_road_shares": [round(s, 3) for s in shares], "text": text}


def colorize(image, cls_map, inst_map, instances, alpha=0.55):
    """Blend the panoptic map over the image; instances get their class colour, shaded per instance."""
    palette = np.array(PALETTE + [(0, 0, 0)] * (256 - len(PALETTE)), np.uint8)
    color = palette[cls_map].astype(np.float32)
    rng = np.random.default_rng(0)
    for inst in instances:
        m = inst_map == inst["id"]
        color[m] = np.clip(color[m] * rng.uniform(0.6, 1.4), 0, 255)
    void = cls_map == IGNORE
    out = image.astype(np.float32) * (1 - alpha) + color * alpha
    out[void] = image[void]
    return out.astype(np.uint8)


def scale_boxes(boxes, from_size, to_size):
    sx, sy = to_size[0] / from_size[0], to_size[1] / from_size[1]
    return boxes * np.array([sx, sy, sx, sy], np.float32)
