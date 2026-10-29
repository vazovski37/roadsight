"""Side-by-side pictures of the deployed model: input | ground truth | prediction, with per-image PQ.

    python tools/visualize.py models/roadsight_int8.onnx --split val --n 8 --out results/samples

Runs the exact code path of the demo API (app.main.Predictor). Images are not hand-picked: the split
is sorted by mean brightness and one image is taken at each of n evenly spaced quantiles, so the
set runs from night to bright day.
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roadsight.labels import gt_panoptic, load_bitmask, load_image, load_sem, read_split, resize_labels  # noqa: E402
from roadsight.metrics import panoptic_metrics, pq_stats  # noqa: E402
from roadsight.postprocess import colorize  # noqa: E402

TILE = (640, 360)


def pick_by_brightness(root, names, n):
    b = [cv2.imread(str(Path(root, "images", f"{x}.jpg")), cv2.IMREAD_REDUCED_GRAYSCALE_4).mean() for x in names]
    order = np.argsort(b)
    return [names[order[int(q * (len(names) - 1))]] for q in np.linspace(0.03, 0.97, n)]


def caption(img, text):
    out = cv2.copyMakeBorder(img, 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--data-root", default="data/bdd10k")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="results/samples")
    a = ap.parse_args()
    os.environ["ROADSIGHT_MODEL"] = a.model
    from app.main import predictor  # loads a.model

    Path(a.out).mkdir(parents=True, exist_ok=True)
    rows = []
    for name in pick_by_brightness(a.data_root, read_split(a.data_root, a.split), a.n):
        rgb = load_image(a.data_root, name)
        h0, w0 = rgb.shape[:2]
        result, cls_map, inst_map = predictor(rgb)
        pred = predictor.overlay(rgb, cls_map, inst_map, result["instances"])

        gt_cls, gt_inst, crowd = gt_panoptic(load_sem(a.data_root, name), load_bitmask(a.data_root, name))
        gt = colorize(rgb, gt_cls, gt_inst, [{"id": i} for i in np.unique(gt_inst[gt_inst > 0])])
        stats = pq_stats(resize_labels(cls_map, w0, h0), resize_labels(inst_map.astype(np.uint16), w0, h0),
                         gt_cls, gt_inst, crowd)
        pq = panoptic_metrics(stats)["PQ"]

        tiles = [caption(cv2.resize(im, TILE, interpolation=cv2.INTER_AREA), t) for im, t in (
            (rgb, f"input {name}"), (gt, "ground truth"),
            (pred, f"prediction  PQ {pq:.1f}  {result['timings']['total_ms']:.0f} ms"))]
        row = np.hstack(tiles)
        row = np.vstack([row, caption(np.full((4, row.shape[1], 3), 255, np.uint8), result["summary"]["text"][:150])])
        cv2.imwrite(str(Path(a.out, f"{name}.jpg")), cv2.cvtColor(row, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        rows.append(row)
        print(f"{name}  PQ {pq:5.1f}  {result['timings']['total_ms']:6.1f} ms  {result['summary']['text']}", flush=True)
    grid = np.vstack(rows)
    cv2.imwrite(str(Path(a.out, "grid.jpg")), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"wrote {len(rows)} samples and grid.jpg to {a.out}")


if __name__ == "__main__":
    main()
