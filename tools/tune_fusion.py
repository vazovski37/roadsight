"""Tune the box-guided fusion thresholds on the holdout split, then report them on val (experiment E5).

Uses the raw outputs a run dumped during evaluation (runs/<name>/dumps/{holdout,val}/*.npz), so no
model is run: only fusion + PQ are recomputed. Search is coordinate-wise to keep it cheap:
  1. score_thr x fill_thr          (stuff settings at default)
  2. stuff_area with / without the thin-class exemption
PQ during the search is computed at the model's resolution (GT downsampled with nearest);
the final val numbers use full 1280x720 resolution like every other result.

    python tools/tune_fusion.py runs/r18_lite --out results/fusion_r18_lite.json
"""
import argparse
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roadsight.labels import gt_panoptic, load_bitmask, load_sem, resize_labels  # noqa: E402
from roadsight.metrics import panoptic_metrics, pq_stats  # noqa: E402
from roadsight.postprocess import DEFAULT_FUSION, fuse  # noqa: E402


def score_image(args):
    """PQ stats of one image for every fusion setting in `grid`."""
    root, npz, grid, full_res = args
    d = np.load(npz)
    name = Path(npz).stem
    sem_gt = load_sem(root, name)
    gt_cls, gt_inst, crowd = gt_panoptic(sem_gt, load_bitmask(root, name))
    h, w = d["sem"].shape
    if not full_res:
        gt_cls = resize_labels(gt_cls, w, h)
        gt_inst = resize_labels(gt_inst.astype(np.uint16), w, h).astype(np.int32)
        crowd = resize_labels(crowd.astype(np.uint8), w, h).astype(bool)
    out = []
    for cfg in grid:
        cls_map, inst_map, _ = fuse(d["sem"], d["boxes"], d["scores"], d["labels"], cfg)
        if full_res:
            h0, w0 = gt_cls.shape
            cls_map = resize_labels(cls_map, w0, h0)
            inst_map = resize_labels(inst_map.astype(np.uint16), w0, h0)
        out.append(pq_stats(cls_map, inst_map, gt_cls, gt_inst, crowd))
    return np.stack(out)


def sweep(root, files, grid, full_res, workers):
    with ProcessPoolExecutor(workers) as pool:
        total = sum(pool.map(score_image, [(root, f, grid, full_res) for f in files], chunksize=8))
    return [panoptic_metrics(s) for s in total]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--data-root", default="data/bdd10k")
    ap.add_argument("--workers", type=int, default=min(os.cpu_count() or 4, 12))
    ap.add_argument("--tune-split", default="holdout")
    ap.add_argument("--report-split", default="val")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    hold = sorted(Path(a.run, "dumps", a.tune_split).glob("*.npz"))
    val = sorted(Path(a.run, "dumps", a.report_split).glob("*.npz"))
    assert hold and val, "run has no dumps; train with eval.dump=true or use tools/evaluate.py --dump"
    log = {"run": a.run, "default": DEFAULT_FUSION, "search": []}

    grid1 = [{**DEFAULT_FUSION, "score_thr": s, "fill_thr": f}
             for s, f in itertools.product([0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3])]
    res1 = sweep(a.data_root, hold, grid1, False, a.workers)
    best = grid1[int(np.argmax([r["PQ"] for r in res1]))]
    log["search"] += [{"cfg": g, "PQ": r["PQ"], "PQ_th": r["PQ_th"], "PQ_st": r["PQ_st"]} for g, r in zip(grid1, res1)]

    grid2 = [{**best, "stuff_area": s, "thin_stuff_area": (0.0 if exempt else s)}
             for s, exempt in itertools.product([0.0, 0.001, 0.002, 0.004], [True, False])]
    res2 = sweep(a.data_root, hold, grid2, False, a.workers)
    best = grid2[int(np.argmax([r["PQ"] for r in res2]))]
    log["search"] += [{"cfg": g, "PQ": r["PQ"], "PQ_th": r["PQ_th"], "PQ_st": r["PQ_st"]} for g, r in zip(grid2, res2)]
    log["best"] = best

    default_val, best_val = sweep(a.data_root, val, [DEFAULT_FUSION, best], True, a.workers)
    log["val_default"] = default_val
    log["val_tuned"] = best_val
    # the cost of the thin-class exemption, measured on val with otherwise tuned settings
    no_exempt = {**best, "thin_stuff_area": best["stuff_area"]}
    log["val_tuned_without_thin_exemption"] = sweep(a.data_root, val, [no_exempt], True, a.workers)[0]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(log, indent=2))
    pick = lambda m: {k: round(m[k], 2) for k in ("PQ", "PQ_th", "PQ_st")}  # noqa: E731
    print("best", best)
    print("val default", pick(default_val), "-> tuned", pick(best_val))


if __name__ == "__main__":
    main()
