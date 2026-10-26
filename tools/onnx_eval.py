"""Accuracy of exported ONNX models (FP32 / FP16 / INT8 / pruned) on a split, on CPU.

    python tools/onnx_eval.py models/r18_lite_640.onnx models/r18_lite_640_int8.onnx --split val

Uses the same post-processing and metrics as the PyTorch evaluation (roadsight.scoring), so the
numbers in the optimisation table differ only because of the model variant.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roadsight.labels import load_image, read_split, resize_image  # noqa: E402
from roadsight.scoring import Accumulator, eval_image  # noqa: E402

DET_CFG = {"score_thr": 0.05, "nms_iou": 0.6, "max_det": 100}


def evaluate_onnx(model, data_root, split, workers, limit=0, fusion=None):
    meta = json.loads(Path(model).with_suffix(".json").read_text())
    w, h = meta["size"]
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    names = read_split(data_root, split)
    if limit:
        names = names[:limit]
    ann = json.loads(Path(data_root, "annotations.json").read_text())
    acc = Accumulator()
    with ProcessPoolExecutor(workers) as pool:
        futures = []
        for name in names:
            x = resize_image(load_image(data_root, name), w, h).transpose(2, 0, 1)[None].astype(np.float32)
            outs = {k: v[0] for k, v in zip(meta["outputs"], sess.run(None, {"image": x}))}
            if "scores" in outs:
                keep = outs["scores"].max(1) > DET_CFG["score_thr"]
                outs["scores"], outs["boxes"] = outs["scores"][keep], outs["boxes"][keep]
            futures.append(pool.submit(eval_image, data_root, name, outs, (w, h), DET_CFG, fusion or meta["fusion"]))
        for f in futures:
            acc.add(f.result())
    return acc.metrics(names, {n: ann[n] for n in names})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--data-root", default="data/bdd10k")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", default="results/onnx")
    a = ap.parse_args()
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    for m in a.models:
        metrics = evaluate_onnx(m, a.data_root, a.split, a.workers, a.limit)
        metrics.update({"model": m, "split": a.split})
        Path(a.out_dir, f"{Path(m).stem}_{a.split}.json").write_text(json.dumps(metrics, indent=2))
        print(f"{Path(m).stem:28s} " + ", ".join(f"{k} {metrics[k]:.2f}" for k in
                                                  ("PQ", "PQ_th", "PQ_st", "mAP", "mIoU", "mIoU_thin") if k in metrics),
              flush=True)


if __name__ == "__main__":
    main()
