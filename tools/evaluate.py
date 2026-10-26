"""Evaluate trained run(s) on a split, optionally at another input size or with other fusion settings.

    python tools/evaluate.py runs/r18_640 --split val
    python tools/evaluate.py runs/r18_640 --split val --size 512 288          # test-time resolution
    python tools/evaluate.py runs/r18_det runs/r18_sem --split val            # E1: two single-task models fused
    python tools/evaluate.py runs/r18_640 --split val --fusion score_thr=0.3 --out results/x.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roadsight.engine import evaluate, load_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--split", default="val")
    ap.add_argument("--size", type=int, nargs=2, default=None, help="width height")
    ap.add_argument("--fusion", nargs="*", default=[], help="key=value overrides of eval.fusion")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--dump", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    nets, cfg = [], None
    for r in args.runs:
        net, c = load_run(r, device)
        nets.append(net)
        cfg = cfg or c
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    for kv in args.fusion:
        k, v = kv.split("=")
        cfg["eval"]["fusion"][k] = yaml.safe_load(v)
    metrics = evaluate(nets, cfg, args.split, size=args.size, dump_dir=args.dump)
    metrics["runs"] = args.runs
    metrics["size"] = args.size or cfg["data"]["size"]
    print(", ".join(f"{k} {v:.2f}" for k, v in metrics.items() if isinstance(v, float) and "/" not in k))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
