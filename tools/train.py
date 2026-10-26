"""Train one experiment.

    python tools/train.py configs/r18_640.yaml [key=value ...]
    python tools/train.py configs/r18_640.yaml name=r18_seed1 train.seed=1
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roadsight.config import load_config  # noqa: E402
from roadsight.engine import train  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    train(cfg, Path(args.out) / cfg["name"])
