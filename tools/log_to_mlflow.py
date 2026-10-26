"""Import all finished runs and evaluation results into an MLflow store.

Every training run writes config.yaml, train_log.csv and metrics.json; this script turns them into
MLflow runs (params, loss curves, final metrics, artifacts), plus one run per evaluation result
in results/ (test-time resolution, E1, ONNX variants, latency). Re-running replaces the store.

    python tools/log_to_mlflow.py --store mlruns
    pip install mlflow && mlflow ui --backend-store-uri ./mlruns

Uses MLflow's file store (a folder): it needs no database driver, which matters on machines where
the SQLAlchemy binaries cannot load. MLflow 3 keeps that store in maintenance mode, so the script
opts in explicitly; `mlflow migrate-filestore` converts the folder to SQLite if you prefer.
"""
import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import mlflow  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roadsight.config import flatten  # noqa: E402


def clean(metrics):
    return {k.replace("/", "_").replace(" ", "_").replace("^", "").replace("(", "").replace(")", "")
            .replace("<", "lt").replace(">", "gt"): float(v)
            for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--results", default="results")
    ap.add_argument("--store", default="mlruns")
    a = ap.parse_args()
    if Path(a.store).exists():
        shutil.rmtree(a.store)
    mlflow.set_tracking_uri(Path(a.store).resolve().as_uri())

    mlflow.set_experiment("roadsight-training")
    for run_dir in sorted(p for p in Path(a.runs).iterdir() if (p / "metrics.json").exists()):
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        with mlflow.start_run(run_name=run_dir.name):
            mlflow.log_params({k: str(v)[:250] for k, v in flatten(cfg).items()})
            log_file = run_dir / "train_log.csv"
            if log_file.exists():
                with open(log_file) as f:
                    for row in csv.DictReader(f):
                        step = int(row["iter"])
                        mlflow.log_metrics(clean({f"train_{k}": float(v) for k, v in row.items()
                                                  if k not in ("iter", "epoch")}), step=step)
            mlflow.log_metrics(clean(json.loads((run_dir / "metrics.json").read_text())))
            for f in ("config.yaml", "metrics.json", "train_log.csv"):
                if (run_dir / f).exists():
                    mlflow.log_artifact(str(run_dir / f))

    mlflow.set_experiment("roadsight-evaluation")
    for res in sorted(Path(a.results).rglob("*.json")):
        data = json.loads(res.read_text())
        if not isinstance(data, dict):
            continue
        with mlflow.start_run(run_name=str(res.relative_to(a.results).with_suffix(""))):
            if "models" in data and isinstance(data["models"], list):  # latency benchmark file
                mlflow.log_params({k: str(v) for k, v in flatten(data.get("protocol", {})).items()})
                mlflow.log_params({f"machine.{k}": str(v) for k, v in data.get("machine", {}).items()})
                for m in data["models"]:
                    mlflow.log_metrics(clean({f"{m['name']}.{k}": v for k, v in m.items()}))
            else:
                mlflow.log_params({k: str(v)[:250] for k, v in data.items() if not isinstance(v, (int, float, dict))
                                   or isinstance(v, bool)})
                mlflow.log_metrics(clean({k: v for k, v in flatten(data).items()}))
            mlflow.log_artifact(str(res))
    print("wrote", a.store)


if __name__ == "__main__":
    main()
