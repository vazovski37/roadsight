"""Training loop (multi-task, optional distillation / pruning) and evaluation."""
import csv
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from .classes import IGNORE, NUM_DET
from .config import flatten
from .data import BDDPanoptic, ScaleBatchSampler, collate
from .fcos import fcos_loss
from .model import DeployModel, build_model, flatten_levels
from .scoring import Accumulator, eval_image


# --------------------------------------------------------------------------- tracking

class Tracker:
    """Thin MLflow wrapper; training still works (and writes metrics.json) if MLflow is missing."""

    def __init__(self, name, cfg):
        self.mlflow = None
        if os.environ.get("ROADSIGHT_MLFLOW", "1") == "0":
            return
        try:
            import mlflow
            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///runs/mlflow.db"))
            mlflow.set_experiment("roadsight")
            mlflow.start_run(run_name=name)
            mlflow.log_params({k: str(v)[:250] for k, v in flatten(cfg).items()})
            self.mlflow = mlflow
        except Exception as e:  # tracking must never kill a training run
            print(f"[tracker] MLflow disabled: {e}")

    def _safe(self, fn, *args, **kwargs):
        if self.mlflow:
            try:
                fn(*args, **kwargs)
            except Exception as e:  # e.g. sqlite locked by a parallel run
                print(f"[tracker] {fn.__name__} failed: {e}")

    def metrics(self, values, step=None):
        clean = {k.replace("/", "_").replace(" ", "_").replace("^", ""): float(v) for k, v in values.items()}
        self._safe(self.mlflow and self.mlflow.log_metrics, clean, step=step)

    def artifact(self, path):
        self._safe(self.mlflow and self.mlflow.log_artifact, str(path))

    def close(self):
        self._safe(self.mlflow and self.mlflow.end_run)


# --------------------------------------------------------------------------- helpers

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_run(run_dir, device="cpu"):
    """Rebuild a trained model from runs/<name>/ (config.yaml + model.pt)."""
    cfg = yaml.safe_load(Path(run_dir, "config.yaml").read_text())
    net = build_model({**cfg, "model": {**cfg["model"], "pretrained": "none"}})
    state = torch.load(Path(run_dir, "model.pt"), map_location="cpu")
    net.load_state_dict(state)
    return net.to(device).eval(), cfg


def param_groups(model, weight_decay, extra=()):
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay + list(extra), "weight_decay": 0.0}]


def lr_lambda(warmup, total):
    def f(it):
        if it < warmup:
            return 0.001 + (1 - 0.001) * it / warmup
        progress = (it - warmup) / max(total - warmup, 1)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))
    return f


def distill_loss(out, t_out, temperature):
    """Dense distillation: semantic KL on softened logits + FCOS class soft targets.

    The detection term is weighted per location by the teacher's max class probability so the
    thousands of easy background locations do not drown the signal.
    """
    losses = {}
    if "sem" in out:
        s = out["sem"].float()
        t = F.interpolate(t_out["sem"].float(), size=s.shape[-2:], mode="bilinear", align_corners=False)
        kl = F.kl_div(F.log_softmax(s / temperature, 1), F.softmax(t / temperature, 1), reduction="none")
        losses["kd_sem"] = kl.sum(1).mean() * temperature ** 2
    if "cls" in out:
        s = flatten_levels(out["cls"]).float()
        p_t = flatten_levels(t_out["cls"]).float().sigmoid()
        w = p_t.max(-1).values
        bce = F.binary_cross_entropy_with_logits(s, p_t, reduction="none").sum(-1)
        losses["kd_det"] = (w * bce).sum() / w.sum().clamp(min=1.0)
    return losses


# --------------------------------------------------------------------------- training

def train(cfg, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    tcfg, dcfg = cfg["train"], cfg["data"]
    seed_everything(tcfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    amp = device.type == "cuda" and tcfg.get("amp", "bf16") == "bf16"

    net = build_model(cfg)
    if tcfg.get("init"):
        net.load_state_dict(torch.load(Path(tcfg["init"], "model.pt"), map_location="cpu"))

    pruned = []
    if tcfg.get("prune"):  # global magnitude pruning; masks stay fixed while fine-tuning
        import torch.nn.utils.prune as prune
        pruned = [(m, "weight") for m in net.modules() if isinstance(m, nn.Conv2d)]
        prune.global_unstructured(pruned, pruning_method=prune.L1Unstructured, amount=tcfg["prune"]["amount"])
    net = net.to(device).to(memory_format=torch.channels_last)

    teacher = None
    if tcfg.get("distill"):
        teacher, _ = load_run(tcfg["distill"]["teacher"], device)
        teacher = teacher.to(memory_format=torch.channels_last)
        for p in teacher.parameters():
            p.requires_grad_(False)

    ds = BDDPanoptic(dcfg["root"], dcfg.get("train_split", "train"), dcfg["size"], train=True)
    sampler = ScaleBatchSampler(len(ds), dcfg["batch_size"], dcfg["train_scale"], seed=tcfg["seed"])
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=dcfg["workers"], collate_fn=collate,
                        pin_memory=device.type == "cuda", persistent_workers=dcfg["workers"] > 0,
                        prefetch_factor=4 if dcfg["workers"] > 0 else None)

    lcfg = tcfg["loss"]
    log_vars = nn.Parameter(torch.zeros(2, device=device)) if lcfg["weighting"] == "uncertainty" else None
    opt = torch.optim.AdamW(param_groups(net, tcfg["weight_decay"], [log_vars] if log_vars is not None else []),
                            lr=tcfg["lr"])
    total_iters = tcfg["epochs"] * len(loader)
    if tcfg.get("max_iters"):
        total_iters = min(total_iters, tcfg["max_iters"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(tcfg["warmup_iters"], total_iters))

    tracker = Tracker(cfg["name"], cfg)
    log_file = open(out_dir / "train_log.csv", "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["iter", "epoch", "lr", "loss", "det", "sem", "kd", "img_per_s"])
    it, t0, seen = 0, time.time(), 0
    net.train()
    for epoch in range(tcfg["epochs"]):
        sampler.set_epoch(epoch)
        for images, sem, targets, _ in loader:
            images = images.to(device, non_blocking=True).float().contiguous(memory_format=torch.channels_last)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                out = net(images)
                t_out = None
                if teacher is not None:
                    with torch.no_grad():
                        t_out = teacher(images)

            losses = {}
            if "cls" in out:
                losses.update(fcos_loss(out, targets, NUM_DET))
            if "sem" in out:
                logits = F.interpolate(out["sem"].float(), size=images.shape[-2:], mode="bilinear", align_corners=False)
                losses["sem"] = F.cross_entropy(logits, sem.to(device, non_blocking=True).long(), ignore_index=IGNORE)
            det = sum(v for k, v in losses.items() if k.startswith("det_")) if "cls" in out else 0.0
            sem_l = losses.get("sem", 0.0)
            if log_vars is not None:  # Kendall et al. 2018: learned task weights
                total = torch.exp(-log_vars[0]) * det + 0.5 * log_vars[0] + torch.exp(-log_vars[1]) * sem_l + 0.5 * log_vars[1]
            else:
                total = det + lcfg["sem_weight"] * sem_l
            kd = 0.0
            if teacher is not None:
                kd_terms = distill_loss(out, t_out, tcfg["distill"]["temperature"])
                kd = tcfg["distill"]["sem_weight"] * kd_terms.get("kd_sem", 0.0) + \
                    tcfg["distill"]["det_weight"] * kd_terms.get("kd_det", 0.0)
                total = total + kd

            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], tcfg["clip_grad"])
            opt.step()
            sched.step()
            it += 1
            seen += len(images)
            if it % 50 == 0 or it == 1:
                speed = seen / (time.time() - t0)
                total, det, sem_l, kd = (float(v.detach()) if torch.is_tensor(v) else float(v) for v in (total, det, sem_l, kd))
                row = [it, epoch, opt.param_groups[0]["lr"], total, det, sem_l, kd, speed]
                log.writerow([f"{v:.5g}" if isinstance(v, float) else v for v in row])
                log_file.flush()
                print(f"[{cfg['name']}] it {it}/{total_iters} ep {epoch} loss {total:.3f} "
                      f"det {det:.3f} sem {sem_l:.3f} kd {kd:.3f} {speed:.0f} img/s", flush=True)
                tracker.metrics({"loss": total, "loss_det": det, "loss_sem": sem_l, "loss_kd": kd}, step=it)
                seen, t0 = 0, time.time()
            if it >= total_iters:
                break
        if it >= total_iters:
            break
    log_file.close()

    if pruned:
        import torch.nn.utils.prune as prune
        for m, name in pruned:
            prune.remove(m, name)
    net = net.to(memory_format=torch.contiguous_format)
    torch.save(net.state_dict(), out_dir / "model.pt")
    if log_vars is not None:
        tracker.metrics({"log_var_det": float(log_vars[0]), "log_var_sem": float(log_vars[1])})

    metrics = {"params_M": sum(p.numel() for p in net.parameters()) / 1e6}
    ecfg = cfg["eval"]
    for split in ecfg["splits"]:
        dump = out_dir / "dumps" / split if ecfg.get("dump") else None
        m = evaluate([net], cfg, split, dump_dir=dump)
        metrics.update({f"{split}/{k}": v for k, v in m.items()})
        print(f"[{cfg['name']}] {split}: " + ", ".join(f"{k} {v:.2f}" for k, v in m.items()
                                                       if "/" not in k), flush=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    tracker.metrics(metrics)
    tracker.artifact(out_dir / "metrics.json")
    tracker.artifact(out_dir / "config.yaml")
    tracker.close()
    return metrics


# --------------------------------------------------------------------------- evaluation

@torch.no_grad()
def run_models(nets, images, amp=True):
    """Merge the deploy outputs of one or more models (E1 combines a det-only and a sem-only model)."""
    merged = {}
    for net in nets:
        deploy = DeployModel(net).eval()
        with torch.autocast(images.device.type, dtype=torch.bfloat16, enabled=amp and images.device.type == "cuda"):
            outs = deploy(images)
        merged.update(dict(zip(deploy.output_names, outs)))
    return merged


@torch.no_grad()
def evaluate(nets, cfg, split, size=None, dump_dir=None, workers=None):
    dcfg, ecfg = cfg["data"], cfg["eval"]
    size = size or dcfg["size"]
    device = next(nets[0].parameters()).device
    for n in nets:
        n.eval()
    ds = BDDPanoptic(dcfg["root"], split, size, train=False)
    loader = DataLoader(ds, batch_size=8, num_workers=min(dcfg["workers"], 8), collate_fn=collate)
    if dump_dir is not None:
        Path(dump_dir).mkdir(parents=True, exist_ok=True)
    det_cfg = {k: ecfg[k] for k in ("score_thr", "nms_iou", "max_det")}
    acc = Accumulator()
    workers = workers or int(os.environ.get("ROADSIGHT_EVAL_WORKERS", min(os.cpu_count() or 4, 16)))
    with ProcessPoolExecutor(workers) as pool:
        futures = []
        for images, _, _, names in loader:
            outs = run_models(nets, images.to(device).float())
            outs = {k: v.float().cpu().numpy() if k != "sem" else v.cpu().numpy() for k, v in outs.items()}
            for i, name in enumerate(names):
                o = {k: v[i] for k, v in outs.items()}
                if "scores" in o:  # only ship candidate locations to the worker
                    keep = o["scores"].max(1) > det_cfg["score_thr"]
                    o["scores"], o["boxes"] = o["scores"][keep], o["boxes"][keep]
                futures.append(pool.submit(eval_image, dcfg["root"], name, o, size, det_cfg, ecfg["fusion"],
                                           None if dump_dir is None else str(dump_dir)))
        for f in futures:
            acc.add(f.result())
    return acc.metrics(ds.names, ds.ann)
