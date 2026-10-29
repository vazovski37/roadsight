"""CPU latency benchmark for exported ONNX models (the deployment target).

    python tools/benchmark.py models/*.onnx --threads 4 --cpus 0 2 4 6 --out results/latency.json

Protocol (fixed before measuring):
  * ONNX Runtime CPU execution provider, batch 1, intra-op threads = --threads
  * the process is pinned to --cpus (on the reference i9-14900KF: one hardware thread on each of
    four performance cores, so Windows cannot move work onto the slower efficiency cores)
  * frames are val images, already decoded at 1280x720 (a camera hands over decoded frames)
  * 20 warm-up frames, then --frames timed frames in a fixed order; median and p90 per stage:
      pre   = resize + layout conversion          model = session.run
      post  = detection decoding/NMS + panoptic fusion + text summary
  * every model runs in its own subprocess, so peak memory belongs to that model
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roadsight.labels import load_image, read_split, resize_image  # noqa: E402
from roadsight.postprocess import decode_detections, fuse, summarize  # noqa: E402


def zipped_mb(path):
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(path, Path(path).name)
    return buf.tell() / 1e6


def torch_runner(run, size, threads):
    """Ladder step 0: the trained PyTorch model in eager mode on CPU (same pre/post-processing)."""
    import torch
    from roadsight.engine import load_run
    from roadsight.model import DeployModel
    torch.set_num_threads(threads)
    net, cfg = load_run(run, "cpu")
    deploy = DeployModel(net).eval()
    meta = {"size": list(size), "outputs": deploy.output_names, "fusion": cfg["eval"]["fusion"]}

    def run_fn(x):
        with torch.inference_mode():
            return [t.numpy() for t in deploy(torch.from_numpy(x))]
    params = sum(p.numel() for p in net.parameters())
    return meta, run_fn, params * 4


def onnx_runner(model, threads):
    import onnxruntime as ort
    meta = json.loads(Path(model).with_suffix(".json").read_text())
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"])
    return meta, lambda x: sess.run(None, {"image": x}), os.path.getsize(model)


def bench_one(model, data_root, frames, threads, cpus, torch_size=None):
    import psutil
    proc = psutil.Process()
    if cpus:
        proc.cpu_affinity(cpus)
    if torch_size:
        meta, run_fn, nbytes = torch_runner(model, torch_size, threads)
    else:
        meta, run_fn, nbytes = onnx_runner(model, threads)
    w, h = meta["size"]
    names = read_split(data_root, "val")[: frames + 20]
    images = [load_image(data_root, n) for n in names]

    times = {"pre": [], "model": [], "post": [], "total": []}
    for i, img in enumerate(images):
        t0 = time.perf_counter()
        x = np.ascontiguousarray(resize_image(img, w, h).transpose(2, 0, 1)[None], dtype=np.float32)
        t1 = time.perf_counter()
        outs = dict(zip(meta["outputs"], run_fn(x)))
        t2 = time.perf_counter()
        if "scores" in outs:
            boxes, scores, labels = decode_detections(outs["scores"][0], outs["boxes"][0], (w, h))
            if "sem" in outs:
                cls_map, _, instances = fuse(outs["sem"][0], boxes, scores, labels, meta.get("fusion"))
                summarize(cls_map, instances)
        t3 = time.perf_counter()
        if i >= 20:  # first 20 frames are warm-up
            for k, seconds in zip(times, (t1 - t0, t2 - t1, t3 - t2, t3 - t0)):
                times[k].append(seconds * 1000)
    stats = {f"{k}_{q}_ms": float(np.percentile(v, p)) for k, v in times.items()
             for q, p in (("median", 50), ("p90", 90))}
    stats.update({
        "model": str(model), "name": (f"{Path(model).name}_torch_{w}" if torch_size else Path(model).stem),
        "runtime": "pytorch-eager" if torch_size else "onnxruntime", "size": meta["size"], "threads": threads,
        "cpus": cpus, "frames": frames, "file_mb": nbytes / 1e6,
        "zipped_mb": None if torch_size else zipped_mb(model),
        "peak_rss_mb": (getattr(proc.memory_info(), "peak_wset", 0) or proc.memory_info().rss) / 1e6,
    })
    return stats


def machine_info():
    info = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        import cpuinfo
        ci = cpuinfo.get_cpu_info()
        flags = set(ci.get("flags", []))
        info.update({"cpu": ci.get("brand_raw"), "logical_cpus": os.cpu_count(),
                     "avx2": "avx2" in flags, "avx512f": "avx512f" in flags,
                     # py-cpuinfo does not read CPUID leaf 7.1, where AVX-VNNI lives: absent means "unknown"
                     "avx_vnni": True if {"avxvnni", "avx_vnni"} & flags else "unknown",
                     "avx512_vnni": "avx512_vnni" in flags,
                     "avx512_fp16": "avx512_fp16" in flags})
    except Exception as e:
        info["cpuinfo_error"] = str(e)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*")
    ap.add_argument("--data-root", default="data/bdd10k")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cpus", type=int, nargs="*", default=[0, 2, 4, 6])
    ap.add_argument("--torch-size", type=int, nargs=2, default=None,
                    help="treat the models as run directories and time PyTorch eager at this size")
    ap.add_argument("--out", default="results/latency.json")
    ap.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.single:  # child process: one model, print JSON
        print("RESULT " + json.dumps(bench_one(a.models[0], a.data_root, a.frames, a.threads, a.cpus, a.torch_size)))
        return

    results = {"machine": machine_info(), "protocol": {"threads": a.threads, "cpus": a.cpus, "frames": a.frames,
                                                      "warmup": 20}, "models": []}
    out = Path(a.out)
    if out.exists():  # append/replace per model so the benchmark can be run in pieces
        old = json.loads(out.read_text())
        if old.get("protocol") != results["protocol"]:
            # numbers from different protocols must not share a table, and silently starting over loses them
            sys.exit(f"{out} was measured with {old.get('protocol')}; use the same settings or another --out")
        results["models"] = old["models"]
    for m in a.models:
        cmd = [sys.executable, __file__, m, "--single", "--data-root", a.data_root, "--frames", str(a.frames),
               "--threads", str(a.threads), "--cpus", *map(str, a.cpus)]
        if a.torch_size:
            cmd += ["--torch-size", *map(str, a.torch_size)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        line = [l for l in res.stdout.splitlines() if l.startswith("RESULT ")]
        if not line:
            print("FAILED", m, res.stderr[-2000:])
            continue
        r = json.loads(line[0][7:])
        results["models"] = [x for x in results["models"] if x["name"] != r["name"]] + [r]
        print(f"{r['name']:30s} total {r['total_median_ms']:7.1f} ms (p90 {r['total_p90_ms']:7.1f})  "
              f"model {r['model_median_ms']:7.1f}  pre {r['pre_median_ms']:5.1f}  post {r['post_median_ms']:5.1f}  "
              f"{r['file_mb']:6.1f} MB", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
