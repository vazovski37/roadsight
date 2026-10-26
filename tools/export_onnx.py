"""Export a trained run to ONNX at a fixed input size and check it against PyTorch.

    python tools/export_onnx.py runs/r18_lite --size 640 360 --out models/r18_lite_640.onnx

The graph is DeployModel: uint8-range image in, semantic labels + per-location scores/boxes out.
Decoding/NMS/fusion stay in numpy (roadsight.postprocess), shared with the evaluation code.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roadsight.engine import load_run  # noqa: E402
from roadsight.labels import load_image, read_split, resize_image  # noqa: E402
from roadsight.model import DeployModel  # noqa: E402


def export(run, size, out, data_root):
    net, cfg = load_run(run, "cpu")
    deploy = DeployModel(net).eval()
    w, h = size
    dummy = torch.zeros(1, 3, h, w)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(deploy, dummy, str(out), input_names=["image"], output_names=deploy.output_names,
                      opset_version=17, dynamo=False)

    # check: same image through PyTorch and ONNX Runtime
    name = read_split(data_root, "val")[0]
    img = resize_image(load_image(data_root, name), w, h)
    x = torch.from_numpy(img).permute(2, 0, 1)[None].float()
    with torch.no_grad():
        ref = [t.numpy() for t in deploy(x)]
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"image": x.numpy()})
    check = {}
    for n, a, b in zip(deploy.output_names, ref, got):
        if n == "sem":
            check["sem_pixel_agreement"] = float((a == b).mean())
        else:
            check[f"{n}_max_abs_diff"] = float(np.abs(a - b).max())
    meta = {"run": str(run), "name": cfg["name"], "size": [w, h], "outputs": deploy.output_names,
            "tasks": list(net.tasks), "fusion": cfg["eval"]["fusion"], "check": check,
            "params_M": sum(p.numel() for p in net.parameters()) / 1e6}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(out, json.dumps(check))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--size", type=int, nargs=2, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default="data/bdd10k")
    a = ap.parse_args()
    export(a.run, a.size, a.out, a.data_root)
