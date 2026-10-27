"""Make FP16 and INT8 variants of an exported ONNX model (optimisation ladder steps 5 and 6a).

    python tools/quantize.py models/r18_lite_640.onnx --calib-split holdout

Writes next to the input:
  *_fp16.onnx        weights and activations in float16 (inputs/outputs stay float32)
  *_int8naive.onnx   static post-training quantisation of every supported op, QDQ format,
                     per-channel weights (U8/S8) - kept as the documented failure case
  *_int8.onnx        same, but the decode subgraph after the prediction convs stays float32
  *_int8mix.onnx     same as int8, and the prediction convs themselves stay float32 too
Calibration uses training-side images (holdout split), never the val split we report on.

Why int8naive fails: without an op filter ORT also quantises the decode ops (Concat/Add/Sub/Mul
after the heads). Box coordinates of all levels share one uint8 scale over 0..W pixels, i.e.
~2.5 px steps at 640x360, which destroys small boxes (AP_S halves) while PQ barely moves.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roadsight.labels import load_image, read_split, resize_image  # noqa: E402

# final convs that turn features into predictions: most sensitive to quantisation error
SENSITIVE = ("cls_logits", "bbox_reg", "bbox_ctrness", "sem_head/predictor")


class ImageReader(CalibrationDataReader):
    def __init__(self, root, split, size, count):
        names = read_split(root, split)[:count]
        self.batches = iter([{"image": resize_image(load_image(root, n), *size).transpose(2, 0, 1)[None]
                              .astype(np.float32)} for n in names])

    def get_next(self):
        return next(self.batches, None)


def to_fp16(src, dst):
    from onnxconverter_common import float16
    # the graph's explicit casts (to float32 / uint8) and the argmax must stay as they are,
    # otherwise the converter produces type-inconsistent Cast nodes
    model = float16.convert_float_to_float16(onnx.load(src), keep_io_types=True, op_block_list=["Cast", "ArgMax"])
    onnx.save(model, dst)


def downstream(graph, roots):
    """Names of every node fed (directly or not) by the outputs of the `roots` nodes."""
    consumers = {}
    for n in graph.node:
        for t in n.input:
            consumers.setdefault(t, []).append(n)
    seen, stack = set(), [t for n in graph.node if n.name in roots for t in n.output]
    while stack:
        for n in consumers.get(stack.pop(), []):
            if n.name not in seen:
                seen.add(n.name)
                stack.extend(n.output)
    return seen


def to_int8(src, dst, reader_fn, mode, method):
    """mode: 'naive' quantises everything, 'int8' keeps the decode ops float, 'mix' also the prediction convs."""
    prep = str(dst).replace(".onnx", "_prep.onnx")
    quant_pre_process(str(src), prep, skip_symbolic_shape=True)
    graph = onnx.load(prep).graph
    heads = {n.name for n in graph.node if any(s in n.name for s in SENSITIVE)}
    exclude = set()
    if mode in ("int8", "mix"):
        exclude |= downstream(graph, heads)
    if mode == "mix":
        exclude |= heads
    exclude = sorted(exclude)
    quantize_static(prep, str(dst), reader_fn(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=method, nodes_to_exclude=exclude,
                    extra_options={"ActivationSymmetric": False, "WeightSymmetric": True})
    Path(prep).unlink()
    return exclude


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--data-root", default="data/bdd10k")
    ap.add_argument("--calib-split", default="holdout")
    ap.add_argument("--calib-count", type=int, default=128)
    ap.add_argument("--method", default="minmax", choices=["minmax", "percentile", "entropy"])
    ap.add_argument("--skip-fp16", action="store_true")
    a = ap.parse_args()

    src = Path(a.model)
    import json
    size = json.loads(src.with_suffix(".json").read_text())["size"]
    method = {"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile,
              "entropy": CalibrationMethod.Entropy}[a.method]
    reader_fn = lambda: ImageReader(a.data_root, a.calib_split, size, a.calib_count)  # noqa: E731
    stem = src.with_suffix("")
    if not a.skip_fp16:
        to_fp16(src, f"{stem}_fp16.onnx")
        print("wrote", f"{stem}_fp16.onnx")
    for mode, suffix in (("naive", "_int8naive"), ("int8", "_int8"), ("mix", "_int8mix")):
        excluded = to_int8(src, Path(f"{stem}{suffix}.onnx"), reader_fn, mode, method)
        print("wrote", f"{stem}{suffix}.onnx", f"({len(excluded)} nodes kept float32)")
    for suffix in ("_fp16", "_int8naive", "_int8", "_int8mix"):
        p = Path(f"{stem}{suffix}.onnx")
        if p.exists():
            meta = json.loads(src.with_suffix(".json").read_text())
            meta["variant"] = suffix[1:]
            p.with_suffix(".json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
