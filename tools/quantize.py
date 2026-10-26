"""Make FP16 and INT8 variants of an exported ONNX model (optimisation ladder steps 5 and 6a).

    python tools/quantize.py models/r18_lite_640.onnx --calib-split holdout

Writes next to the input:
  *_fp16.onnx        weights and activations in float16 (inputs/outputs stay float32)
  *_int8.onnx        static post-training quantisation, QDQ format, per-channel weights (U8/S8)
  *_int8mix.onnx     same, but the final prediction convs stay float32 (mixed precision)
Calibration uses training-side images (holdout split), never the val split we report on.
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


def to_int8(src, dst, reader_fn, exclude_sensitive, method):
    prep = str(dst).replace(".onnx", "_prep.onnx")
    quant_pre_process(str(src), prep, skip_symbolic_shape=True)
    nodes = [n.name for n in onnx.load(prep).graph.node]
    exclude = [n for n in nodes if exclude_sensitive and any(s in n for s in SENSITIVE)]
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
    to_int8(src, Path(f"{stem}_int8.onnx"), reader_fn, False, method)
    print("wrote", f"{stem}_int8.onnx")
    excluded = to_int8(src, Path(f"{stem}_int8mix.onnx"), reader_fn, True, method)
    print("wrote", f"{stem}_int8mix.onnx", "(float32 kept for:", ", ".join(excluded) + ")")
    for suffix in ("_fp16", "_int8", "_int8mix"):
        p = Path(f"{stem}{suffix}.onnx")
        if p.exists():
            meta = json.loads(src.with_suffix(".json").read_text())
            meta["variant"] = suffix[1:]
            p.with_suffix(".json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
