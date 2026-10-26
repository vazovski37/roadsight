"""RoadSight demo API: upload a dashcam frame, get the panoptic overlay, numbered objects and a summary.

    ROADSIGHT_MODEL=models/roadsight_int8.onnx uvicorn app.main:app --port 8000

Runs the exported ONNX model with ONNX Runtime on CPU; no PyTorch needed.
"""
import base64
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roadsight.labels import resize_image, resize_labels  # noqa: E402
from roadsight.postprocess import colorize, decode_detections, fuse, scale_boxes, summarize  # noqa: E402

MAX_BYTES = 15 * 1024 * 1024
STATIC = Path(__file__).parent / "static"


class Predictor:
    def __init__(self, model_path, threads=4):
        self.meta = json.loads(Path(model_path).with_suffix(".json").read_text())
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.name = Path(model_path).stem

    def __call__(self, rgb):
        h0, w0 = rgb.shape[:2]
        w, h = self.meta["size"]
        t0 = time.perf_counter()
        x = np.ascontiguousarray(resize_image(rgb, w, h).transpose(2, 0, 1)[None], dtype=np.float32)
        t1 = time.perf_counter()
        outs = dict(zip(self.meta["outputs"], self.sess.run(None, {"image": x})))
        t2 = time.perf_counter()
        boxes, scores, labels = decode_detections(outs["scores"][0], outs["boxes"][0], (w, h))
        cls_map, inst_map, instances = fuse(outs["sem"][0], boxes, scores, labels, self.meta.get("fusion"))
        summary = summarize(cls_map, instances)
        t3 = time.perf_counter()

        # number objects per class in score order: car #1, car #2, ...
        per_class = {}
        for inst in instances:
            per_class[inst["label"]] = per_class.get(inst["label"], 0) + 1
            inst["name"] = f"{inst['label']} #{per_class[inst['label']]}"
            inst["box"] = [round(float(v), 1) for v in scale_boxes(np.array([inst["box"]], np.float32), (w, h), (w0, h0))[0]]
            inst["score"] = round(inst["score"], 3)
        timings = {"pre_ms": (t1 - t0) * 1000, "model_ms": (t2 - t1) * 1000, "post_ms": (t3 - t2) * 1000,
                   "total_ms": (t3 - t0) * 1000}
        return {"model": self.name, "model_input": [w, h], "image_size": [w0, h0], "summary": summary,
                "instances": instances, "timings": {k: round(v, 1) for k, v in timings.items()}}, cls_map, inst_map

    def overlay(self, rgb, cls_map, inst_map, instances):
        h0, w0 = rgb.shape[:2]
        cls_full = resize_labels(cls_map, w0, h0)
        inst_full = resize_labels(inst_map.astype(np.uint16), w0, h0)
        out = colorize(rgb, cls_full, inst_full, instances)
        for inst in instances:
            x1, y1, x2, y2 = (int(round(v)) for v in inst["box"])
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)
            label = inst["name"]
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, max(y1, th + 6)), (255, 255, 255), -1)
            cv2.putText(out, label, (x1 + 2, max(y1 - 4, th + 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                        cv2.LINE_AA)
        return out


MODEL = os.environ.get("ROADSIGHT_MODEL", "models/roadsight_int8.onnx")
predictor = Predictor(MODEL, int(os.environ.get("ROADSIGHT_THREADS", "4")))
app = FastAPI(title="RoadSight", description="Panoptic dashcam scene parsing on CPU")


def read_upload(data: bytes):
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image too large")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "could not decode image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model": predictor.name, "input": predictor.meta["size"]}


@app.post("/predict")
async def predict(file: UploadFile = File(...), overlay: bool = True):
    rgb = read_upload(await file.read())
    result, cls_map, inst_map = predictor(rgb)
    if overlay:
        png = predictor.overlay(rgb, cls_map, inst_map, result["instances"])
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(png, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        result["overlay_jpeg_base64"] = base64.b64encode(buf.tobytes()).decode()
    return result


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    rgb = read_upload(await file.read())
    result, cls_map, inst_map = predictor(rgb)
    out = predictor.overlay(rgb, cls_map, inst_map, result["instances"])
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(buf.tobytes(), media_type="image/jpeg",
                    headers={"X-Summary": result["summary"]["text"], "X-Total-Ms": str(result["timings"]["total_ms"])})
