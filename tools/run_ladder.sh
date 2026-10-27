#!/usr/bin/env bash
# The CPU optimisation ladder, run on the target machine after training:
#   export to ONNX -> FP16 / INT8 variants -> accuracy on val (CPU) -> latency (4 pinned P-cores).
#
#   PY=../.venv/Scripts/python DATA=../data/bdd10k CHOSEN=r18_lite_kd bash tools/run_ladder.sh
#
# Every model gets a <name>.onnx + <name>.json pair in models/; accuracy goes to results/onnx/,
# latency to results/latency.json. Re-running skips models that already exist.
set -euo pipefail
PY=${PY:-python}
DATA=${DATA:-data/bdd10k}
CHOSEN=${CHOSEN:-r18_lite_kd}   # the student that gets quantised (picked from the GPU results)
M=models
mkdir -p "$M" results/onnx

export_run() {  # run W H name
  [ -f "$M/$4.onnx" ] || $PY tools/export_onnx.py "runs/$1" --size "$2" "$3" --out "$M/$4.onnx" --data-root "$DATA"
}

# step 0 -> 1: the teacher in PyTorch vs ONNX Runtime; step 2: the same weights at lower input sizes
export_run teacher_r50 1280 720 teacher_r50_1280
export_run teacher_r50 960 540 teacher_r50_960
export_run teacher_r50 640 360 teacher_r50_640
# step 2 (trained at that size) and step 3 (smaller models)
export_run r18_960 960 540 r18_960
export_run r18_640 640 360 r18_640
export_run r18_512 512 288 r18_512
export_run r50_640_coco 640 360 r50_640_coco
export_run r50_640_kd 640 360 r50_640_kd
export_run r18_lite 640 360 r18_lite
export_run r18_lite_p3 640 360 r18_lite_p3
export_run mbv3_lite 640 360 mbv3_lite
# step 4: distilled students; step 6b: pruned
export_run r18_lite_kd 640 360 r18_lite_kd
export_run mbv3_lite_kd 640 360 mbv3_lite_kd
export_run r18_lite_prune50 640 360 r18_lite_prune50

# step 5 / 6a: quantise the chosen student (FP16, INT8, INT8 with FP32 prediction layers)
[ -f "$M/${CHOSEN}_int8.onnx" ] || $PY tools/quantize.py "$M/$CHOSEN.onnx" --data-root "$DATA"
# INT8 also for the full-width models: the heads, not the backbone, dominate latency, so a
# quantised R50 student can still fit the budget
for m in r18_640 r50_640_coco r50_640_kd r18_lite mbv3_lite_kd; do
  [ -f "$M/${m}_int8.onnx" ] || [ "$m" = "$CHOSEN" ] || $PY tools/quantize.py "$M/$m.onnx" --data-root "$DATA" --skip-fp16
done

# accuracy of every ONNX variant on val (FP16 is evaluated separately: it may be impractically slow)
MODELS=$(ls $M/*.onnx | grep -v -e _fp16 -e _prep)
for m in $MODELS; do
  n=$(basename "$m" .onnx)
  [ -f "results/onnx/${n}_val.json" ] || $PY tools/onnx_eval.py "$m" --data-root "$DATA" --split val
done

# latency: PyTorch eager reference for step 0, then every ONNX model
$PY tools/benchmark.py runs/teacher_r50 --torch-size 1280 720 --data-root "$DATA" --frames 50
$PY tools/benchmark.py runs/$CHOSEN --torch-size 640 360 --data-root "$DATA"
$PY tools/benchmark.py $MODELS --data-root "$DATA"
