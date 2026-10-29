#!/usr/bin/env bash
# The CPU optimisation ladder, run on the target machine after training:
#   export to ONNX -> INT8 variants -> accuracy on val (CPU) -> latency (4 pinned P-cores)
#   -> the deployed model: fusion tuned on holdout, error analysis, its own latency.
#
#   PY=../.venv/Scripts/python DATA=../data/bdd10k bash tools/run_ladder.sh
#
# Every model gets a <name>.onnx + <name>.json pair in models/; accuracy goes to results/onnx/,
# latency to results/latency.json. Re-running skips models that already exist.
# FP16 is skipped: onnxconverter_common needed 17 GB of RAM for the ResNet-50 models.
set -euo pipefail
PY=${PY:-python}
DATA=${DATA:-data/bdd10k}
CHOSEN=${CHOSEN:-r50_lite_kd_800}   # highest PQ within 100 ms after INT8 (see docs/REPORT.ka.md, 3.5)
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
export_run r18_det 640 360 r18_det
export_run r18_sem 640 360 r18_sem
export_run r50_640_coco 640 360 r50_640_coco
export_run r50_640_kd 640 360 r50_640_kd
export_run r18_lite 640 360 r18_lite
export_run r18_lite_p3 640 360 r18_lite_p3
export_run mbv3_lite 640 360 mbv3_lite
export_run small 640 360 small
# step 4: distilled students (R50 with slim heads also at 800x450); step 6b: pruned
export_run r18_lite_kd 640 360 r18_lite_kd
export_run mbv3_lite_kd 640 360 mbv3_lite_kd
export_run r50_lite_kd 640 360 r50_lite_kd
export_run r50_lite_kd_800 800 450 r50_lite_kd_800
export_run r18_lite_prune50 640 360 r18_lite_prune50

# step 5: INT8 (naive, decode kept float, prediction convs float too). Also for the full-width models:
# the heads, not the backbone, dominate latency, so a quantised R50 student can still fit the budget
for m in r18_640 r50_640_coco r50_640_kd r18_lite r18_lite_kd mbv3_lite mbv3_lite_kd r50_lite_kd r50_lite_kd_800; do
  [ -f "$M/${m}_int8.onnx" ] || $PY tools/quantize.py "$M/$m.onnx" --data-root "$DATA" --skip-fp16
done

# accuracy of every ONNX variant on val
MODELS=$(ls $M/*.onnx | grep -v -e _fp16 -e _prep -e roadsight_)
for m in $MODELS; do
  n=$(basename "$m" .onnx)
  [ -f "results/onnx/${n}_val.json" ] || $PY tools/onnx_eval.py "$m" --data-root "$DATA" --split val
done

# latency: PyTorch eager reference for step 0, then every ONNX model (all with the same protocol)
$PY tools/benchmark.py runs/teacher_r50 --torch-size 1280 720 --data-root "$DATA"
$PY tools/benchmark.py runs/$CHOSEN --torch-size 800 450 --data-root "$DATA"
$PY tools/benchmark.py $MODELS --data-root "$DATA"

# the deployed model: tune the fusion thresholds on holdout (reported on val), write them into its .json
D=roadsight_int8
cp "$M/${CHOSEN}_int8.onnx" "$M/$D.onnx"
cp "$M/${CHOSEN}_int8.json" "$M/$D.json"
for s in holdout val; do $PY tools/onnx_eval.py "$M/$D.onnx" --data-root "$DATA" --split $s --dump "runs_onnx/$D"; done
$PY tools/tune_fusion.py "runs_onnx/$D" --data-root "$DATA" --out "results/fusion_$D.json"
$PY - "$M/$D.json" "results/fusion_$D.json" "$M/${CHOSEN}_int8.onnx" <<'EOF'
import json, sys
meta_path, tune_path, source = sys.argv[1:]
meta = json.load(open(meta_path))
meta.update(fusion=json.load(open(tune_path))["best"], deployed_from=source,
            fusion_source=f"{tune_path} (tuned on holdout)")
open(meta_path, "w").write(json.dumps(meta, indent=2))
EOF
$PY tools/onnx_eval.py "$M/$D.onnx" --data-root "$DATA" --split val
$PY tools/error_analysis.py "runs_onnx/$D" --data-root "$DATA" --fusion "results/fusion_$D.json" --out "results/analysis/$D"
$PY tools/benchmark.py "$M/$D.onnx" --data-root "$DATA" --out results/latency_deployed.json
