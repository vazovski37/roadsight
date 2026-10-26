"""Download and convert BDD100K 10K segmentation data into the layout roadsight.data expects.

Sources (the official ETH mirror, dl.cv.ethz.ch, no longer resolves as of 2026-10):
  * semantic masks (19 classes): Berkeley mirror http://dl.yf.io/bdd100k/bdd100k_seg_maps.zip
  * images + instance bitmasks: Hugging Face dataset adhisetiawan/bdd10k-bitmasks
    (parquet with the original BDD100K file names and RGBA bitmasks, 7K train / 1K val)
Images present in both sources are kept; the rest are counted in stats.json.

Also writes EDA statistics (class frequencies, object sizes, same-class box overlap,
agreement between the two label sources) to <out>/stats.json.

    python tools/prepare_data.py --raw data/raw --out data/bdd10k --download
"""
import argparse
import json
import random
import sys
import urllib.request
import zipfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roadsight.classes import INS_CAT_TO_DET, NUM_SEM, det_to_sem  # noqa: E402
from roadsight.labels import decode_bitmask  # noqa: E402

HF = "https://huggingface.co/datasets/adhisetiawan/bdd10k-bitmasks/resolve/main/data/"
PARQUETS = {
    "train": ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"],
    "val": ["validation-00000-of-00001.parquet"],
}
SEG_ZIP = "http://dl.yf.io/bdd100k/bdd100k_seg_maps.zip"


def download(url, dest):
    if dest.exists():
        return
    print("downloading", url)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def box_iou(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None] - inter)


def process(job):
    """Write one image's files and derive its box annotations + statistics."""
    out, name, img_bytes, mask_bytes, sem_bytes = job
    out = Path(out)
    (out / "images" / f"{name}.jpg").write_bytes(img_bytes)
    (out / "inst" / f"{name}.png").write_bytes(mask_bytes)
    (out / "sem" / f"{name}.png").write_bytes(sem_bytes)

    import io
    bitmask = np.asarray(Image.open(io.BytesIO(mask_bytes)))
    sem = np.asarray(Image.open(io.BytesIO(sem_bytes)))
    category, attributes, instance = decode_bitmask(bitmask)

    boxes, labels, crowd, areas = [], [], [], []
    agree = np.zeros(2, np.int64)  # [instance pixels whose semantic label matches, all instance pixels]
    things = category > 0
    if things.any():
        ids, inverse = np.unique(instance[things], return_inverse=True)
        relabeled = np.zeros(instance.shape, np.int32)
        relabeled[things] = inverse + 1
        pixel_count = np.bincount(relabeled.ravel(), minlength=len(ids) + 1)
        for k, sl in enumerate(ndimage.find_objects(relabeled), start=1):
            if sl is None:
                continue
            m = relabeled[sl] == k
            cat = int(category[sl][m][0])
            attr = int(attributes[sl][m][0])
            det = INS_CAT_TO_DET[cat]
            # continuous coordinates: pixel columns x1..x2 cover [x1, x2 + 1)
            boxes.append([sl[1].start, sl[0].start, sl[1].stop, sl[0].stop])
            labels.append(det)
            crowd.append(bool(attr & 0b11))
            areas.append(int(pixel_count[k]))
            agree += [int((sem[sl][m] == det_to_sem(det)).sum()), int(m.sum())]

    # same-class pairs of (non-crowd) boxes that overlap with IoU > 0.3: the fusion's weak spot
    overlap_pairs = 0
    b = np.asarray(boxes, np.float32).reshape(-1, 4)
    lab = np.asarray(labels)
    keep = ~np.asarray(crowd, bool)
    for c in np.unique(lab[keep]) if keep.any() else []:
        bc = b[keep & (lab == c)]
        if len(bc) > 1:
            iou = box_iou(bc, bc)
            overlap_pairs += int((np.triu(iou, 1) > 0.3).sum())

    stats = {
        "sem_pixels": np.bincount(sem.ravel(), minlength=256)[:NUM_SEM],
        "agree": agree,
        "overlap_pairs": overlap_pairs,
    }
    ann = {"boxes": boxes, "labels": labels, "crowd": crowd, "areas": areas}
    return name, ann, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/bdd10k")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--holdout", type=int, default=500, help="train images held out for tuning")
    ap.add_argument("--limit", type=int, default=0, help="debug: images per split")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    raw.mkdir(parents=True, exist_ok=True)
    for d in ("images", "inst", "sem", "splits"):
        (out / d).mkdir(parents=True, exist_ok=True)
    if args.download:
        download(SEG_ZIP, raw / "bdd100k_seg_maps.zip")
        for split in args.splits:
            for f in PARQUETS[split]:
                download(HF + f, raw / f"ins_{f}")

    seg_zip = zipfile.ZipFile(raw / "bdd100k_seg_maps.zip")
    sem_names = {
        split: {n.split("/")[-1][: -len("_train_id.png")]: n
                for n in seg_zip.namelist() if n.startswith(f"labels/{split}/") and n.endswith(".png")}
        for split in ("train", "val")
    }

    annotations, split_names = {}, defaultdict(list)
    totals = {"sem_pixels": np.zeros(NUM_SEM, np.int64), "agree": np.zeros(2, np.int64), "overlap_pairs": 0}
    missing = defaultdict(int)
    with Pool(args.workers) as pool:
        for split in args.splits:
            def jobs():
                count = 0
                for f in PARQUETS[split]:
                    pf = pq.ParquetFile(raw / f"ins_{f}")
                    for g in range(pf.num_row_groups):
                        t = pf.read_row_group(g, columns=["image", "label"])
                        for img, lab in zip(t.column("image").to_pylist(), t.column("label").to_pylist()):
                            name = Path(img["path"]).stem
                            if name not in sem_names[split]:
                                missing[split] += 1
                                continue
                            yield str(out), name, img["bytes"], lab["bytes"], seg_zip.read(sem_names[split][name])
                            count += 1
                            if args.limit and count >= args.limit:
                                return

            for name, ann, stats in pool.imap_unordered(process, jobs(), chunksize=16):
                annotations[name] = ann
                split_names[split].append(name)
                for k in totals:
                    totals[k] = totals[k] + stats[k]
            print(f"{split}: {len(split_names[split])} images, {missing[split]} without semantic mask")

    names_train = sorted(split_names.get("train", []))
    random.Random(0).shuffle(names_train)
    splits = {
        "holdout": sorted(names_train[: args.holdout]),
        "train": sorted(names_train[args.holdout:]),
        "val": sorted(split_names.get("val", [])),
    }
    for s, names in splits.items():
        (out / "splits" / f"{s}.txt").write_text("\n".join(names) + "\n")
    with open(out / "annotations.json", "w") as f:
        json.dump(annotations, f)

    # ---- EDA ----
    inst_count = np.zeros(8, np.int64)
    small = np.zeros(8, np.int64)
    per_image = []
    for ann in annotations.values():
        n = 0
        for lab, cr, area in zip(ann["labels"], ann["crowd"], ann["areas"]):
            if cr:
                continue
            inst_count[lab] += 1
            small[lab] += area < 32 * 32
            n += 1
        per_image.append(n)
    from roadsight.classes import DET_CLASSES, SEM_CLASSES
    stats = {
        "images": {s: len(v) for s, v in splits.items()},
        "missing_semantic_mask": dict(missing),
        "semantic_pixel_share": dict(zip(SEM_CLASSES, (totals["sem_pixels"] / totals["sem_pixels"].sum()).round(5).tolist())),
        "instances": dict(zip(DET_CLASSES, inst_count.tolist())),
        "small_instance_share(<32^2 px)": dict(zip(DET_CLASSES, (small / np.maximum(inst_count, 1)).round(3).tolist())),
        "instances_per_image": {"mean": float(np.mean(per_image)), "median": float(np.median(per_image)),
                                "max": int(np.max(per_image))},
        "same_class_box_pairs_iou>0.3_per_image": totals["overlap_pairs"] / max(len(annotations), 1),
        "instance_pixels_matching_semantic_label": float(totals["agree"][0] / max(totals["agree"][1], 1)),
    }
    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
