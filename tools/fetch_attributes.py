"""Scene attributes (time of day, weather, scene type) for our 10K images, for error analysis.

The 10K segmentation set has no attribute labels, but most of its images also appear in the 100K
detection set, whose per-image JSON files carry them. Images without a match get "unknown".

    python tools/fetch_attributes.py --out data/bdd10k/attributes.json
"""
import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roadsight.labels import read_split  # noqa: E402

URL = "http://dl.yf.io/bdd100k/bdd100k_labels.zip"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--data-root", default="data/bdd10k")
    ap.add_argument("--out", default="data/bdd10k/attributes.json")
    a = ap.parse_args()
    zpath = Path(a.raw, "bdd100k_labels.zip")
    if not zpath.exists():
        zpath.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, zpath)
    z = zipfile.ZipFile(zpath)
    index = {Path(n).stem: n for n in z.namelist() if n.endswith(".json")}
    names = sum((read_split(a.data_root, s) for s in ("train", "holdout", "val")), [])
    attrs, found = {}, 0
    for name in names:
        if name in index:
            meta = json.loads(z.read(index[name]))
            attrs[name] = meta.get("attributes", {})
            found += 1
        else:
            attrs[name] = {"timeofday": "unknown", "weather": "unknown", "scene": "unknown"}
    Path(a.out).write_text(json.dumps(attrs))
    print(f"attributes for {found}/{len(names)} images")
