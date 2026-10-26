"""YAML configs with inheritance (`base: other.yaml`) and command-line overrides (`a.b=value`)."""
import copy
from pathlib import Path

import yaml


def deep_update(base, extra):
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path, overrides=()):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    if "base" in cfg:
        cfg = deep_update(load_config(path.parent / cfg.pop("base")), cfg)
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


def flatten(cfg, prefix=""):
    """{'a': {'b': 1}} -> {'a.b': 1} (for experiment tracking)."""
    out = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out
