"""RoadSight network: one backbone + FPN shared by an FCOS detection head and a semantic head.

    image -> backbone (C2..C5) -> FPN -+- P3..P7 -> FCOS head      -> per-location class scores + boxes
                                       +- P2..P5 -> semantic head  -> 19-class logits at 1/4 resolution
"""
import math
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import FeaturePyramidNetwork, FrozenBatchNorm2d
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

from .classes import NUM_DET, NUM_SEM

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
STRIDES = (8, 16, 32, 64, 128)  # FCOS levels P3..P7


class Backbone(nn.Module):
    """ResNet or MobileNetV3 trunk returning C2..C5 (strides 4, 8, 16, 32)."""

    def __init__(self, name, pretrained=True):
        super().__init__()
        if name.startswith("resnet"):
            weights = "IMAGENET1K_V1" if pretrained else None
            net = getattr(torchvision.models, name)(weights=weights, norm_layer=FrozenBatchNorm2d)
            # detection convention: freeze the stem and the first stage
            for module in (net.conv1, net.layer1):
                for p in module.parameters():
                    p.requires_grad_(False)
            layers = {"layer1": "c2", "layer2": "c3", "layer3": "c4", "layer4": "c5"}
            self.body = IntermediateLayerGetter(net, layers)
        elif name == "mobilenet_v3_large":
            weights = "IMAGENET1K_V1" if pretrained else None
            net = torchvision.models.mobilenet_v3_large(
                weights=weights, norm_layer=partial(FrozenBatchNorm2d, eps=0.001))
            for p in net.features[:4].parameters():
                p.requires_grad_(False)
            layers = {"3": "c2", "6": "c3", "12": "c4", "16": "c5"}
            self.body = IntermediateLayerGetter(net.features, layers)
        else:
            raise ValueError(f"unknown backbone {name}")
        with torch.no_grad():
            feats = self.body(torch.zeros(1, 3, 64, 64))
        self.channels = [f.shape[1] for f in feats.values()]
        assert [64 // f.shape[-1] for f in feats.values()] == [4, 8, 16, 32]

    def forward(self, x):
        return self.body(x)


def conv_gn_relu(cin, cout):
    return [nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(32, cout), nn.ReLU(inplace=True)]


class FCOSHead(nn.Module):
    """FCOS head shared across levels. Layer names mirror torchvision's FCOS so COCO weights load."""

    def __init__(self, channels, num_convs=4, num_classes=NUM_DET, prior=0.01):
        super().__init__()
        self.cls_tower = nn.Sequential(*sum([conv_gn_relu(channels, channels) for _ in range(num_convs)], []))
        self.reg_tower = nn.Sequential(*sum([conv_gn_relu(channels, channels) for _ in range(num_convs)], []))
        self.cls_logits = nn.Conv2d(channels, num_classes, 3, padding=1)
        self.bbox_reg = nn.Conv2d(channels, 4, 3, padding=1)
        self.bbox_ctrness = nn.Conv2d(channels, 1, 3, padding=1)
        self.scales = nn.Parameter(torch.ones(len(STRIDES)))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
        nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior) / prior))

    def forward(self, feats):
        cls, reg, ctr = [], [], []
        for level, f in enumerate(feats):
            c = self.cls_tower(f)
            r = self.reg_tower(f)
            cls.append(self.cls_logits(c))
            # distances (left, top, right, bottom) in units of the level's stride
            reg.append(F.relu(self.scales[level] * self.bbox_reg(r)))
            ctr.append(self.bbox_ctrness(r))
        return cls, reg, ctr


class SemanticHead(nn.Module):
    """Panoptic-FPN semantic branch: bring every level to the finest one, sum, classify."""

    def __init__(self, in_channels, level_strides, channels=128, num_classes=NUM_SEM):
        super().__init__()
        base = level_strides[0]
        self.branches = nn.ModuleList()
        for s in level_strides:
            n_up = int(math.log2(s // base))
            layers = conv_gn_relu(in_channels, channels)
            for _ in range(max(n_up - 1, 0)):
                layers += conv_gn_relu(channels, channels)
            self.branches.append(nn.Sequential(*layers))
        self.predictor = nn.Conv2d(channels, num_classes, 1)

    def forward(self, feats):
        size = feats[0].shape[-2:]
        out = 0
        for branch, f in zip(self.branches, feats):
            # one conv per 2x step, then a single resize to the exact target size
            # (exact size also works for inputs that are not multiples of 32)
            y = branch(f)
            if y.shape[-2:] != size:
                y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
            out = out + y
        return self.predictor(out)


class RoadSightNet(nn.Module):
    def __init__(self, backbone="resnet18", pretrained="imagenet", fpn_channels=256, head_convs=4,
                 sem_channels=128, sem_from="p2", tasks=("det", "sem")):
        super().__init__()
        self.tasks = tuple(tasks)
        self.sem_from = sem_from
        self.backbone = Backbone(backbone, pretrained=pretrained == "imagenet")
        use_c2 = "sem" in self.tasks and sem_from == "p2"
        in_channels = self.backbone.channels if use_c2 else self.backbone.channels[1:]
        self.use_c2 = use_c2
        self.fpn = FeaturePyramidNetwork(in_channels, fpn_channels,
                                         extra_blocks=LastLevelP6P7(fpn_channels, fpn_channels))
        if "det" in self.tasks:
            self.det_head = FCOSHead(fpn_channels, head_convs)
        if "sem" in self.tasks:
            strides = [4, 8, 16, 32] if use_c2 else [8, 16, 32]
            self.sem_head = SemanticHead(fpn_channels, strides, sem_channels)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1) * 255, persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1) * 255, persistent=False)
        if pretrained == "coco_fcos":
            load_coco_fcos(self)

    def forward(self, images):
        """images: float tensor [N, 3, H, W] in 0..255 RGB."""
        x = (images - self.mean) / self.std
        c = self.backbone(x)
        if not self.use_c2:
            c.pop("c2")
        p = list(self.fpn(c).values())  # [P2], P3, P4, P5, P6, P7
        out = {}
        if "det" in self.tasks:
            out["cls"], out["reg"], out["ctr"] = self.det_head(p[-5:])
        if "sem" in self.tasks:
            out["sem"] = self.sem_head(p[: (4 if self.use_c2 else 3)])
        return out


def build_model(cfg):
    m = cfg["model"]
    return RoadSightNet(backbone=m["backbone"], pretrained=m["pretrained"], fpn_channels=m["fpn_channels"],
                        head_convs=m["head_convs"], sem_channels=m["sem_channels"], sem_from=m["sem_from"],
                        tasks=m["tasks"])


def load_coco_fcos(model):
    """Initialise backbone, FPN and the FCOS towers from torchvision's COCO FCOS-R50.

    Our FPN may have one extra (C2) lateral at index 0, so COCO FPN indices shift by one.
    The 80-class classifier is not loaded; the box/centerness predictors are class-agnostic and are.
    """
    from torchvision.models.detection import FCOS_ResNet50_FPN_Weights
    state = FCOS_ResNet50_FPN_Weights.COCO_V1.get_state_dict(progress=False)
    shift = 1 if model.use_c2 else 0
    mapped = {}
    for k, v in state.items():
        if k.startswith("backbone.body."):
            mapped[k.replace("backbone.body.", "backbone.body.")] = v
        elif k.startswith("backbone.fpn.inner_blocks.") or k.startswith("backbone.fpn.layer_blocks."):
            parts = k.split(".")
            parts[3] = str(int(parts[3]) + shift)
            mapped[".".join(parts[1:])] = v
        elif k.startswith("backbone.fpn.extra_blocks."):
            mapped[k[len("backbone."):]] = v
        elif k.startswith("head.classification_head.conv."):
            mapped[k.replace("head.classification_head.conv.", "det_head.cls_tower.")] = v
        elif k.startswith("head.regression_head.conv."):
            mapped[k.replace("head.regression_head.conv.", "det_head.reg_tower.")] = v
        elif k.startswith("head.regression_head.bbox_reg."):
            mapped[k.replace("head.regression_head.bbox_reg.", "det_head.bbox_reg.")] = v
        elif k.startswith("head.regression_head.bbox_ctrness."):
            mapped[k.replace("head.regression_head.bbox_ctrness.", "det_head.bbox_ctrness.")] = v
    if "det" not in model.tasks:
        mapped = {k: v for k, v in mapped.items() if not k.startswith("det_head.")}
    # lite models (128-channel FPN, 2-conv towers) can only take what fits: the backbone always does
    own = model.state_dict()
    skipped = [k for k, v in mapped.items() if k not in own or own[k].shape != v.shape]
    assert not any(k.startswith("backbone.body.") for k in skipped), "COCO backbone must load completely"
    mapped = {k: v for k, v in mapped.items() if k not in skipped}
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    assert not unexpected, unexpected
    return missing


def make_locations(feature_sizes, device):
    """Centre of every feature-map cell in input pixels, per level: list of [H*W, 2] (x, y)."""
    locs = []
    for (h, w), s in zip(feature_sizes, STRIDES):
        ys = torch.arange(h, device=device, dtype=torch.float32) * s + s // 2
        xs = torch.arange(w, device=device, dtype=torch.float32) * s + s // 2
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        locs.append(torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1))
    return locs


def flatten_levels(maps):
    """List of [N, C, H, W] -> [N, sum(H*W), C]."""
    return torch.cat([m.flatten(2).transpose(1, 2) for m in maps], dim=1)


class DeployModel(nn.Module):
    """Inference graph used for evaluation, ONNX export and the demo.

    Input: [N, 3, H, W] float 0..255 RGB.
    Outputs (only those of the trained tasks):
      sem    [N, H, W] uint8 class per pixel (bilinear-upsampled logits, then argmax)
      scores [N, L, 8] sqrt(class prob * centerness) per location
      boxes  [N, L, 4] x1, y1, x2, y2 in input pixels
    NMS and panoptic fusion run outside the graph (roadsight.postprocess).
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, images):
        out = self.net(images)
        h, w = images.shape[-2:]
        results = OrderedDict()
        if "sem" in out:
            logits = F.interpolate(out["sem"], size=(h, w), mode="bilinear", align_corners=False)
            results["sem"] = logits.argmax(1).to(torch.uint8)
        if "cls" in out:
            sizes = [c.shape[-2:] for c in out["cls"]]
            locs = torch.cat(make_locations(sizes, images.device))
            strides = torch.cat([torch.full((hh * ww,), float(s), device=images.device)
                                 for (hh, ww), s in zip(sizes, STRIDES)])
            cls = flatten_levels(out["cls"]).float().sigmoid()
            ctr = flatten_levels(out["ctr"]).float().sigmoid()
            reg = flatten_levels(out["reg"]).float() * strides[None, :, None]
            results["scores"] = torch.sqrt(cls * ctr)
            results["boxes"] = torch.cat([locs[None] - reg[..., :2], locs[None] + reg[..., 2:]], dim=-1)
        return tuple(results.values())

    @property
    def output_names(self):
        names = []
        if "sem" in self.net.tasks:
            names.append("sem")
        if "det" in self.net.tasks:
            names += ["scores", "boxes"]
        return names
