"""FCOS training targets and losses (Tian et al., 2019, with the common improvements:
centre sampling, GIoU box loss weighted by centerness, normalised regression targets)."""
import torch
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou_loss, sigmoid_focal_loss

from .model import STRIDES, flatten_levels, make_locations

# Each level is responsible for objects whose largest ltrb distance falls in its range (input pixels).
SIZE_RANGES = ((-1, 64), (64, 128), (128, 256), (256, 512), (512, float("inf")))
CENTER_RADIUS = 1.5  # in strides


@torch.no_grad()
def fcos_targets(locations, targets, num_classes):
    """Assign every location to at most one ground-truth box.

    locations: list (per level) of [Li, 2]; targets: list (per image) of dicts with
    boxes [K, 4] (x1 y1 x2 y2, input pixels), labels [K], crowd [K].
    Returns cls [N, L] (class index or -1 = background), reg [N, L, 4] (ltrb / stride)
    and ignore [N, L] (background locations inside crowd regions: excluded from the class loss).
    """
    device = locations[0].device
    locs = torch.cat(locations)
    stride = torch.cat([torch.full((len(l),), float(s), device=device) for l, s in zip(locations, STRIDES)])
    lo = torch.cat([torch.full((len(l),), float(r[0]), device=device) for l, r in zip(locations, SIZE_RANGES)])
    hi = torch.cat([torch.full((len(l),), float(r[1]), device=device) for l, r in zip(locations, SIZE_RANGES)])
    x, y = locs[:, 0:1], locs[:, 1:2]

    cls_t, reg_t, ign_t = [], [], []
    for t in targets:
        boxes, labels, crowd = t["boxes"].to(device), t["labels"].to(device), t["crowd"].to(device)
        cls = torch.full((len(locs),), -1, dtype=torch.long, device=device)
        reg = torch.zeros(len(locs), 4, device=device)
        ignore = torch.zeros(len(locs), dtype=torch.bool, device=device)
        keep = ~crowd
        b, lab = boxes[keep], labels[keep]
        if len(b):
            ltrb = torch.stack([x - b[:, 0], y - b[:, 1], b[:, 2] - x, b[:, 3] - y], dim=2)  # [L, K, 4]
            # centre sampling: the location must lie within radius*stride of the box centre (and in the box)
            cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
            r = stride[:, None] * CENTER_RADIUS
            cx1, cy1 = torch.maximum(cx - r, b[:, 0]), torch.maximum(cy - r, b[:, 1])
            cx2, cy2 = torch.minimum(cx + r, b[:, 2]), torch.minimum(cy + r, b[:, 3])
            in_center = (x > cx1) & (y > cy1) & (x < cx2) & (y < cy2)
            max_d = ltrb.max(dim=2).values
            in_range = (max_d > lo[:, None]) & (max_d <= hi[:, None])
            area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            cand_area = torch.where(in_center & in_range, area[None].expand_as(max_d), torch.full_like(max_d, float("inf")))
            min_area, idx = cand_area.min(dim=1)  # ambiguous locations go to the smallest box
            pos = torch.isfinite(min_area)
            cls[pos] = lab[idx[pos]]
            reg = ltrb[torch.arange(len(locs), device=device), idx] / stride[:, None]
        if crowd.any():
            cb = boxes[crowd]
            inside = (x > cb[:, 0]) & (y > cb[:, 1]) & (x < cb[:, 2]) & (y < cb[:, 3])
            ignore = inside.any(dim=1) & (cls < 0)
        cls_t.append(cls)
        reg_t.append(reg)
        ign_t.append(ignore)
    return torch.stack(cls_t), torch.stack(reg_t), torch.stack(ign_t)


def centerness(ltrb):
    lr, tb = ltrb[:, [0, 2]], ltrb[:, [1, 3]]
    return torch.sqrt((lr.min(1).values / lr.max(1).values).clamp(min=0) *
                      (tb.min(1).values / tb.max(1).values).clamp(min=0))


def fcos_loss(out, targets, num_classes):
    """Focal loss (classes) + GIoU (boxes, weighted by centerness) + BCE (centerness)."""
    sizes = [c.shape[-2:] for c in out["cls"]]
    locations = make_locations(sizes, out["cls"][0].device)
    cls_t, reg_t, ignore = fcos_targets(locations, targets, num_classes)

    cls_p = flatten_levels(out["cls"]).float()           # [N, L, C]
    reg_p = flatten_levels(out["reg"]).float()           # [N, L, 4]
    ctr_p = flatten_levels(out["ctr"]).float()[..., 0]   # [N, L]

    pos = cls_t >= 0
    num_pos = max(pos.sum().item(), 1.0)
    onehot = torch.zeros_like(cls_p)
    onehot[pos, cls_t[pos]] = 1.0
    valid = ~ignore
    loss_cls = sigmoid_focal_loss(cls_p[valid], onehot[valid], alpha=0.25, gamma=2.0, reduction="sum") / num_pos

    if pos.any():
        rt, rp = reg_t[pos], reg_p[pos]
        ctr_t = centerness(rt)
        # boxes relative to the location; GIoU is scale invariant so stride units are fine
        to_box = lambda d: torch.stack([-d[:, 0], -d[:, 1], d[:, 2], d[:, 3]], dim=1)
        giou = generalized_box_iou_loss(to_box(rp), to_box(rt), reduction="none")
        loss_reg = (giou * ctr_t).sum() / ctr_t.sum().clamp(min=1e-6)
        loss_ctr = F.binary_cross_entropy_with_logits(ctr_p[pos], ctr_t, reduction="sum") / num_pos
    else:
        loss_reg = reg_p.sum() * 0.0
        loss_ctr = ctr_p.sum() * 0.0
    return {"det_cls": loss_cls, "det_reg": loss_reg, "det_ctr": loss_ctr}
