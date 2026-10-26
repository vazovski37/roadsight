"""Sanity tests for the parts that are easy to get subtly wrong: metrics, fusion, FCOS targets, model shapes.

Tests marked `data` need the prepared val split in data/bdd10k (or ROADSIGHT_DATA).
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from roadsight.classes import IGNORE, NUM_SEM
from roadsight.labels import gt_panoptic, load_bitmask, load_sem, read_split
from roadsight.fcos import fcos_targets
from roadsight.metrics import confusion, panoptic_metrics, pq_stats, semantic_metrics
from roadsight.model import DeployModel, RoadSightNet, make_locations
from roadsight.postprocess import fuse, nms

DATA = Path(os.environ.get("ROADSIGHT_DATA", Path(__file__).resolve().parents[2] / "data" / "bdd10k"))
needs_data = pytest.mark.skipif(not (DATA / "splits" / "val.txt").exists(), reason="prepared data not found")


def test_nms_keeps_best_and_drops_duplicates():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    assert nms(boxes, scores, 0.5).tolist() == [0, 2]


def test_pq_perfect_prediction_is_100():
    cls = np.zeros((8, 8), np.uint8)
    cls[2:5, 2:5] = 13  # a car
    cls[6:8, 6:8] = 13  # another car
    inst = np.zeros((8, 8), np.int32)
    inst[2:5, 2:5] = 1
    inst[6:8, 6:8] = 2
    crowd = np.zeros((8, 8), bool)
    m = panoptic_metrics(pq_stats(cls, inst, cls, inst, crowd))
    assert m["PQ"] == pytest.approx(100.0)


def test_pq_counts_false_positive_and_missed_instance():
    gt_cls = np.zeros((8, 8), np.uint8)
    gt_cls[0:4, 0:4] = 13
    gt_inst = np.zeros((8, 8), np.int32)
    gt_inst[0:4, 0:4] = 1
    pred_cls = gt_cls.copy()
    pred_inst = np.zeros((8, 8), np.int32)
    pred_cls[0:4, 0:4] = 0       # car predicted as road -> missed car
    pred_cls[6:8, 6:8] = 13      # car where there is road -> false positive
    pred_inst[6:8, 6:8] = 1
    stats = pq_stats(pred_cls, pred_inst, gt_cls, gt_inst, np.zeros((8, 8), bool))
    tp, fp, fn, _ = stats[13]
    assert (tp, fp, fn) == (0, 1, 1)


def test_miou_of_perfect_prediction():
    gt = np.random.default_rng(0).integers(0, NUM_SEM, (16, 16)).astype(np.uint8)
    gt[0, 0] = IGNORE
    m = semantic_metrics(confusion(np.where(gt == IGNORE, 0, gt), gt))
    assert m["mIoU"] == pytest.approx(100.0)


def test_fusion_assigns_box_pixels_of_matching_class():
    sem = np.zeros((20, 20), np.uint8)
    sem[5:10, 5:10] = 13
    boxes = np.array([[5, 5, 10, 10], [12, 12, 18, 18]], np.float32)
    scores = np.array([0.9, 0.8], np.float32)
    labels = np.array([2, 2])  # car; the second box covers only road -> rejected by fill_thr
    cls_map, inst_map, instances = fuse(sem, boxes, scores, labels)
    assert len(instances) == 1
    assert (inst_map == 1).sum() == 25
    assert cls_map[0, 0] == 0  # road stays stuff


def test_fcos_assigns_small_box_to_p3_and_ignores_crowd():
    locs = make_locations([(10, 10), (5, 5), (3, 3), (2, 2), (1, 1)], "cpu")
    targets = [{"boxes": torch.tensor([[16.0, 16.0, 48.0, 48.0], [56.0, 56.0, 80.0, 80.0]]),
                "labels": torch.tensor([2, 0]), "crowd": torch.tensor([False, True])}]
    cls, reg, ignore = fcos_targets(locs, targets, 8)
    pos = cls[0] >= 0
    assert pos.any() and (cls[0][pos] == 2).all()
    assert pos[:100].sum() == pos.sum()          # all positives on P3 (box is 32 px)
    assert ignore[0].any() and not (ignore[0] & pos).any()


@pytest.mark.parametrize("backbone,sem_from", [("resnet18", "p2"), ("resnet18", "p3"), ("mobilenet_v3_large", "p2")])
def test_model_output_shapes(backbone, sem_from):
    net = RoadSightNet(backbone=backbone, pretrained="none", fpn_channels=128, head_convs=2,
                       sem_channels=64, sem_from=sem_from).eval()
    with torch.no_grad():
        sem, scores, boxes = DeployModel(net)(torch.rand(1, 3, 360, 640) * 255)
    assert sem.shape == (1, 360, 640) and sem.dtype == torch.uint8
    assert scores.shape[2] == 8 and boxes.shape[2] == 4 and scores.shape[1] == boxes.shape[1]


@needs_data
def test_ground_truth_through_fusion_scores_high_pq():
    """GT semantic map + GT boxes through our fusion must give near-perfect PQ.

    This checks the whole panoptic path (GT construction, fusion, PQ) end to end; the remaining
    gap is exactly the cost of box-guided fusion on overlapping same-class objects.
    """
    ann = json.loads((DATA / "annotations.json").read_text())
    stats = 0
    for name in read_split(DATA, "val")[:40]:
        sem = load_sem(DATA, name)
        gt_cls, gt_inst, crowd = gt_panoptic(sem, load_bitmask(DATA, name))
        a = ann[name]
        keep = ~np.asarray(a["crowd"], bool)
        boxes = np.asarray(a["boxes"], np.float32).reshape(-1, 4)[keep]
        labels = np.asarray(a["labels"])[keep]
        # semantic input = GT classes (things from instances), void -> road just to have a value
        sem_in = np.where(gt_cls == IGNORE, 0, gt_cls).astype(np.uint8)
        cls_map, inst_map, _ = fuse(sem_in, boxes, np.ones(len(boxes), np.float32), labels,
                                    {"score_thr": 0.0, "fill_thr": 0.0, "stuff_area": 0.0})
        stats = stats + pq_stats(cls_map, inst_map, gt_cls, gt_inst, crowd)
    m = panoptic_metrics(stats)
    assert m["PQ_st"] > 95
    assert m["PQ_th"] > 80
