"""Class definitions shared by data preparation, training, evaluation and the demo.

Taxonomy: the 19 Cityscapes/BDD100K semantic classes (train ids 0..18), split into
11 "stuff" classes (uncountable background) and 8 "thing" classes (countable objects).
Traffic lights, traffic signs and poles are stuff: BDD100K has no instance masks for them.
"""

IGNORE = 255

SEM_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]
NUM_SEM = len(SEM_CLASSES)  # 19

STUFF_IDS = list(range(0, 11))   # road .. sky
THING_IDS = list(range(11, 19))  # person .. bicycle (semantic train ids)

# Detection classes = the 8 thing classes, indexed 0..7 for the FCOS head.
DET_CLASSES = SEM_CLASSES[11:]
NUM_DET = len(DET_CLASSES)  # 8


def det_to_sem(det_id: int) -> int:
    """Detection class index (0..7) -> semantic train id (11..18)."""
    return det_id + 11


# BDD100K ins_seg bitmask category ids (R channel, 1-based) -> detection index.
# 1 pedestrian, 2 rider, 3 car, 4 truck, 5 bus, 6 train, 7 motorcycle, 8 bicycle
INS_CAT_TO_DET = {i + 1: i for i in range(NUM_DET)}

# Classes watched separately because they degrade first when the model gets smaller.
THIN_SEM = ["pole", "traffic light", "traffic sign"]
THIN_DET = ["rider", "motorcycle", "bicycle"]

# Cityscapes palette, used for visualisation only.
PALETTE = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32),
]
