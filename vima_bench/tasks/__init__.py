import os

import importlib_resources
from omegaconf import OmegaConf

from .partition_files import *
from .task_suite import *

__all__ = ["ALL_TASKS", "ALL_PARTITIONS", "PARTITION_TO_SPECS"]

_ALL_TASKS = {
    "instruction_following": [
        SimpleManipulation,
        SceneUnderstanding,
        Rotate,
    ],
    "constraint_satisfaction": [
        WithoutExceeding,
        WithoutTouching,
    ],
    "novel_concept_grounding": [
        NovelAdjAndNoun,
        NovelAdj,
        NovelNoun,
        Twist,
    ],
    "one_shot_imitation": [
        FollowMotion,
        FollowOrder,
    ],
    "rearrangement": [Rearrange],
    "require_memory": [
        ManipulateOldNeighbor,
        PickInOrderThenRestore,
        RearrangeThenRestore,
    ],
    "require_reasoning": [
        SameColor,
        SameProfile,
    ],
}
ALL_TASKS = {
    f"{group}/{task.task_name}": task
    for group, tasks in _ALL_TASKS.items()
    for task in tasks
}


_ALL_TASK_SUB_NAMES = [
    task.task_name for tasks in _ALL_TASKS.values() for task in tasks
]


def _partition_file_path(fname) -> str:
    with importlib_resources.files("vima_bench.tasks.partition_files") as p:
        return os.path.join(str(p), fname)


def _load_partition_file(file: str):
    file = _partition_file_path(file)
    partition = OmegaConf.to_container(OmegaConf.load(file), resolve=True)
    partition_keys = set(partition.keys())
    for k in partition_keys:
        if k not in _ALL_TASK_SUB_NAMES:
            partition.pop(k)
    return partition


# train
TRAIN_PARTITION = _load_partition_file("train.yaml")

# test
PLACEMENT_GENERALIZATION = _load_partition_file("placement_generalization.yaml")
COMBINATORIAL_GENERALIZATION = _load_partition_file("combinatorial_generalization.yaml")
NOVEL_OBJECT_GENERALIZATION = _load_partition_file("novel_object_generalization.yaml")
NOVEL_TASK_GENERALIZATION = _load_partition_file("novel_task_generalization.yaml")


ALL_PARTITIONS = [
    "placement_generalization",
    "combinatorial_generalization",
    "novel_object_generalization",
    "novel_task_generalization",
]

PARTITION_TO_SPECS = {
    "train": TRAIN_PARTITION,
    "test": {
        "placement_generalization": PLACEMENT_GENERALIZATION,
        "combinatorial_generalization": COMBINATORIAL_GENERALIZATION,
        "novel_object_generalization": NOVEL_OBJECT_GENERALIZATION,
        "novel_task_generalization": NOVEL_TASK_GENERALIZATION,
    },
}
