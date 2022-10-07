import math
from enum import Enum
from typing import List

from .definitions import ObjEntry, SizeRange
from .profiles import ProfilePedia
from .replace_fns import *


class ObjPedia(Enum):
    """
    An encyclopedia of objects in VIMA world.
    """

    BOWL = ObjEntry(
        name="bowl",
        assets="bowl/bowl.urdf",
        size_range=SizeRange(
            low=(0.17, 0.17, 0),
            high=(0.17, 0.17, 0),
        ),
        symmetry=0,
        profile=ProfilePedia.CIRCLE_LIKE,
    )
    BLOCK = ObjEntry(
        name="block",
        alias=["cube"],
        assets="stacking/block.urdf",
        size_range=SizeRange(
            low=(0.07, 0.07, 0.07),
            high=(0.07, 0.07, 0.07),
        ),
        from_template=True,
        symmetry=1 / 4 * math.pi,
        profile=ProfilePedia.SQUARE_LIKE,
    )
    SHORTER_BLOCK = ObjEntry(
        name="shorter block",
        alias=["cube"],
        assets="stacking/block.urdf",
        size_range=SizeRange(
            low=(0.07, 0.07, 0.03),
            high=(0.07, 0.07, 0.03),
        ),
        from_template=True,
        symmetry=1 / 4 * math.pi,
        profile=ProfilePedia.SQUARE_LIKE,
    )
    PALLET = ObjEntry(
        name="pallet",
        assets="pallet/pallet.urdf",
        size_range=SizeRange(
            low=(0.3 * 0.7, 0.25 * 0.7, 0.25 * 0.7 - 0.14),
            high=(0.3 * 0.7, 0.25 * 0.7, 0.25 * 0.7 - 0.14),
        ),
        profile=ProfilePedia.SQUARE_LIKE,
    )
    FRAME = ObjEntry(
        name="frame",
        novel_name=["zone"],
        assets="zone/zone.urdf",
        size_range=SizeRange(
            low=(0.15 * 1.5, 0.15 * 1.5, 0),
            high=(0.15 * 1.5, 0.15 * 1.5, 0),
        ),
        profile=ProfilePedia.SQUARE_LIKE,
    )
    CONTAINER = ObjEntry(
        name="container",
        assets="container/container-template.urdf",
        size_range=SizeRange(
            low=(0.15, 0.15, 0.05),
            high=(0.17, 0.17, 0.05),
        ),
        from_template=True,
        replace_fn=container_replace_fn,
        profile=ProfilePedia.SQUARE_LIKE,
    )
    THREE_SIDED_RECTANGLE = ObjEntry(
        name="three-sided rectangle",
        assets="square/square-template.urdf",
        size_range=SizeRange(
            low=(0.2, 0.2, 0.0),
            high=(0.2, 0.2, 0.0),
        ),
        from_template=True,
        replace_fn=container_replace_fn,
    )
    SMALL_BLOCK = ObjEntry(
        name="small block",
        assets="block/small.urdf",
        size_range=SizeRange(
            low=(0.03, 0.03, 0.03),
            high=(0.03, 0.03, 0.03),
        ),
        from_template=True,  # this will activate the replace dict, i.e. (SIZE here)
    )
    LINE = ObjEntry(
        name="line",
        assets="line/line-template.urdf",
        size_range=SizeRange(
            low=(0.25, 0.04, 0.001),
            high=(0.25, 0.04, 0.001),
        ),
        from_template=True,  # this will activate the replace dict, i.e. (SIZE here)
    )
    SQUARE = ObjEntry(
        name="square",
        assets="square/square-template-allsides.urdf",
        size_range=SizeRange(
            low=(0.2, 0.04, 0.001),
            high=(0.2, 0.04, 0.001),
        ),
        from_template=True,
        replace_fn=container_replace_fn,
        profile=ProfilePedia.SQUARE_LIKE,
    )

    CAPITAL_LETTER_A = ObjEntry(
        name="letter A",
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
            high=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("capital_letter_a.obj"),
        symmetry=2 * math.pi,
    )

    CAPITAL_LETTER_E = ObjEntry(
        name="letter E",
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
            high=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("capital_letter_e.obj"),
        symmetry=2 * math.pi,
    )

    CAPITAL_LETTER_G = ObjEntry(
        name="letter G",
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
            high=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("capital_letter_g.obj"),
        symmetry=2 * math.pi,
    )

    CAPITAL_LETTER_M = ObjEntry(
        name="letter M",
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.7, 0.08 * 1.7, 0.02 * 1.7),
            high=(0.08 * 1.7, 0.08 * 1.7, 0.02 * 1.7),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("capital_letter_m.obj"),
        symmetry=2 * math.pi,
    )

    CAPITAL_LETTER_R = ObjEntry(
        name="letter R",
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
            high=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("capital_letter_r.obj"),
        symmetry=2 * math.pi,
    )

    CAPITAL_LETTER_T = ObjEntry(
        name="letter T",
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
            high=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("capital_letter_t.obj"),
        symmetry=2 * math.pi,
    )

    CAPITAL_LETTER_V = ObjEntry(
        name="letter V",
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
            high=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("capital_letter_v.obj"),
        symmetry=2 * math.pi,
    )

    CROSS = ObjEntry(
        name="cross",
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.6, 0.08 * 1.6, 0.02 * 1.6),
            high=(0.08 * 1.6, 0.08 * 1.6, 0.02 * 1.6),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("cross.obj"),
        symmetry=math.pi,
    )

    DIAMOND = ObjEntry(
        name="diamond",
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.6, 0.08 * 1.6, 0.02 * 1.6),
            high=(0.08 * 1.6, 0.08 * 1.6, 0.02 * 1.6),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("diamond.obj"),
        symmetry=math.pi,
    )

    TRIANGLE = ObjEntry(
        name="triangle",
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.6, 0.08 * 1.6, 0.02 * 1.6),
            high=(0.08 * 1.6, 0.08 * 1.6, 0.02 * 1.6),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("triangle.obj"),
        symmetry=2 / 3 * math.pi,
    )

    FLOWER = ObjEntry(
        name="flower",
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
            high=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("flower.obj"),
        symmetry=math.pi / 2,
    )

    HEART = ObjEntry(
        name="heart",
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
            high=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("heart.obj"),
        symmetry=2 * math.pi,
    )

    HEXAGON = ObjEntry(
        name="hexagon",
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.6, 0.08 * 1.6, 0.02 * 1.6),
            high=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("hexagon.obj"),
        symmetry=2 / 6 * math.pi,
    )

    PENTAGON = ObjEntry(
        name="pentagon",
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.6, 0.08 * 1.6, 0.02 * 1.6),
            high=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("pentagon.obj"),
        symmetry=2 / 5 * math.pi,
    )

    L_BLOCK = ObjEntry(
        name="L-shaped block",
        alias=["L-shape"],
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.6, 0.08 * 1.6, 0.02 * 1.6),
            high=(0.08 * 1.6, 0.08 * 1.6, 0.02 * 1.6),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("right_angle.obj"),
        symmetry=2 * math.pi,
    )

    RING = ObjEntry(
        name="ring",
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.6, 0.08 * 1.6, 0.02 * 1.6),
            high=(0.08 * 1.6, 0.08 * 1.6, 0.02 * 1.6),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("ring.obj"),
        symmetry=0,
        profile=ProfilePedia.CIRCLE_LIKE,
    )

    ROUND = ObjEntry(
        name="round",
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.6, 0.08 * 1.6, 0.02 * 1.6),
            high=(0.08 * 1.6, 0.08 * 1.6, 0.02 * 1.6),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("round.obj"),
        profile=ProfilePedia.CIRCLE_LIKE,
    )

    STAR = ObjEntry(
        name="star",
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.6, 0.08 * 1.6, 0.02 * 1.6),
            high=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("star.obj"),
        symmetry=2 * math.pi / 5,
    )

    # Google scanned objects
    PAN = ObjEntry(
        name="pan",
        assets="google/object-template.urdf",
        # size_range=SizeRange(low=(0.275, 0.275, 0.05), high=(0.275, 0.275, 0.05),),
        size_range=SizeRange(
            low=(0.16, 0.24, 0.03),
            high=(0.16, 0.24, 0.03),
        ),
        from_template=True,
        replace_fn=google_scanned_obj_fn("frypan.obj"),
        symmetry=0,
        profile=ProfilePedia.CIRCLE_LIKE,
    )

    HANOI_STAND = ObjEntry(
        name="stand",
        assets="hanoi/stand.urdf",
        size_range=SizeRange(
            low=(0.18, 0.54, 0.01),
            high=(0.18, 0.54, 0.01),
        ),
    )
    HANOI_DISK = ObjEntry(
        name="disk",
        assets="hanoi/disk-mod.urdf",
        size_range=SizeRange(low=(0.18, 0.18, 0.035), high=(0.18, 0.18, 0.035)),
        profile=ProfilePedia.CIRCLE_LIKE,
    )

    @classmethod
    def all_entries(cls) -> List[ObjEntry]:
        return [e for e in cls]

    @classmethod
    def lookup_object_by_name(cls, name: str) -> ObjEntry:
        """
        Given an object name, return corresponding object entry
        """
        for e in cls:
            if name == e.value.name:
                return e
        raise ValueError(f"Cannot find provided object {name}")

    @classmethod
    def all_entries_no_rotational_symmetry(cls) -> List[ObjEntry]:
        return [
            e
            for e in cls
            if e.value.symmetry is not None
            and math.isclose(e.value.symmetry, 2 * math.pi, rel_tol=1e-6, abs_tol=1e-8)
        ]
