"""Scan2CAD / S2C-Completion annotations.

Each scene annotation has a scan `trs` (ScanNet z-up -> Scan2CAD y-up world)
and a list of `aligned_models`, each with a CAD `trs`, the CAD's half-extent
`bbox` and `center` in CAD coordinates, a symmetry tag, and (S2C-Completion
only) the matched `scannet_instance_id` (-1 if unmatched).
"""

import json
from functools import cache
from pathlib import Path

import numpy as np

from caoa.geometry import compose, decompose, quat_to_matrix


@cache
def load_annotations(path: Path) -> dict[str, dict]:
    with open(path) as f:
        return json.load(f)


def trs_matrix(trs: dict) -> np.ndarray:
    """T @ R @ S from a Scan2CAD `trs` dict."""
    return compose(trs["translation"], quat_to_matrix(trs["rotation"]), trs["scale"])


def bbox_matrix(model: dict) -> np.ndarray:
    """Maps the cube [-1, 1]^3 onto the model's oriented bounding box in world coordinates."""
    return trs_matrix(model["trs"]) @ compose(t=model["center"]) @ compose(s=model["bbox"])


def oriented_box(model: dict):
    """(center, rotation, full extents) of the model's bounding box in world coordinates.

    This is the alignment target the OCAM is trained and scored against, as opposed to the CAD model's own `trs`.
    """
    t, R, half = decompose(bbox_matrix(model))
    return t, R, 2.0 * half
