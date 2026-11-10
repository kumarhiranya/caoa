import numpy as np
import pytest

from caoa.data import scan2cad
from caoa.data.labels import raw_to_nyu40_name, scannet_split
from caoa.data.scannet import load_scene, scene_files
from caoa.paths import load_paths

PATHS = load_paths()
needs_annotations = pytest.mark.skipif(not PATHS.s2c_annotations.exists(), reason="S2C annotations not found")
needs_scannet = pytest.mark.skipif(not PATHS.scannet_scans.exists(), reason="ScanNet scans not found")


@needs_annotations
def test_oriented_box_contains_the_cad_bounding_box():
    """The box maps the unit cube onto the CAD's bounding box (`bbox` half extents about `center`) placed by its trs."""
    annos = scan2cad.load_annotations(PATHS.s2c_annotations)
    cube = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)])
    for anno in list(annos.values())[:100]:
        for model in anno["aligned_models"]:
            t, R, extents = scan2cad.oriented_box(model)
            corners = cube * extents @ R.T + t
            cad = (np.asarray(model["center"]) + 2 * cube * np.asarray(model["bbox"]))
            M = scan2cad.trs_matrix(model["trs"])
            np.testing.assert_allclose(corners, cad @ M[:3, :3].T + M[:3, 3], atol=1e-6)


def test_label_map():
    m = raw_to_nyu40_name()
    assert m["chair"] == "chair" and m["kitchen cabinets"] == "cabinet" and m["wall"] == "wall"


def test_splits_are_official_scannet_v2():
    assert scannet_split("scene0000_00") == "train"
    assert scannet_split("scene0011_00") == "val"
    assert scannet_split("scene0707_00") is None  # test split


@needs_scannet
@pytest.mark.data
@pytest.mark.parametrize("scene_id", ["scene0011_00", "scene0217_00"])
def test_scene_loading(scene_id):
    if not scene_files(PATHS.scannet_scans, scene_id)["mesh"].exists():
        pytest.skip(f"{scene_id} not extracted")
    scene = load_scene(PATHS.scannet_scans, scene_id)
    n = len(scene.points)
    assert scene.colors.shape == (n, 3) and scene.semantic.shape == (n,) and scene.instance.shape == (n,)
    assert (scene.instance >= 0).any() and (scene.instance < 0).any()  # objects and layout
