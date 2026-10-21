import numpy as np
import pytest

from caoa.data import scan2cad
from caoa.data.labels import raw_to_nyu40_name, scannet_split
from caoa.data.scannet import load_scene, scene_files
from caoa.geometry import quat_to_matrix
from caoa.paths import RESOURCES, load_paths
from tests import legacy_ref

PATHS = load_paths()
needs_annotations = pytest.mark.skipif(not PATHS.s2c_annotations.exists(), reason="S2C annotations not found")
needs_scannet = pytest.mark.skipif(not PATHS.scannet_scans.exists(), reason="ScanNet scans not found")


@needs_annotations
def test_box_and_cad_matrices_match_legacy():
    annos = scan2cad.load_annotations(PATHS.s2c_annotations)
    for anno in list(annos.values())[:200]:
        np.testing.assert_allclose(scan2cad.trs_matrix(anno["trs"]), legacy_ref.calc_Mcad(anno), atol=1e-9)
        for model in anno["aligned_models"]:
            np.testing.assert_allclose(scan2cad.trs_matrix(model["trs"]), legacy_ref.calc_Mcad(model), atol=1e-9)
            t_ref, q_ref, s_ref = legacy_ref.decompose_mat4(legacy_ref.calc_Mbbox(model))
            t, R, extents = scan2cad.oriented_box(model)
            np.testing.assert_allclose(t, t_ref, atol=1e-9)
            np.testing.assert_allclose(R, quat_to_matrix(q_ref), atol=1e-6)
            np.testing.assert_allclose(extents, 2 * s_ref, atol=1e-9)


def test_label_map_matches_softgroup():
    ref = legacy_ref.get_raw2scannetv2_label_map(RESOURCES / "scannetv2-labels.combined.tsv")
    assert raw_to_nyu40_name() == ref


def test_splits_are_official_scannet_v2():
    assert scannet_split("scene0000_00") == "train"
    assert scannet_split("scene0011_00") == "val"
    assert scannet_split("scene0707_00") is None  # test split


@needs_scannet
@pytest.mark.data
@pytest.mark.parametrize("scene_id", ["scene0011_00", "scene0217_00", "scene0000_00"])
def test_scene_loading_matches_softgroup_prep(scene_id):
    if not scene_files(PATHS.scannet_scans, scene_id)["mesh"].exists():
        pytest.skip(f"{scene_id} not extracted")
    scene = load_scene(PATHS.scannet_scans, scene_id)
    ref_points, ref_colors, ref_sem, ref_inst = legacy_ref.prepare_inst(
        str(scene_files(PATHS.scannet_scans, scene_id)["mesh"]), raw_to_nyu40_name(), split=scannet_split(scene_id)
    )
    np.testing.assert_allclose(scene.points, ref_points, atol=1e-6)
    np.testing.assert_allclose(scene.colors / 127.5 - 1, ref_colors, atol=1e-6)  # ref went through float32
    np.testing.assert_array_equal(scene.semantic, ref_sem.astype(int))
    np.testing.assert_array_equal(scene.instance, ref_inst.astype(int))
