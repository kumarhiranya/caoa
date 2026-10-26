---
name: caoa-data-locations
description: "Where CAOA datasets, checkpoints, git history and the SN-Indoor generator actually live on this machine (external drive + /home/data)"
metadata:
  node_type: memory
  type: reference
  originSessionId: be0601d6-11b5-43be-b0f8-dc7f68e3618a
  modified: 2026-10-02T17:57:50.414Z
---

Verified 2026-10-02. `~/research/3D Datasets` does NOT exist (`~/research/datasets` holds only ScanNet++ 2D crops). The real data is on the external drive `/media/hiranya/KHG Elements/` (slow; a recursive `find` over it times out, so use targeted `ls`):

- `3D Datasets/ShapeNetV2/`: complete ShapeNetCore v2 (55 categories, taxonomy.json, splits.json, aug_train_test_splits.pkl). All 3,049 CAD models referenced by S2C-Completion are present.
- `3D Datasets/ShapeNetV2-augmented/`: the SN-Indoor partials (35 categories, 8 .ply files per model).
- `3D Datasets/ScanNet_v2/dataset/scans/`: all 1,513 train/val scenes, but only as `.7z` archives that include `.sens` (~1.1 TB). No `train/`, `val/` or `instances/` folders and no `scannetv2_*.txt` split lists. `prepare_data_inst.py` there regenerates `*_inst_nostuff_v2.pth`.
- `mmlab-server-data/data/hiranya/Github/vr_reconstruction/`: server copy of the legacy repo with `.git` (remote github.com/kumarhiranya/vr_reconstruction), `checkpoints/` (pose_mnk, pose_ptv3, symmetry_ptv3, completion_checkpoints, …) and `results/*.csv`.
- `mmlab-server-data/data/minhas/ShapeNetAugmentation/ShapeNetAugmentation.py`: the SN-Indoor generator (Open3D RaycastingScene + non-linear crop).

`/home/data/datasets/ScanNet_v2/dataset/scans/` has 609 + 206 (all remaining val, extracted 2026-10-20, ~3 GB) extracted scenes. Originally 609 extracted scenes (scene0000_00 to scene0609_00, all with the 4 files prep needs). They cover 605/1,506 S2C scenes and 4,358/11,152 matched pairs.

Still not found: a CAPCM+SEM alignment checkpoint (all pose_mnk runs are type `mnkpv3`, which ignores the symmetry embedding), any SphericalMask outputs, and the cached `dataset/instances/*.npy`.

**Update 2026-10-21:** all needed data is local (the user said not to read from the external drive).
- ScanNet: 1,505 annotated scenes in `/home/data/datasets/ScanNet_v2/dataset/scans/` (all except `scene0212_00`, whose .7z is an empty 32-byte shell on every copy).
- Local ShapeNet mirror `/home/data/datasets/ShapeNetV2_caoa/{ShapeNetV2,ShapeNetV2-augmented}`: 22,213 CAD .obj + 171,992 SN-Indoor partials + split files. `configs/paths.yaml` points there.
- Legacy pose train set reproduced as 6,663 = 6,671 − 8 (scene0212_00).

**DGX (2026-10-24):** `~/lab` = `/network/rit/lab/kumar_lab` (5.8 TB free); `~/dgx_lab` = `/network/rit/dgx/dgx_kumar_lab` (96% full); home has a 13 GB quota.
- The lab already has `research_data/3D Datasets/{ScanNet_v2/dataset/scans, ShapeNetV2, ShapeNetV2-augmented, Scan2CAD}`. They're identical to the workstation data (MD5 for the bundle's files, sizes for the whole ShapeNet mirror). `scene0212_00` is missing there too.
- The CAOA data package is at `research_data/3D Datasets/caoa_data` (moved there from `research_data/Github/vr_reconstruction/caoa_data` on 2026-10-24). Inside the Slurm jobs' pyxis container the lab share is `/workspace/lab`, so `configs/paths.local.yaml` there uses that prefix. And the caoa clone is in `~/dgx_lab/research_data/Github/caoa`.
- The DGX is reachable from the workstation only over the GlobalProtect VPN (`ssh dgx`).
