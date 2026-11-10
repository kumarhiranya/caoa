# Instance predictions (SoftGroup, SphericalMask)

The full pipeline runs on SoftGroup or SphericalMask instance predictions for the 312 ScanNet val scans. The data
package ships both (`data/instances/<model>/val/<scene>_sg_preds.npz`). They were made with each model's official code
and released ScanNet checkpoint, by the drivers in `scripts/external/`, which write the per-scene format
`caoa.data.instances` reads: run-length masks over the scene's `.ply` vertices, a 1-based `label_id` over the 18
instance classes, `conf`, and per-point semantic labels. To make them yourself:

## Common setup

Both models use SoftGroup's ScanNet preprocessing (`dataset/scannetv2/prepare_data_inst.py` in either repository: one
`<scene>_inst_nostuff.pth` per val scene) and the superpoints of the val scenes. We ran both in one venv: torch 2.9.1 +
cu126, spconv-cu126, each repository's CUDA ops built in place (`python setup.py build_ext --inplace`, with the
sparsehash headers on `CPATH`), and plyfile instead of open3d for the preprocessing.

## SoftGroup

Code: github.com/thangvubk/SoftGroup; checkpoint `softgroup_scannet_spconv2.pth` (linked in its README). Two
changes are needed with current libraries, both in the driver; neither alters the model:
- PyTorch >= 2 no longer indexes a CPU tensor with CUDA indices, which `get_instances` does: `proposals_idx` is moved
  to the GPU first;
- NumPy 2 removed `np.in1d`, `np.float`, `np.int`, `np.bool` (used by upstream `ScanNetEval`): restored as aliases.

**Check:** ScanNet val AP / AP50 / AP25 = 45.8 / 67.3 / 79.1 (published: 45.8 / 67.4 / 79.1). Inference takes about
5 minutes on one A100.

```bash
cd <softgroup>/repo && TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=0 <python> \
    <caoa>/scripts/external/softgroup_infer.py configs/softgroup/softgroup_scannet.yaml \
    ../ckpt/softgroup_scannet_spconv2.pth ../preds/val --data-root <folder holding val/>
```

## SphericalMask

Code: the official SphericalMask repository and its released ScanNet val checkpoint.

Its `pointnet2` and `segmentator` ops are built as well. Changes needed to run the released code, none of which alter
the model:
- the checkpoint names one module `inst_ray_head`, the code `rid_module` (same 14 tensors and shapes); upstream loads
  non-strictly and would leave it random, so the driver renames it and loads strictly;
- NumPy 2 removed `np.in1d`, `np.float`, `np.int`, `np.bool` (used by upstream `ScanNetEval`): restored as aliases;
- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (torch >= 2.6 default); the data directory must not contain "scannetv2", which
  upstream's dataset code rewrites;
- masks are stored as SphericalMask's `rle_encode_gpu_batch` writes them (SoftGroup's runs, as an integer array instead
  of a string); `caoa.data.instances.rle_decode` reads both.

**Check:** ScanNet val AP / AP50 / AP25 = 62.55 / 82.36 / 88.32 (published: 62.6 / 81.9). Inference takes about
9 minutes on one A100.

```bash
cd <sphericalmask>/repo && TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=0 ../venv/bin/python \
    <caoa>/scripts/external/sphericalmask_infer.py ../configs/spherical_mask_val.yaml ../ckpt/spherical_mask_155.pth ../preds/val
```
Point `softgroup_preds` / `sphericalmask_preds` in `configs/paths.local.yaml` at the output folders.
