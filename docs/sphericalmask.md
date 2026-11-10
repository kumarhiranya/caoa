# SphericalMask instance predictions

The full pipeline runs on SoftGroup or SphericalMask instance predictions for the ScanNet val scans. For
SphericalMask, the official code and released ScanNet val checkpoint are run once over the 312 val scenes, and
`scripts/external/sphericalmask_infer.py` writes the predictions in the per-scene format `caoa.data.instances` reads.

**Setup** (in SphericalMask's own environment, about 11 GB: venv 7.1, data 3.0, predictions 0.7, checkpoint 0.35):
a separate venv (torch 2.9.1 + cu126, spconv, the upstream `pointnet2` and `segmentator` ops built locally, sparsehash
headers in `local/include`), and upstream's ScanNet preprocessing of the val scenes and superpoints (with plyfile
instead of open3d). Changes needed to run the released code, none of which alter the model:
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
Then point `sphericalmask_preds` in `configs/paths.local.yaml` at the output folder.
