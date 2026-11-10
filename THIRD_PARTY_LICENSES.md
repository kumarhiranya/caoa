# Third-party code

The MIT license in `LICENSE` covers this project's own code. The files below are ported from or vendor third-party
code and remain under their authors' licenses, which take precedence for those files.

| Files | Origin | License |
|---|---|---|
| `caoa/models/ptv3/` | Point Transformer V3, Xiaoyang Wu et al. (github.com/Pointcept/PointTransformerV3) | CC BY-NC 4.0 |
| `caoa/eval/scan2cad_benchmark.py` (port), `tests/scan2cad_ref.py` (copy) | Scan2CAD benchmark script, Armen Avetisyan et al., TUM (github.com/skanti/Scan2CAD) | CC BY-NC-SA 3.0 |
| `caoa/models/odgnet.py`, `caoa/models/pointops.py`, `caoa/models/chamfer.py`, `caoa/data/snindoor_dataset.py`, parts of `caoa/data/completion_dataset.py` and `caoa/eval/completion_metrics.py` | ODGNet, Pingping Cai et al. (github.com/corecai163/ODGNet) | no license stated by the authors |
| `caoa/models/sparse.py` (semantics of) | MinkowskiEngine (github.com/NVIDIA/MinkowskiEngine) | MIT |
| `scripts/external/softgroup_infer.py` (runs) | SoftGroup (github.com/thangvubk/SoftGroup) | MIT |
| `scripts/external/sphericalmask_infer.py` (runs) | SphericalMask (official repository) | see its repository |

Because of the non-commercial terms above, and those of the ScanNet, ShapeNet and Scan2CAD data the models are
trained on, the repository as a whole is for non-commercial research use. The trained models and data on Zenodo are
under CC BY-NC 4.0.
