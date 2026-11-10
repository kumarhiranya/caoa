#!/bin/bash
# Download the CAOA archives from Zenodo (doi:10.5281/zenodo.24121532), check their SHA-256 and extract them in the
# repository root.  Usage: scripts/download.sh [all|pipeline|data] ...   (default: all data)
#   all       caoa_models_v1.tar.zst           every trained model (3.3 GB)
#   pipeline  caoa_models_v1_pipeline.tar.zst  only the models the pipeline runs (364 MB)
#   data      caoa_data_v1.tar.zst             redistributable data: annotations ids, splits, embeddings, predictions (134 MB)
set -euo pipefail
cd "$(dirname "$0")/.."
URL=https://zenodo.org/records/24121532/files
[ $# -eq 0 ] && set -- all data
curl -fsSL "$URL/packages.json?download=1" -o packages.json
for what in "$@"; do
  case $what in
    all) f=caoa_models_v1.tar.zst ;; pipeline) f=caoa_models_v1_pipeline.tar.zst ;; data) f=caoa_data_v1.tar.zst ;;
    *) echo "unknown archive: $what (all | pipeline | data)"; exit 1 ;;
  esac
  [ -f "$f" ] || curl -fL --retry 3 -C - "$URL/$f?download=1" -o "$f"
  want=$(python3 -c "import json; print(json.load(open('packages.json'))['$f']['sha256'])")
  echo "$want  $f" | sha256sum -c -
  tar -I zstd -xf "$f"
done
