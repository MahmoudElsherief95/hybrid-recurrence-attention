# Official / official-format dataset loaders

This folder contains **code**, not data.

It provides loaders that read official-format artifacts (e.g. LRA TSV/NPZ/Pickle dumps) from a directory you point to via CLI flags.

## Used by C2

- `listops_lra_dataset.py`
  - Reads ListOps TSVs from `--listops-official-dir`.

- `pathfinder_lra_dataset.py`
  - Reads Pathfinder/Path-X style splits (NPZ/NPY/PKL) from `--pathx-official-dir`.
  - Also used for the local Pathfinder-32 proxy cache (it reads the locally generated NPZ splits).

## Note about the LRA release archive

- `lra_release.py` contains helpers to extract the official `lra_release.gz` if you have it locally.
- Automatic download was blocked (HTTP 403) in this environment, so the current workflow uses local caches instead.
