# Synthetic Path-X (proxy)

This folder contains a **synthetic proxy** Path-X generator (`path_x_dataset.py`).

## Important

- The *official-like* C2 workflow does **not** use this synthetic generator.
- For C2 Path-X you should use:
  - `--pathx-source local` (local Pathfinder-32 hard proxy), or
  - `--pathx-source lra` (official LRA release, if you have the archive locally).

See: `Griffin_pw/data/official/pathfinder_lra_dataset.py` and `Griffin_pw/data/local/pathfinder32_hard_generator.py`.
