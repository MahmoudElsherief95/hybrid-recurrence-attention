# Griffin vs Hawk vs Local Attention : Benchmark Suite

An empirical comparison of three small sequence model architectures on synthetic benchmarks designed to isolate the specific capabilities each architecture is built for. The central question is whether Griffin's hybrid design (recurrence + local attention) delivers the best of both worlds without trading one off against the other.

All three models are matched at approximately 2M parameters for a fair architectural comparison.

---

## Architecture Overview

| Model | Type | Memory Mechanism | d_model | Layers | Local Window |
|---|---|---|---|---|---|
| Griffin | Hybrid | Recurrent state + local attention | 144 | 6 | 64 tokens |
| Hawk | Pure recurrence | Recurrent hidden state only | 144 | 6 | N/A |
| Local Attention | Pure attention | Local attention window only | 144 | 6 | 64 tokens |

---

## Experimental Design

This benchmark suite focuses on a **small, repeatable set of tasks** that isolate different sequence-processing demands (associative recall, long-context LM, exact copying, and structured classification). The goal is to compare whether the **hybrid** design (recurrence + local attention) can match the strengths of pure recurrence and pure local attention without regressing.

---

## Requirements

- Python 3.9+
- PyTorch 2.0+

```bash
pip install -r requirements.txt
```

---

## Running the Benchmarks

This repo’s **maintained** benchmark runner is:

```bash
python benchmarks/run_selected_benchmarks.py
```

Entrypoint file:

- `Griffin_pw/benchmarks/run_selected_benchmarks.py`

Windows path (example):

- `C:\Users\Elsherief\Documents\hybrid-recurrence-attention-1\Griffin_pw\benchmarks\run_selected_benchmarks.py`

It runs the **exact 8 datasets used in** `notebooks/model_analysis.ipynb` and writes the canonical results file:

- `results/complete_benchmark.json`

Note:
- `results/complete_benchmark.json` is the only file intended to be treated as the **canonical output**.
- `results/trials/` is for intermediate runs, merged files, and backups.

### Quickstart (recommended)

From `Griffin_pw/`:

```bash
python benchmarks/run_selected_benchmarks.py --run-id 1
```

### Run a specific dataset (for a quick test)

List available task keys:

```bash
python benchmarks/run_selected_benchmarks.py --list-tasks
```

Run one dataset:

```bash
python benchmarks/run_selected_benchmarks.py --tasks MQAR
python benchmarks/run_selected_benchmarks.py --tasks path_x_clf
```

Run a small subset:

```bash
python benchmarks/run_selected_benchmarks.py --tasks MQAR,copy_len100_seq256,listops_clf
```

Notes:
- PG-19 is **optional** and is skipped unless you pass `--pg19-root` or `--pg19-zip`.
- Results are written after **every model**, so runs are crash-safe.
- If `results/complete_benchmark.json` already exists, the runner will **resume** and skip entries that are already present.
    Use `--overwrite` (or `--out results/some_new_file.json`) to start fresh.

---

## Using Official Upstream Datasets (recommended)

This repo includes several synthetic/proxy datasets. When available, the selected runner supports using official upstream sources.

### Option A — Point to an existing LRA ListOps directory

If you already have the LRA TSV files (e.g. `basic_train.tsv`, `basic_val.tsv`, `basic_test.tsv`), run:

```bash
python benchmarks/run_selected_benchmarks.py --tasks listops_clf \
    --listops-source lra --listops-official-dir /path/to/listops_tsv_dir
```

### Option B — Auto-download the LRA public release archive

```bash
python benchmarks/run_selected_benchmarks.py --tasks listops_clf \
    --listops-source lra --download-official
```

Notes:
- Downloading can be large; the archive is cached under `data_cache/` (inside `Griffin_pw/`) by default.
- Other tasks in this suite are still synthetic/proxy unless explicitly wired to an upstream source.

### If download fails (e.g. HTTP 403)

Some networks block direct downloads from `storage.googleapis.com`. If `--download-official` fails, download `lra_release.gz` manually (browser or `curl.exe`) and point the code at the local file:

- PowerShell example:

```powershell
$env:LRA_RELEASE_ARCHIVE_PATH = "C:\path\to\lra_release.gz"
python benchmarks/run_selected_benchmarks.py --tasks listops_clf --listops-source lra
```

If the archive is access-controlled (AccessDenied) and you cannot download it at all, you can still generate **official-format ListOps TSV** locally using the generator (official ListOps is defined by the generator + protocol):

In that case, you can still run the benchmark suite using the synthetic/proxy datasets (default settings).

### PG-19 (official split)

PG-19 is a byte-level LM task and requires local data.

```bash
python benchmarks/run_selected_benchmarks.py --tasks PG19 --pg19-root /path/to/pg19
# or
python benchmarks/run_selected_benchmarks.py --tasks PG19 --pg19-zip /path/to/pg19.zip
```

---

## Datasets in the notebook subset

The selected runner executes these 8 tasks:

### Task list

| Dataset | Task key | Primary metric in analysis |
|---|---|---|
| MQAR (seq=256) | `MQAR` | Perplexity ↓ |
| MQAR (seq=512) | `MQAR_seq512` | Perplexity ↓ |
| MQAR (seq=1024) | `MQAR_seq1024` | Perplexity ↓ |
| PG-19 (byte LM, optional) | `PG19` | Perplexity ↓ |
| Copy (len=100, seq=256) | `copy_len100_seq256` | Perplexity ↓ |
| Path-X (classification) | `path_x_clf` | Classification perplexity (val) ↓ |
| Chomsky (classification) | `chomsky_clf` | Classification perplexity (val) ↓ |
| ListOps (classification) | `listops_clf` | Classification perplexity (val) ↓ |

---

## Repository Structure

```
Griffin_pw/
├── benchmarks/
│   └── run_selected_benchmarks.py     # Canonical runner (8 notebook datasets)
│
├── data/                              # Dataset implementations
│   ├── unified_datasets.py            # Shared dataset interface for LM runners
│   ├── aan_retrieval/
│   ├── chomsky/
│   ├── copy/
│   ├── induction_heads/
│   ├── listops/
│   ├── mqar/
│   ├── needle_in_haystack/
│   ├── path_x/
│   ├── selective_copy/
│   ├── sequential_niah/
│   └── variable_binding/
│
├── models/                            # Model definitions
│   ├── griffin/                       # Hybrid recurrence + local attention
│   ├── hawk/                          # Pure recurrence
│   └── local_attention/               # Local attention only
│
├── training/
│   └── trainer.py
│
├── evaluation/
│   └── evaluator.py
│
├── results/
│   └── complete_benchmark.json        # Canonical results file (generated)
│
├── notebooks/
│   └── model_analysis.ipynb           # Full results analysis and visualization
│
└── requirements.txt
```

---

## Results Summary

All results are stored in `results/complete_benchmark.json`. To load and inspect them:

```python
import json

with open("results/complete_benchmark.json") as f:
    results = json.load(f)


def get_task_block(results_dict: dict, task_key: str) -> dict:
    """Find a task block regardless of which top-level section it lives in."""
    for section_key, section_val in results_dict.items():
        if section_key == "_meta" or not isinstance(section_val, dict):
            continue
        task_block = section_val.get(task_key)
        if isinstance(task_block, dict):
            return task_block
    raise KeyError(f"Task '{task_key}' not found")

# Example: MQAR perplexity across models
mqar = get_task_block(results, "MQAR")
for model, data in mqar.items():
    if not isinstance(data, dict):
        continue
    ppl = data.get("validation_perplexity", data.get("perplexity", float("nan")))
    print(f"{model:20s}  ppl={ppl:.2f}")

# Example: classification perplexity (val) from explicit field, or exp(validation_loss)
import math

pathx = get_task_block(results, "path_x_clf")
for model, data in pathx.items():
    if not isinstance(data, dict):
        continue
    clfppl = data.get("validation_classification_perplexity", data.get("classification_perplexity"))
    if not isinstance(clfppl, (int, float)):
        val_loss = data.get("validation_loss")
        clfppl = math.exp(val_loss) if isinstance(val_loss, (int, float)) else float("nan")
    acc = data.get("val_accuracy_pct", float("nan"))
    print(f"{model:20s}  clfppl={clfppl:.2f}  acc={acc:.2f}%")
```

For a full analysis with tables and figures, open `notebooks/model_analysis.ipynb`.

---
### Key findings (high level)

- The suite is designed so each task stresses a particular capability (long-range recall, local precision, or structured classification).
- The notebook reports **perplexity-only** for the main comparison table; classification uses **classification perplexity (val)** (with `exp(validation_loss)` as fallback when needed).
- The most reliable way to interpret results is via `notebooks/model_analysis.ipynb`, which generates the tables/figures directly from `results/complete_benchmark.json`.

---

## Useful commands (end-to-end + single-task)

List available task keys:

```bash
python benchmarks/run_selected_benchmarks.py --list-tasks
```

Run a single dataset (quick test):

```bash
python benchmarks/run_selected_benchmarks.py --tasks MQAR
python benchmarks/run_selected_benchmarks.py --tasks path_x_clf
```

Run a small subset:

```bash
python benchmarks/run_selected_benchmarks.py --tasks MQAR,copy_len100_seq256,listops_clf
```

---

## Training Configuration

All benchmarks use the following shared hyperparameters:

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 0.01 |
| Gradient clip | 1.0 |
| LR schedule | CosineAnnealingLR |
| Batch size | 16 |
| Train samples | 1,000 |
| Val samples | 200 |
| Epochs (LM tasks) | 5 |
| Epochs (scaling tasks) | 1 |
| Device | CPU |

---

## References

De et al. (2024). *Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models.* arXiv:2402.19427

Botev et al. (2024). *RecurrentGemma: Moving Past Transformers for Efficient Open Language Models.* arXiv:2404.07839
