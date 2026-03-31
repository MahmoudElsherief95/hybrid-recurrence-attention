# Griffin vs Hawk vs Local Attention — Benchmark Suite

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

Benchmarks are split into two groups based on what the task structurally demands:

- C1 — Recurrence Dominant: Tasks where relevant context lies beyond the 64-token window. Recurrent models (Griffin, Hawk) should dominate; Local Attention should degrade as sequence length grows.
- C2 — Local Attention Dominant: Tasks requiring both local precision and broader reach. Griffin's hybrid design should outperform both pure baselines.

---

## Requirements

- Python 3.9+
- PyTorch 2.0+

```bash
pip install -r requirements.txt
```

---

## Running the Benchmarks

### Run everything

```bash
python run_all_benchmarks.py
```

This runs all six benchmark suites sequentially. Results accumulate in `results/complete_benchmark.json`. The run is crash-safe: each model's result is written immediately after training, so a partial run can be resumed without re-running completed entries.

### Run individual suites

```bash
# Core language modelling tasks (MQAR, Induction Heads, Copy, Selective Copy, NIAH)
python run_5_benchmarks.py

# AAN long-range recall test (400-token filler gap)
python run_aan_benchmark.py

# Classification tasks (Chomsky, Path-X, ListOps)
python run_classification_benchmarks.py

# Copy task at increasing sequence lengths
python run_copy_seq_len.py

# MQAR + Induction Heads at seq = 256 / 512 / 1024
python run_seq_scaling.py

# MQAR gap-size ablation (find Local Attention's exact failure point)
python run_mqar_breaking_point.py
```

All scripts are located in the project root. Equivalent copies are also available under `benchmarks/`.

---

## Benchmark Suites

### 1 · Core LM Benchmarks — `run_5_benchmarks.py`

Language modelling tasks; metric is perplexity (lower is better).

| Dataset | JSON key | What it tests |
|---|---|---|
| MQAR | `MQAR` | Multi-query associative recall — KV pairs spread > 64 tokens apart |
| Induction Heads | `induction_heads` | Copy-from-context induction pattern |
| Copy | `copy` | Exact copy of a prefix (copy_len=100 > window=64) |
| Selective Copy | `selective_copy` | Copy only marked tokens — requires selection and memory |
| Sequential NIAH | `sequential_niah` | Multi-needle sequential retrieval |

### 2 · AAN Long-Range Test — `run_aan_benchmark.py`

Forces recall across a 400-token filler gap (> 384 tokens = 6 × the 64-token local window). JSON key: `aan_long_range`.

### 3 · Classification Benchmarks — `run_classification_benchmarks.py`

Metric is validation accuracy % (higher is better). A classification head is trained on the final hidden state.

| Dataset | JSON key | Classes | What it tests |
|---|---|---|---|
| Chomsky CFG | `chomsky_clf` | 2 | Balanced-parentheses recognition |
| Path-X | `path_x_clf` | 2 | Long-range pixel-path connectivity |
| ListOps | `listops_clf` | 10 | Nested MIN/MAX/SUM expression evaluation |

### 4 · Copy Seq-Length Scaling — `run_copy_seq_len.py`

Runs the copy task at increasing lengths to show Local Attention degrading as `copy_len` exceeds the window.

| Config | JSON key | Note |
|---|---|---|
| copy_len=100, seq=256 | `copy_len100_seq256` | copy span > window |
| copy_len=256, seq=512 | `copy_len256_seq512` | copy span >> window |
| copy_len=512, seq=1024 | `copy_len512_seq1024` | copy span >>> window |

### 5 · Sequence-Length Scaling — `run_seq_scaling.py`

MQAR and Induction Heads at three sequence lengths to demonstrate recurrence scalability.

| Dataset | seq_len | JSON key |
|---|---|---|
| MQAR | 256 | `MQAR` |
| MQAR | 512 | `MQAR_seq512` |
| MQAR | 1024 | `MQAR_seq1024` |
| Induction Heads | 256 | `induction_heads` |
| Induction Heads | 512 | `induction_heads_seq512` |
| Induction Heads | 1024 | `induction_heads_seq1024` |

### 6 · MQAR Gap Ablation — `run_mqar_breaking_point.py`

Varies the gap between KV pairs and their queries to locate Local Attention's exact breaking point.

| Gap | JSON key |
|---|---|
| 0 | `MQAR_gap0` |
| 100 | `MQAR_gap100` |
| 200 | `MQAR_gap200` |

---

## Repository Structure

```
Griffin_pw/
├── run_all_benchmarks.py              # Master runner — executes all 6 suites
├── run_5_benchmarks.py                # Core LM benchmarks
├── run_aan_benchmark.py               # AAN long-range test
├── run_classification_benchmarks.py   # Chomsky / Path-X / ListOps
├── run_copy_seq_len.py                # Copy task scaling
├── run_seq_scaling.py                 # MQAR + Induction Heads scaling
├── run_mqar_breaking_point.py         # MQAR gap ablation
├── griffin.yaml                       # Griffin model config
├── hawk.yaml                          # Hawk model config
├── local_attention.yaml               # Local Attention model config
│
├── benchmarks/                        # Duplicate copies of benchmark runners
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
│   ├── complete_benchmark.json        # Canonical results file
│   ├── benchmark_performance.csv      # Performance pivot (auto-generated)
│   ├── benchmark_efficiency.csv       # Efficiency metrics (auto-generated)
│   ├── figures/                       # Saved plots
│   └── logs/                          # Per-run training logs
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

# Example: MQAR perplexity across models
for model, data in results["group_1_recurrence_dominant"]["MQAR"].items():
    print(f"{model:20s}  ppl={data['perplexity']:.4f}")
```

For a full analysis with tables and figures, open `notebooks/model_analysis.ipynb`.

### Key findings

C1 : Recurrence Dominant (MQAR):

| Task | Griffin | Hawk | Local Attention |
|---|---|---|---|
| MQAR (seq=256) | 1.014 | 1.007 | 1.251 |
| MQAR (seq=512) | 1.227 | 1.082 | 1.771 |
| MQAR (seq=1024) | 1.125 | 1.052 | 1.341 |

Griffin stays within 0.15 PPL of Hawk across all sequence lengths — the hybrid does not regress on recurrence quality.

C2 : Local Attention Dominant:

| Task | Griffin | Hawk | Local Attention |
|---|---|---|---|
| copy_len100 (PPL) | 2.86 | 15.03 | 13.37 |
| Path-X (Acc %) | 99.5 | 57.0 | 98.5 |
| Chomsky (Acc %) | 78.5 | 66.0 | 77.0 |
| ListOps (Acc %) | 45.0 | 25.0 | 45.0 |

Griffin uniquely solves `copy_len100` (copy span > local window), achieving 5× lower perplexity than either pure model.

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
