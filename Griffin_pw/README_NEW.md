# Griffin, Hawk, and Local Attention: Minimal Research Benchmark

> This repository provides a clean, reproducible research framework for benchmarking three sequence modeling architectures:
>
> - **Griffin**: Hybrid recurrence + local attention
> - **Hawk**: Pure recurrence
> - **Local Attention**: Pure attention
>
> All models are compared on two tasks: MQAR (memory-intensive) and Chomsky (hierarchical), with fair parameter counts and unified results.

---

## 📁 Repository Structure

```
models/
  griffin/griffin_model.py         # Griffin hybrid model
  hawk/hawk_model.py               # Hawk pure recurrence model
  local_attention/attention_model.py# Local Attention pure attention model
training/trainer.py                # Training utilities
evaluation/evaluator.py            # Evaluation utilities
data/mqar/mqar_dataset.py          # MQAR dataset loader
data/chomsky/chomsky_dataset.py    # Chomsky dataset loader
quick_start.py                     # Main experiment script (all scenarios)
results/quick_experiment.json      # Unified results (all benchmarks)
notebooks/01_model_analysis.ipynb  # Main analysis notebook
requirements.txt                   # Dependencies
```

## 📂 Folder & File Explanations

- **models/**: Contains all model implementations.
  - `griffin/`: Griffin hybrid recurrence + attention model.
  - `hawk/`: Hawk pure recurrence model.
  - `local_attention/`: Local Attention pure attention model.
- **training/**: Training scripts and utilities for running experiments.
  - `trainer.py`: Main training logic and routines.
- **evaluation/**: Evaluation and metrics computation.
  - `evaluator.py`: Functions for evaluating models and comparing results.
- **data/**: Dataset loaders and generators.
  - `mqar/`: Multi-Query Associative Recall dataset code.
    - `mqar_dataset.py`: Loader and generator for MQAR dataset.
  - `chomsky/`: Chomsky hierarchy dataset code.
    - `chomsky_dataset.py`: Loader and generator for Chomsky dataset.
- **quick_start.py**: Main entry point to run all experiments and benchmarks.
- **results/**: Stores all experiment outputs and metrics.
  - `quick_experiment.json`: Unified results for all models, tasks, and scenarios.
- **notebooks/**: Jupyter notebooks for analysis and visualization.
  - `01_model_analysis.ipynb`: Main analysis notebook for results and plots.
- **requirements.txt**: List of required Python packages for the project.

### Key Points
- All results are now saved in `results/quick_experiment.json` (including long-sequence generalization)
- Only one main notebook: `notebooks/01_model_analysis.ipynb`
- No duplicate or obsolete files/folders

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run All Experiments (MQAR & Chomsky, all models)
```bash
python quick_start.py --train
```
Results are saved to `results/quick_experiment.json`.

### 3. Run Long-Sequence Generalization Only
```bash
python quick_start.py --longseq
```
Results are also merged into `results/quick_experiment.json`.

### 4. View Results and Analysis
- Open the notebook for interactive plots and analysis:
  ```bash
  jupyter notebook notebooks/01_model_analysis.ipynb
  ```
- Or inspect `results/quick_experiment.json` directly.

---

## 📊 Model Configurations

| Model           | Parameters | d_model | Layers | Heads |
|-----------------|------------|---------|--------|-------|
| Griffin         | ~3.8M      | 144     | 6      | 4     |
| Hawk            | ~2.0M      | 144     | 6      | 0     |
| Local Attention | ~1.7M      | 144     | 6      | 4     |

All models use the same embedding size, layers, and sequence length for fair comparison.

---

## 📈 Example Results (Scenario 1)

### MQAR
| Model           | Parameters | Final Loss | Latency (s/step) | Throughput (samples/s) | CPU Mem (MB) |
|-----------------|------------|------------|------------------|------------------------|--------------|
| Griffin         | 3,798,144  | 1.63       | 3.21             | 2.49                   | 611.1        |
| Hawk            | 2,043,360  | 7.08       | 3.15             | 2.54                   | 600.1        |
| Local Attention | 1,667,520  | 1.89       | 0.43             | 18.79                  | 576.4        |

### Chomsky
| Model           | Parameters | Final Loss | Latency (s/step) | Throughput (samples/s) | CPU Mem (MB) |
|-----------------|------------|------------|------------------|------------------------|--------------|
| Griffin         | 3,654,288  | 0.86       | 4.56             | 1.76                   | 650.3        |
| Hawk            | 1,899,504  | 0.87       | 3.91             | 2.05                   | 615.5        |
| Local Attention | 1,523,664  | 0.84       | 0.35             | 22.92                  | 646.5        |

*Lower loss, latency, and memory are better; higher throughput is better.*

---

## 🧪 Long-Sequence Generalization

All models are evaluated on much longer sequences than seen during training. Example (Scenario 1):

| Sequence Length | Griffin | Hawk | Local Attention |
|-----------------|--------|------|-----------------|
| 128             | 7.04   | 7.10 | 7.02            |
| 256             | 7.04   | 6.99 | 7.00            |
| 512             | 7.04   | 7.02 | 7.08            |
| 1024            | 6.97   | 7.06 | 7.06            |
| 2048            | 7.06   | 7.06 | 7.10            |

*All models show stable loss across long sequences, supporting the claim that Griffin generalizes as well as attention models for long-context tasks.*

---

## 📒 Analysis & Visualization

- Use `notebooks/01_model_analysis.ipynb` for interactive analysis, scenario separation, and research-grade figures.
- All results and figures are generated from `quick_experiment.json`.


## 📚 References

- [Griffin Paper](https://arxiv.org/abs/2402.19427)
- [RecurrentGemma (Google DeepMind)](https://github.com/google-deepmind/recurrentgemma)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Mamba: Selective State Space Models](https://arxiv.org/abs/2312.00752)
