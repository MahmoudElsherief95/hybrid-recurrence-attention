"""
MQAR Breaking Point Experiment.

Tests MQAR associative recall as the gap between KV pairs and queries grows:
  - gap=0:   queries are ~33 tokens from KV pairs (WITHIN local window=64) -> LA works
  - gap=100: queries are ~133 tokens from KV pairs (BEYOND window) -> LA starts failing
  - gap=200: queries are ~233 tokens from KV pairs (FAR beyond window) -> LA fails completely

Griffin and Hawk use recurrent state to bridge any gap. Local Attention is blind beyond 64 tokens.

Results saved to results/complete_benchmark.json under keys:
  "MQAR_gap0", "MQAR_gap100", "MQAR_gap200"
"""

import sys, os, json, time
import torch
import torch.nn as nn
import numpy as np
import psutil
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.griffin import GriffinModel
from models.hawk import HawkModel
from models.local_attention import LocalAttentionModel
from data.unified_datasets import UnifiedDataset

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
BENCH_FILE   = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'complete_benchmark.json')
DEVICE       = 'cpu'
SEQ_LENGTH   = 256
LOCAL_WINDOW = 64          # must match model config
N_TRAIN      = 1000
N_VAL        = 200
BATCH_SIZE   = 16
NUM_EPOCHS   = 5           # enough to see convergence difference
LR           = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP    = 1.0

# Three stages: gap=0 (in-window), gap=100 (beyond), gap=200 (far beyond)
GAPS = [
    (0,   'MQAR_gap0',   'IN-WINDOW  (gap=0,   ~33 tokens from KV -> within window=64)'),
    (100, 'MQAR_gap100', 'EDGE       (gap=100, ~133 tokens from KV -> beyond window=64)'),
    (200, 'MQAR_gap200', 'FAILURE    (gap=200, ~233 tokens from KV -> far beyond window=64)'),
]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _forward(model, model_name, x, mask):
    if model_name in ('Hawk', 'Local Attention'):
        return model(x)
    return model(x, attention_mask=mask)


def make_models(vocab_size):
    base = dict(d_model=144, num_layers=6, vocab_size=vocab_size, max_seq_len=SEQ_LENGTH)
    return {
        'Griffin':         GriffinModel(dict(**base, local_window=LOCAL_WINDOW, num_heads=8)),
        'Hawk':            HawkModel(dict(**base, num_heads=8)),
        'Local Attention': LocalAttentionModel(dict(**base, local_window=LOCAL_WINDOW, num_heads=8)),
    }


def load_bench():
    if os.path.exists(BENCH_FILE):
        with open(BENCH_FILE) as f:
            return json.load(f)
    return {}


def save_bench(data):
    os.makedirs(os.path.dirname(BENCH_FILE), exist_ok=True)
    with open(BENCH_FILE, 'w') as f:
        json.dump(data, f, indent=2)


# ------------------------------------------------------------------
# Training loop (same as run_5_benchmarks.py)
# ------------------------------------------------------------------
def train_and_eval(model, model_name, train_loader, val_loader):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS * len(train_loader), eta_min=LR / 10)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    total_time = 0.0
    peak_cpu = 0.0
    grad_norms, step_times = [], []
    convergence_steps = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss, n_steps = 0.0, 0
        t_epoch = time.time()

        for batch in train_loader:
            x = batch['input_ids'].to(DEVICE)
            y = batch['labels'].to(DEVICE)
            mask = batch.get('attention_mask')
            if mask is not None:
                mask = mask.to(DEVICE).bool()

            t_step = time.time()
            optimizer.zero_grad()
            out = _forward(model, model_name, x, mask)
            logits = out['logits'] if isinstance(out, dict) else out
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            loss.backward()
            gn = sum(p.grad.data.norm(2).item() ** 2
                     for p in model.parameters() if p.grad is not None) ** 0.5
            grad_norms.append(gn)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_steps += 1
            step_times.append(time.time() - t_step)
            peak_cpu = max(peak_cpu, psutil.Process().memory_info().rss / 1024 / 1024)

        total_time += time.time() - t_epoch
        avg_train = epoch_loss / n_steps if n_steps else float('nan')

        # Validation
        model.eval()
        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['input_ids'].to(DEVICE)
                y = batch['labels'].to(DEVICE)
                mask = batch.get('attention_mask')
                if mask is not None:
                    mask = mask.to(DEVICE).bool()
                out = _forward(model, model_name, x, mask)
                logits = out['logits'] if isinstance(out, dict) else out
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()
                    vn += 1

        avg_val = val_loss / vn if vn else float('nan')
        ppl = float(torch.exp(torch.tensor(avg_val)).item())
        print(f"  Epoch {epoch+1:2d}/{NUM_EPOCHS} -- train {avg_train:.4f}  val {avg_val:.4f}  ppl {ppl:.3f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            convergence_steps = (epoch + 1) * len(train_loader)

    param_count = sum(p.numel() for p in model.parameters())
    best_ppl = float(torch.exp(torch.tensor(best_val_loss)).item())
    return {
        'parameters':                     param_count,
        'final_loss':                     best_val_loss,
        'perplexity':                     round(best_ppl, 4),
        'validation_loss':                best_val_loss,
        'validation_perplexity':          round(best_ppl, 4),
        'training_time_hours':            total_time / 3600,
        'convergence_steps':              convergence_steps,
        'peak_gradient_norm':             float(np.mean(grad_norms)) if grad_norms else 0.0,
        'memory_efficiency_mb_per_param': peak_cpu / param_count if param_count else 0.0,
        'latency_sec_per_step':           float(np.mean(step_times)) if step_times else 0.0,
        'throughput_samples_per_sec':     N_TRAIN / total_time if total_time else 0.0,
        'cpu_mem_peak_mb':                peak_cpu,
        'gpu_mem_peak_mb':                0,
        'model_size_mb':                  param_count * 4 / 1024 / 1024,
        'flops_per_token':                0,
        'gap_tokens':                     None,   # filled by caller
        'local_window':                   LOCAL_WINDOW,
    }


# ------------------------------------------------------------------
# Run one gap stage
# ------------------------------------------------------------------
def run_gap_stage(gap, json_key, description):
    print(f"\n{'='*65}")
    print(f"  MQAR BREAKING POINT -- {description}")
    print(f"  gap_tokens={gap}  local_window={LOCAL_WINDOW}  seq_len={SEQ_LENGTH}")
    print(f"{'='*65}")

    train_ds = UnifiedDataset('mqar', num_samples=N_TRAIN, seq_length=SEQ_LENGTH,
                              gap_tokens=gap)
    val_ds   = UnifiedDataset('mqar', num_samples=N_VAL,   seq_length=SEQ_LENGTH,
                              gap_tokens=gap)
    vocab_size = train_ds.vocab_size
    print(f"  vocab_size={vocab_size}  train={len(train_ds)}  val={len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    models = make_models(vocab_size)
    bench = load_bench()
    if json_key not in bench:
        bench[json_key] = {}

    for model_name, model in models.items():
        existing = bench[json_key].get(model_name, {})
        existing_ppl = existing.get('perplexity')
        if (isinstance(existing_ppl, (int, float))
                and existing_ppl == existing_ppl
                and existing_ppl != float('inf')):
            print(f"\n  --- {model_name} --- [SKIP -- ppl={existing_ppl}]")
            continue

        print(f"\n  --- {model_name} ---")
        try:
            metrics = train_and_eval(model, model_name, train_dl, val_dl)
            metrics['gap_tokens'] = gap
            metrics['local_window'] = LOCAL_WINDOW
            metrics['stage_description'] = description
            bench[json_key][model_name] = metrics
            print(f"  [OK] {model_name}: ppl={metrics['perplexity']}")
        except Exception as e:
            import traceback; traceback.print_exc()
            bench[json_key][model_name] = {'error': str(e)}
            print(f"  [FAIL] {model_name}: {e}")

        save_bench(bench)
        print(f"  [SAVED] -> {BENCH_FILE}")

    return bench[json_key]


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == '__main__':
    print("MQAR Breaking Point Experiment")
    print(f"local_window={LOCAL_WINDOW}  seq_len={SEQ_LENGTH}  epochs={NUM_EPOCHS}")
    print(f"Testing: gaps = {[g for g,_,_ in GAPS]}")
    print()

    for gap, json_key, desc in GAPS:
        try:
            run_gap_stage(gap, json_key, desc)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[FAIL] gap={gap}: {e}")

    print("\n=== BREAKING POINT EXPERIMENT DONE ===")

    bench = load_bench()
    print("\nSummary (best validation perplexity over 5 epochs):")
    print(f"  {'Stage':<45} {'Griffin':>10} {'Hawk':>10} {'LA':>10}")
    print("  " + "-" * 77)
    for gap, json_key, desc in GAPS:
        entry = bench.get(json_key, {})
        def ppl(m):
            r = entry.get(m, {})
            v = r.get('perplexity')
            return f"{v:.3f}" if (isinstance(v, (int, float)) and v == v) else "---"
        print(f"  {desc:<45} {ppl('Griffin'):>10} {ppl('Hawk'):>10} {ppl('Local Attention'):>10}")

    print()
    print("Expected pattern:")
    print("  gap=0  (in-window):  G~1.01  H~1.00  LA~1.00  (all can see KVs)")
    print("  gap=100 (beyond):   G~1.01  H~1.01  LA>>1    (LA starts failing)")
    print("  gap=200 (far):      G~1.01  H~1.01  LA>>5    (LA completely blind)")
