"""
Copy task at increasing sequence lengths to demonstrate Local Attention's
window-size failure.

Tests three configurations:
  copy_len=100  seq_len=256   (copy > window=64,  short padding)
  copy_len=256  seq_len=512   (copy >> window=64, medium padding)
  copy_len=512  seq_len=1024  (copy >>> window=64, model must recall
                               tokens from 512 positions back)

Expected result:
  Griffin and Hawk stay competitive (recurrence spans arbitrary distance).
  Local Attention perplexity degrades sharply as copy_len >> window (64).

Saves results to results/complete_benchmark.json under keys:
  "copy_len100_seq256"   (same as existing "copy" benchmark, re-run for comparison)
  "copy_len256_seq512"
  "copy_len512_seq1024"

Run from Griffin_pw/:
    python run_copy_seq_len.py
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
from data.copy.copy_dataset import CopyDataset

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
BENCH_FILE   = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'complete_benchmark.json')
DEVICE       = 'cpu'
D_MODEL      = 144
NUM_LAYERS   = 6
LOCAL_WINDOW = 64
NUM_HEADS    = 8
N_TRAIN      = 1000
N_VAL        = 200
BATCH_SIZE   = 8        # smaller batch to handle longer sequences
NUM_EPOCHS   = 5
LR           = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP    = 1.0

# Three configurations — each progressively harder for Local Attention
CONFIGS = [
    # (json_key,          copy_len, seq_len)
    ("copy_len100_seq256",  100,     256),
    ("copy_len256_seq512",  256,     512),
    ("copy_len512_seq1024", 512,    1024),
]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _forward(model, model_name, x, mask):
    if model_name in ('Hawk', 'Local Attention'):
        return model(x)
    return model(x, attention_mask=mask)


def make_models(vocab_size, seq_len):
    base = dict(d_model=D_MODEL, num_layers=NUM_LAYERS, vocab_size=vocab_size,
                max_seq_len=seq_len)
    return {
        'Griffin':         GriffinModel(dict(**base, local_window=LOCAL_WINDOW, num_heads=NUM_HEADS)),
        'Hawk':            HawkModel(dict(**base, num_heads=NUM_HEADS)),
        'Local Attention': LocalAttentionModel(dict(**base, local_window=LOCAL_WINDOW, num_heads=NUM_HEADS)),
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
# Training loop
# ------------------------------------------------------------------
def train_and_eval(model, model_name, train_loader, val_loader):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS * len(train_loader), eta_min=LR / 10)
    criterion = nn.CrossEntropyLoss()

    best_val_loss     = float('inf')
    total_time        = 0.0
    peak_cpu          = 0.0
    grad_norms        = []
    step_times        = []
    convergence_steps = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss, n_steps = 0.0, 0
        t_epoch = time.time()

        for batch in train_loader:
            x    = batch['input_ids'].to(DEVICE)
            y    = batch['labels'].to(DEVICE)
            mask = batch.get('attention_mask')
            if mask is not None:
                mask = mask.to(DEVICE).bool()

            t_step = time.time()
            optimizer.zero_grad()
            out    = _forward(model, model_name, x, mask)
            logits = out['logits'] if isinstance(out, dict) else out
            loss   = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

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
            n_steps    += 1
            step_times.append(time.time() - t_step)
            peak_cpu = max(peak_cpu, psutil.Process().memory_info().rss / 1024 / 1024)

        total_time += time.time() - t_epoch
        avg_train   = epoch_loss / n_steps if n_steps else float('nan')

        # ---- validation ----
        model.eval()
        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x    = batch['input_ids'].to(DEVICE)
                y    = batch['labels'].to(DEVICE)
                mask = batch.get('attention_mask')
                if mask is not None:
                    mask = mask.to(DEVICE).bool()
                out    = _forward(model, model_name, x, mask)
                logits = out['logits'] if isinstance(out, dict) else out
                loss   = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()
                    vn       += 1

        avg_val = val_loss / vn if vn else float('nan')
        ppl     = float(torch.exp(torch.tensor(avg_val)).item())
        print(f"  Epoch {epoch+1:2d}/{NUM_EPOCHS} -- train {avg_train:.4f}  val {avg_val:.4f}  ppl {ppl:.3f}")

        if avg_val < best_val_loss:
            best_val_loss     = avg_val
            convergence_steps = (epoch + 1) * len(train_loader)

    param_count = sum(p.numel() for p in model.parameters())
    best_ppl    = float(torch.exp(torch.tensor(best_val_loss)).item())
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
    }


# ------------------------------------------------------------------
# Run one config
# ------------------------------------------------------------------
def run_config(json_key, copy_len, seq_len):
    print(f"\n{'='*65}")
    print(f"  Copy task: copy_len={copy_len}  seq_len={seq_len}  window={LOCAL_WINDOW}")
    print(f"  Ratio copy_len/window = {copy_len/LOCAL_WINDOW:.1f}x   key={json_key}")
    print(f"{'='*65}")

    vocab_size = 1002   # matches CopyDataset default (vocab_size=1000 + 2 special)

    train_ds = CopyDataset(num_sequences=N_TRAIN, seq_len=seq_len,
                           copy_length=copy_len, seed=42)
    val_ds   = CopyDataset(num_sequences=N_VAL,   seq_len=seq_len,
                           copy_length=copy_len, seed=123)
    print(f"  vocab_size={train_ds.total_vocab_size}  train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    bench = load_bench()
    if json_key not in bench:
        bench[json_key] = {}

    models = make_models(train_ds.total_vocab_size, seq_len)
    for name, model in models.items():
        if bench[json_key].get(name, {}).get('perplexity') is not None:
            ppl = bench[json_key][name]['perplexity']
            print(f"  [SKIP] {name}: already ppl={ppl}")
            continue

        print(f"\n  --- {name} ---")
        result = train_and_eval(model, name, train_loader, val_loader)
        print(f"  [OK] {name}: ppl={result['perplexity']}")

        bench[json_key][name] = result
        save_bench(bench)
        print(f"  [SAVED] -> {BENCH_FILE}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print(f"Copy seq-length scaling experiment")
    print(f"Local Attention window = {LOCAL_WINDOW}")
    print(f"epochs={NUM_EPOCHS}  n_train={N_TRAIN}  n_val={N_VAL}  batch={BATCH_SIZE}\n")
    print(f"  copy_len/window=1.6x  ->  2x harder  ->  8x harder\n")

    for json_key, copy_len, seq_len in CONFIGS:
        run_config(json_key, copy_len, seq_len)

    # ---- Summary ----
    bench = load_bench()
    print(f"\n{'='*65}")
    print(f"COPY TASK: perplexity vs sequence length (window={LOCAL_WINDOW})")
    print(f"  (lower perplexity = better;  LA degrades as copy_len >> window)")
    print(f"{'='*65}")
    print(f"  {'Config':<26} {'copy/win':>8}  {'Griffin':>10} {'Hawk':>10} {'LA':>10}")
    print(f"  {'-'*62}")
    for json_key, copy_len, seq_len in CONFIGS:
        e    = bench.get(json_key, {})
        g    = e.get('Griffin',        {}).get('perplexity', '?')
        h    = e.get('Hawk',           {}).get('perplexity', '?')
        la   = e.get('Local Attention', {}).get('perplexity', '?')
        rat  = f"{copy_len/LOCAL_WINDOW:.1f}x"
        label = f"copy{copy_len}_seq{seq_len}"
        print(f"  {label:<26} {rat:>8}  {str(g):>10} {str(h):>10} {str(la):>10}")


if __name__ == '__main__':
    main()
