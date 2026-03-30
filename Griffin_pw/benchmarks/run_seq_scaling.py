"""
Sequence-length scaling experiment for MQAR and Induction Heads.

Tests each dataset at seq_len = 256 / 512 / 1024.
seq_len=256 already exists in complete_benchmark.json → will be skipped.
Only 512 and 1024 are trained from scratch.

JSON keys produced:
  "MQAR_seq256"          (SKIP — already "MQAR")
  "MQAR_seq512"
  "MQAR_seq1024"
  "induction_heads_seq256"   (SKIP — already "induction_heads")
  "induction_heads_seq512"
  "induction_heads_seq1024"

The scaling story this demonstrates:
  As seq_len grows, the critical dependency gap between KV-pairs and
  their queries (MQAR) / between the AB pattern and its repeat
  (Induction Heads) grows beyond local_window=64. Local Attention
  degrades steadily; Hawk/Griffin hold stable → Claim 2 scaling.

Run from Griffin_pw/:
    python run_seq_scaling.py
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
N_TRAIN      = 1000
N_VAL        = 200
BATCH_SIZE   = 16
NUM_EPOCHS   = 1
LR           = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP    = 1.0
D_MODEL      = 144
LOCAL_WINDOW = 64

SEQ_LENGTHS  = [256, 512, 1024]

# Existing base keys that already hold the seq=256 results → skip those
EXISTING_KEYS = {
    'mqar':            'MQAR',
    'induction_heads': 'induction_heads',
}

BENCHMARKS = [
    ('mqar',            {}),
    ('induction_heads', {}),
]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _forward(model, model_name, x, mask):
    if model_name in ('Hawk', 'Local Attention'):
        return model(x)
    return model(x, attention_mask=mask)


def make_models(vocab_size, seq_len):
    base = dict(d_model=D_MODEL, num_layers=6, vocab_size=vocab_size,
                max_seq_len=seq_len)
    return {
        'Griffin':         GriffinModel(dict(**base, local_window=LOCAL_WINDOW, num_heads=8)),
        'Hawk':            HawkModel(dict(**base, num_heads=8)),
        'Local Attention': LocalAttentionModel(dict(**base, local_window=LOCAL_WINDOW,
                                                    num_heads=8)),
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
# Training loop (identical to run_5_benchmarks.py)
# ------------------------------------------------------------------
def train_and_eval(model, model_name, train_loader, val_loader, n_train):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS * len(train_loader), eta_min=LR / 10)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    total_time = 0.0
    peak_cpu = 0.0
    grad_norms = []
    step_times = []
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

        # -- validation --
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
        print(f"    Epoch {epoch+1:2d}/{NUM_EPOCHS}  train={avg_train:.4f}  "
              f"val={avg_val:.4f}  ppl={ppl:.4f}")

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
        'throughput_samples_per_sec':     n_train / total_time if total_time else 0.0,
        'cpu_mem_peak_mb':                peak_cpu,
        'gpu_mem_peak_mb':                0,
        'model_size_mb':                  param_count * 4 / 1024 / 1024,
        'flops_per_token':                0,
    }


# ------------------------------------------------------------------
# Run one (dataset, seq_len) combination
# ------------------------------------------------------------------
def run_one(ds_name, json_key, seq_len, extra_kwargs):
    print(f"\n{'='*62}")
    print(f"  {json_key}   seq_len={seq_len}")
    print(f"{'='*62}")

    train_ds   = UnifiedDataset(ds_name, num_samples=N_TRAIN,
                                seq_length=seq_len, **extra_kwargs)
    val_ds     = UnifiedDataset(ds_name, num_samples=N_VAL,
                                seq_length=seq_len, **extra_kwargs)
    vocab_size = train_ds.vocab_size
    print(f"  vocab_size={vocab_size}  train={len(train_ds)}  val={len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          drop_last=False)

    bench = load_bench()
    if json_key not in bench:
        bench[json_key] = {}

    models = make_models(vocab_size, seq_len)
    for model_name, model in models.items():
        existing_ppl = bench[json_key].get(model_name, {}).get('perplexity')
        if (isinstance(existing_ppl, (int, float))
                and existing_ppl == existing_ppl
                and existing_ppl != float('inf')):
            print(f"\n  [SKIP] {model_name}: ppl={existing_ppl} already saved")
            continue

        print(f"\n  --- {model_name} ---")
        try:
            metrics = train_and_eval(model, model_name, train_dl, val_dl,
                                     n_train=N_TRAIN)
            bench[json_key][model_name] = metrics
            print(f"  [OK] {model_name}: ppl={metrics['perplexity']}")
        except Exception as e:
            import traceback; traceback.print_exc()
            bench[json_key][model_name] = {'error': str(e)}
            print(f"  [FAIL] {model_name}: {e}")

        save_bench(bench)
        print(f"  [SAVED] -> {BENCH_FILE}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print(f"Sequence-length scaling benchmark")
    print(f"Datasets: MQAR, Induction Heads")
    print(f"seq_lengths: {SEQ_LENGTHS}")
    print(f"epochs={NUM_EPOCHS}  n_train={N_TRAIN}  device={DEVICE}\n")

    for ds_name, extra_kwargs in BENCHMARKS:
        base_key = EXISTING_KEYS[ds_name]
        for seq_len in SEQ_LENGTHS:
            if seq_len == 256:
                # Results already exist under the original key — skip training,
                # just copy a pointer comment into the scaling key if needed.
                scaling_key = f"{base_key}_seq256"
                bench = load_bench()
                if scaling_key not in bench and base_key in bench:
                    bench[scaling_key] = {
                        "_note": f"See '{base_key}' for seq_len=256 results "
                                 f"(already trained, identical config)."
                    }
                    save_bench(bench)
                    print(f"  [POINTER] {scaling_key} -> '{base_key}'")
                continue

            scaling_key = f"{base_key}_seq{seq_len}"
            run_one(ds_name, scaling_key, seq_len, extra_kwargs)

    # -- Summary --
    bench = load_bench()
    print(f"\n{'='*62}")
    print(f"  SCALING SUMMARY  (perplexity ↓ better)")
    print(f"{'='*62}")
    print(f"  {'Dataset':<30} {'seq':>6} {'Griffin':>10} {'Hawk':>10} {'LA':>10}")
    print(f"  {'-'*67}")

    display = [
        ('MQAR',            'MQAR',            256),
        ('MQAR',            'MQAR_seq512',      512),
        ('MQAR',            'MQAR_seq1024',    1024),
        ('Induction Heads', 'induction_heads',  256),
        ('Induction Heads', 'induction_heads_seq512',  512),
        ('Induction Heads', 'induction_heads_seq1024', 1024),
    ]
    for label, key, sl in display:
        e  = bench.get(key, {})
        if '_note' in e:
            continue
        g  = e.get('Griffin',        {}).get('perplexity', '?')
        h  = e.get('Hawk',           {}).get('perplexity', '?')
        la = e.get('Local Attention', {}).get('perplexity', '?')
        print(f"  {label:<30} {sl:>6} {str(g):>10} {str(h):>10} {str(la):>10}")


if __name__ == '__main__':
    main()
