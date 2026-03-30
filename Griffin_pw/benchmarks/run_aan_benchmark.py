"""
AAN Forced Long-Range Recall Benchmark
=======================================
Purpose: prove Local Attention's 64-token window limitation.

Task design:
  Position   0..49  : 50 random "needle" tokens (vocab 1..50)
  Position  50..449 : 400 filler tokens (token 51) -- the gap
  Position 450      : SEP token (52)
  Position 451..500 : same 50 needles must be recalled in order
  Position 501..511 : filler (51)
  seq_length = 512

Why Local Attention fails:
  Max receptive field = local_window * num_layers = 64 * 6 = 384 tokens
  The gap between needle zone (pos 0-49) and recall zone (pos 451-500) is
  401-450 positions -- all strictly > 384 -- so the window PHYSICALLY
  cannot see the needle, regardless of training.

Why Griffin succeeds:
  RNN state carries needle information through 400 filler steps.
  The filler is constant (token 51), so gradients push Griffin to
  preserve needle state rather than overwrite it.

Expected outcome:
  Griffin perplexity  ~1.05-2.00  (recalls needles via recurrence)
  Hawk perplexity     ~2.00-4.00  (recurrence but may struggle with long BPTT)
  Local Attn perplexity ~5.0      (random guess: ln(5)=1.61 nats, ppl=5)
"""

import sys, os, json, time
import torch
import torch.nn as nn
import numpy as np
import psutil
from torch.utils.data import Dataset, DataLoader

# Run from Griffin_pw/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.griffin import GriffinModel
from models.hawk import HawkModel
from models.local_attention import LocalAttentionModel

# ------------------------------------------------------------------ #
#  Dataset                                                             #
# ------------------------------------------------------------------ #
N_NEEDLES      = 1    # 1 needle token to memorise — simple enough to learn in reasonable time
N_NEEDLE_VALS  = 50   # vocab 1..50 for the needle (random baseline ppl = 50)
FILLER_TOKEN   = 51
SEP_TOKEN      = 52
VOCAB_SIZE     = 53   # 0=PAD, 1-50=needle values, 51=FILLER, 52=SEP
SEQ_LEN        = 512
GAP_LEN        = 400  # filler tokens between needle zone and recall zone
# gap=400 > 384 = 6*64 : structurally unreachable by Local Attention (window=64, 6 layers)
# Random perplexity baseline at recall position: 50.0 (uniform over 50 needle values)

class ForcedLongRangeDataset(Dataset):
    """
    Each sample forces the model to recall 5 tokens after a 400-token gap.
    Local Attention (6 layers x window=64 = 384) cannot bridge this gap (402 > 384).
    """
    def __init__(self, n_samples=300):
        self.data = [self._gen() for _ in range(n_samples)]

    def _gen(self):
        # 1 random needle (value 1..50)
        needle = torch.randint(1, N_NEEDLE_VALS + 1, ()).item()

        seq  = [needle]                      # pos 0       (needle zone)
        seq += [FILLER_TOKEN] * GAP_LEN      # pos 1..400  (gap)
        seq += [SEP_TOKEN]                   # pos 401     (separator)
        seq += [needle]                      # pos 402     (recall zone) <-- key position
        pad  = SEQ_LEN - len(seq)
        seq += [FILLER_TOKEN] * pad          # pos 403..511
        seq  = seq[:SEQ_LEN]

        # LM labels: seq shifted left by 1
        labels = seq[1:] + [0]

        return (torch.tensor(seq,    dtype=torch.long),
                torch.tensor(labels, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return {'input_ids': x, 'labels': y}


# ------------------------------------------------------------------ #
#  Training                                                            #
# ------------------------------------------------------------------ #
DEVICE      = 'cpu'
BATCH_SIZE  = 8
N_TRAIN     = 800   # 100 steps/epoch with batch 8
N_VAL       = 100
EPOCHS      = 5     # 500 total steps per model — enough for single-needle recall
GRAD_CLIP   = 1.0

# Sequence layout (1-needle version):
#   input[0]      : needle token (value 1..50)
#   input[1..400] : 400 filler tokens
#   input[401]    : SEP token
#   input[402]    : recall zone (same needle) <-- THIS is what we measure
# Labels (shifted left by 1):
#   label[401] = input[402] = needle recall position
RECALL_LABEL_POS = N_NEEDLES + GAP_LEN  # = 1 + 400 = 401


def recall_loss(logits, labels):
    """Loss computed ONLY at the single recall position — where Local Attention is blind."""
    # logits: (B, SEQ_LEN, VOCAB_SIZE)  labels: (B, SEQ_LEN)
    recall_logits = logits[:, RECALL_LABEL_POS, :]  # (B, V)
    recall_labels = labels[:, RECALL_LABEL_POS]     # (B,)
    return nn.functional.cross_entropy(
        recall_logits,
        recall_labels,
        ignore_index=0
    )


def run_model(name, model, grad_clip=1.0):
    model = model.to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    train_ds = ForcedLongRangeDataset(N_TRAIN)
    val_ds   = ForcedLongRangeDataset(N_VAL)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    total_steps = EPOCHS * len(train_dl)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(total_steps, 1), eta_min=1e-5)

    param_count = sum(p.numel() for p in model.parameters())
    peak_cpu = 0
    grad_norms = []
    step_times = []
    t0 = time.time()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss, steps = 0.0, 0
        for batch in train_dl:
            x = batch['input_ids'].to(DEVICE)
            y = batch['labels'].to(DEVICE)
            t_step = time.time()
            optim.zero_grad()
            out    = model(x)
            logits = out['logits'] if isinstance(out, dict) else out
            loss   = recall_loss(logits, y)  # ONLY recall zone
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            gn = sum(
                p.grad.data.norm(2).item()**2
                for p in model.parameters() if p.grad is not None
            ) ** 0.5
            grad_norms.append(gn)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            scheduler.step()
            epoch_loss += loss.item()
            steps += 1
            step_times.append(time.time() - t_step)
            peak_cpu = max(peak_cpu, psutil.Process().memory_info().rss / 1024 / 1024)

        avg_train = epoch_loss / steps if steps > 0 else float('nan')
        print(f'  Epoch {epoch+1}/{EPOCHS} - Recall-zone Train Loss: {avg_train:.4f}')

    # Validation
    model.eval()
    val_loss, vn = 0.0, 0
    with torch.no_grad():
        for batch in val_dl:
            x = batch['input_ids'].to(DEVICE)
            y = batch['labels'].to(DEVICE)
            out    = model(x)
            logits = out['logits'] if isinstance(out, dict) else out
            loss   = recall_loss(logits, y)  # ONLY recall zone
            if not (torch.isnan(loss) or torch.isinf(loss)):
                val_loss += loss.item()
                vn += 1

    avg_val = val_loss / vn if vn > 0 else float('nan')
    ppl     = float(torch.exp(torch.tensor(avg_val)).item()) if not np.isnan(avg_val) else None
    total   = time.time() - t0

    print(f'  Val Recall-zone Loss: {avg_val:.4f}  Perplexity: {ppl:.4f}')
    print(f'  Baseline random perplexity = {N_NEEDLE_VALS} (1 needle from {N_NEEDLE_VALS} values, uniform guess)')

    return {
        'parameters':              param_count,
        'final_loss':             avg_val,
        'perplexity':             ppl,
        'validation_loss':        avg_val,
        'validation_perplexity':  ppl,
        'training_time_hours':    total / 3600,
        'convergence_steps':      steps,
        'peak_gradient_norm':     float(np.mean(grad_norms)) if grad_norms else 0.0,
        'memory_efficiency_mb_per_param': peak_cpu / param_count if param_count > 0 else 0,
        'latency_sec_per_step':   float(np.mean(step_times)) if step_times else 0.0,
        'throughput_samples_per_sec': N_TRAIN / total,
        'cpu_mem_peak_mb':        peak_cpu,
        'gpu_mem_peak_mb':        0,
        'model_size_mb':          param_count * 4 / 1024 / 1024,
        'flops_per_token':        0,
        'task_notes': (
            'Forced long-range recall: 1 needle at pos 0, '
            '400-token filler gap, recall at pos 402. '
            'Gap = 402 > 384 = 6x64: Local Attention structurally cannot bridge this. '
            'Random baseline perplexity = 50 (1 needle from 50-value vocab).'
        ),
    }


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #
MODELS = {
    'Griffin': (GriffinModel({
        'd_model': 144, 'num_layers': 6, 'num_heads': 8,
        'vocab_size': VOCAB_SIZE, 'max_seq_len': SEQ_LEN,
        'local_window': 64, 'recurrence_chunk_size': 64,
    }), 1.0),
    'Hawk': (HawkModel({
        'd_model': 144, 'num_layers': 6, 'num_heads': 8,
        'vocab_size': VOCAB_SIZE, 'max_seq_len': SEQ_LEN,
    }), 1.0),   # RG-LRU is provably stable; no aggressive clipping needed
    'Local Attention': (LocalAttentionModel({
        'd_model': 144, 'num_layers': 6, 'num_heads': 8,
        'vocab_size': VOCAB_SIZE, 'max_seq_len': SEQ_LEN,
        'local_window': 64,
    }), 1.0),
}

BENCH_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'complete_benchmark.json')
with open(BENCH_FILE) as f:
    bench = json.load(f)

bench['aan_long_range'] = {}

for model_name, (model, clip) in MODELS.items():
    print(f'\n--- Testing {model_name} ---')
    try:
        result = run_model(model_name, model, grad_clip=clip)
        bench['aan_long_range'][model_name] = result
        with open(BENCH_FILE, 'w') as f:
            json.dump(bench, f, indent=2)
        print(f'[SAVED] {model_name}')
    except Exception as e:
        import traceback
        traceback.print_exc()
        bench['aan_long_range'][model_name] = {'error': str(e)}
        with open(BENCH_FILE, 'w') as f:
            json.dump(bench, f, indent=2)
        print(f'[FAIL] {model_name}: {e}')

print('\n=== ALL DONE ===')
print('aan_long_range results:')
for m, r in bench['aan_long_range'].items():
    ppl = r.get('perplexity', 'N/A')
    print(f'  {m}: perplexity = {ppl}')
