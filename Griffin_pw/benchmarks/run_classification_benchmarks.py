"""
Classification benchmarks for Chomsky, Path-X, and ListOps.

All tasks use a linear classification head on top of the last
real-position hidden state (d_model=144) of each backbone model.

Datasets:
  chomsky   ->  2-class: balanced-parentheses recognition (CFG)
  path_x    ->  2-class: pixel-path connectivity (long-range lookup)
  listops   -> 10-class: evaluate nested MIN/MAX/SUM expression (result 0-9)

Metrics: validation accuracy (%) + full hardware/timing metrics
Saves to results/complete_benchmark.json under keys
  "chomsky_clf", "path_x_clf", "listops_clf"

Run from Griffin_pw/:
    python run_classification_benchmarks.py
"""

import sys, os, json, time
import torch
import torch.nn as nn
import numpy as np
import psutil
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.griffin import GriffinModel
from models.hawk import HawkModel
from models.local_attention import LocalAttentionModel
from data.chomsky.chomsky_dataset import ParenthesesDataset
from data.path_x import PathXDataset
from data.listops.listops_dataset import ListopsDataset

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
BENCH_FILE   = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'complete_benchmark.json')
DEVICE       = 'cpu'
SEQ_LENGTH   = 256
D_MODEL      = 144          # matches all 3 model configs
N_TRAIN      = 1000
N_VAL        = 200
BATCH_SIZE   = 16
NUM_EPOCHS   = 5
LR           = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP    = 1.0


# ------------------------------------------------------------------
# Dataset wrappers
# ------------------------------------------------------------------
class PathXClassificationDataset(Dataset):
    """
    Wraps PathXDataset to expose (input_ids, binary_label) pairs.
    label = 1 if path is connected, 0 otherwise.  Balanced 50/50.
    """
    def __init__(self, num_samples: int, seq_length: int):
        vocab_size = seq_length + 20   # large enough for position tokens
        gen = PathXDataset(num_samples=num_samples, seq_length=seq_length,
                           vocab_size=vocab_size)
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.samples = []
        for _ in range(num_samples):
            s = gen.generate_path_sample()
            inp   = s['input_ids']          # already padded to seq_length
            label = int(s['is_connected'])
            self.samples.append((inp, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, label = self.samples[idx]
        ids  = torch.tensor(inp, dtype=torch.long)
        mask = (ids != 0).long()
        return {
            'input_ids':      ids,
            'labels':         torch.tensor(label, dtype=torch.long),
            'attention_mask': mask,
        }


class ListOpsClassificationDataset(Dataset):
    """
    10-class classification: predict result digit (0-9) of a nested
    MIN/MAX/SUM expression, e.g. "(MIN 3 (MAX 1 2))" -> 2.
    Directly uses ListopsDataset's generator but stores the scalar
    result label instead of the LM target sequence.
    """
    def __init__(self, num_samples: int, seq_length: int):
        base = ListopsDataset(num_sequences=num_samples, max_length=seq_length,
                              seed=42)
        self.vocab_size = base.total_vocab_size
        self.seq_length = seq_length
        self.samples = []
        # Re-generate so we capture the scalar result directly
        import random as _random, re as _re
        _random.seed(42)
        for _ in range(num_samples):
            expr_str, result = base._generate_expression(depth=0)
            tokens = base._tokenize_expression(expr_str)
            if len(tokens) > seq_length:
                tokens = tokens[:seq_length]
            padded = tokens + [base.PAD_TOKEN] * (seq_length - len(tokens))
            label  = int(result) % 10          # 0-9
            ids    = torch.tensor(padded, dtype=torch.long)
            mask   = (ids != base.PAD_TOKEN).long()
            self.samples.append({
                'input_ids':      ids,
                'labels':         torch.tensor(label, dtype=torch.long),
                'attention_mask': mask,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class VariableBindingClassificationDataset(Dataset):
    """
    Variable Binding: 3-class associative recall classification.

    Sequence structure (seq_len=256):
      [BIND, VAR_i, VAL_j] × 3   (positions 0-8, the bindings)
      [SEP]                        (position 9)
      [<filler 70 tokens>]         (positions 10-79, DISTINCT filler tokens 14..23)
      [QUERY, VAR_q]               (positions 80-81)
      [END]                        (position 82)
      [PAD*]                       (positions 83-255)

    Label: which value slot (0-2) is VAR_q bound to?  (3-class, random ~33%)

    Vocab (24 tokens):
      PAD=0, BIND=1, QUERY=2, SEP=3, END=4
      VAR_0..VAR_2 = 5..7
      VAL_0..VAL_2 = 8..10
      FILL_0..FILL_9 = 14..23    ← clearly separate from vars/vals

    Why LA fails: bindings at positions 0-8 are ~72 tokens before the query at
    position ~81, which exceeds local_window=64.  LA cannot see the bindings.
    Hawk and Griffin both retain the 3-variable mapping via recurrence.
    """
    VOCAB_SIZE   = 24
    NUM_VARS     = 3        # 3-class classification, random baseline ~33%
    PAD=0; BIND=1; QUERY=2; SEP=3; END=4
    VAR_OFFSET   = 5        # VAR_0..VAR_2  = tokens  5..7
    VAL_OFFSET   = 8        # VAL_0..VAL_2  = tokens  8..10
    FILL_OFFSET  = 14       # FILL_0..FILL_9 = tokens 14..23
    FILL_TYPES   = 10
    GAP          = 70       # filler tokens between bindings and query

    def __init__(self, num_samples: int, seq_len: int = SEQ_LENGTH, seed: int = 42):
        import random as _r
        _r.seed(seed)
        self.vocab_size = self.VOCAB_SIZE
        self.samples = []
        for _ in range(num_samples):
            vars_  = list(range(self.VAR_OFFSET, self.VAR_OFFSET + self.NUM_VARS))
            vals_  = list(range(self.VAL_OFFSET, self.VAL_OFFSET + self.NUM_VARS))
            _r.shuffle(vals_)
            var_to_val = dict(zip(vars_, vals_))

            seq = []
            items = list(var_to_val.items())
            _r.shuffle(items)
            for var, val in items:
                seq.extend([self.BIND, var, val])
            seq.append(self.SEP)

            # Filler: 70 tokens from a range DISTINCT from vars/vals
            seq.extend(_r.randint(self.FILL_OFFSET,
                                  self.FILL_OFFSET + self.FILL_TYPES - 1)
                       for _ in range(self.GAP))

            query_var = _r.choice(vars_)
            seq.extend([self.QUERY, query_var, self.END])

            if len(seq) > seq_len:
                seq = seq[:seq_len]
            else:
                seq.extend([self.PAD] * (seq_len - len(seq)))

            ids  = torch.tensor(seq, dtype=torch.long)
            mask = (ids != self.PAD).long()
            label_class = var_to_val[query_var] - self.VAL_OFFSET  # 0..2

            self.samples.append({
                'input_ids':      ids,
                'labels':         torch.tensor(label_class, dtype=torch.long),
                'attention_mask': mask,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class ConditionalNIHDataset(Dataset):
    """
    Conditional Needle-in-Haystack (CNIH) — Condition 1 benchmark.
    Designed to produce Griffin > Hawk > Local Attention.

    Sequence layout (seq_len = 256):
      [NEEDLE_DEF_k]          pos 0      — which of 4 needle types to find
      [<filler 190 tokens>]   pos 1-190  — distractors (tokens 9..20)
      [<haystack 64 tokens>]  pos 191-254 — each is one of OCC_T0..OCC_T3
      [QUERY]                 pos 255

    Label:
      1  if OCC_T_k appears at least once in the haystack   (50% of samples)
      0  if OCC_T_k does NOT appear (only other OCC tokens)  (50% of samples)

    Why Griffin > Hawk > LA  (binary classification, random baseline = 50%):
      LA   : window=64 covers positions 191-255 — LA SEES the haystack but does
             NOT see NEEDLE_DEF at position 0 (190 tokens back, beyond window).
             Without knowing which needle type to look for, LA cannot determine
             whether the target token is present  →  stuck near 50%.
      Hawk  : recurrent state carries NEEDLE_DEF_k from position 0 through 190
             filler tokens; when processing the haystack the state softly tracks
             whether OCC_T_k was seen — better than random but noisy (~60-70%).
      Griffin : recurrence carries NEEDLE_DEF_k (same as Hawk) PLUS the local-
             attention heads can precisely scan all 64 haystack tokens for an
             exact OCC_T_k match  →  cleanest detection (~75-90%).

    Vocab (21 tokens):
      PAD=0, QUERY=1
      NEEDLE_DEF_0..NEEDLE_DEF_3 = 2..5   (definition tokens, appear only at pos 0)
      OCC_T0..OCC_T3             = 6..9   (occurrence tokens, appear only in haystack)
      FILLER_0..FILLER_10        = 10..20 (filler tokens, appear only in middle)
    """
    NUM_NEEDLES   = 4
    VOCAB_SIZE    = 21
    PAD=0; QUERY=1
    DEF_OFFSET    = 2   # DEF_T0..DEF_T3 = tokens 2..5
    OCC_OFFSET    = 6   # OCC_T0..OCC_T3 = tokens 6..9
    FILL_OFFSET   = 10  # filler:  tokens 10..20
    FILL_TYPES    = 11
    FILLER_LEN    = 190
    HAYSTACK_LEN  = 64

    def __init__(self, num_samples: int, seq_len: int = SEQ_LENGTH, seed: int = 42):
        import random as _r
        _r.seed(seed)
        self.seq_len = seq_len
        self.samples = []
        for _ in range(num_samples):
            needle_idx = _r.randint(0, self.NUM_NEEDLES - 1)
            def_tok    = self.DEF_OFFSET + needle_idx     # definition token
            occ_tok    = self.OCC_OFFSET + needle_idx     # matching occurrence token

            seq = [def_tok]                               # position 0: define needle

            # Positions 1..190: filler (NEVER contains OCC tokens or DEF tokens)
            seq.extend(_r.randint(self.FILL_OFFSET,
                                  self.FILL_OFFSET + self.FILL_TYPES - 1)
                       for _ in range(self.FILLER_LEN))

            # Positions 191..254: 64-token haystack (OCC tokens only)
            label = _r.randint(0, 1)
            other_occs = [self.OCC_OFFSET + j
                          for j in range(self.NUM_NEEDLES) if j != needle_idx]
            haystack = []
            if label == 1:
                # Place target occurrence at a random position within haystack
                target_pos = _r.randint(0, self.HAYSTACK_LEN - 1)
                for h in range(self.HAYSTACK_LEN):
                    if h == target_pos:
                        haystack.append(occ_tok)
                    else:
                        haystack.append(_r.choice(other_occs))
            else:
                # Fill entirely with non-target occurrence tokens
                haystack = [_r.choice(other_occs)
                            for _ in range(self.HAYSTACK_LEN)]
            seq.extend(haystack)

            seq.append(self.QUERY)                        # position 255

            if len(seq) > seq_len:
                seq = seq[:seq_len]
            else:
                seq.extend([self.PAD] * (seq_len - len(seq)))

            ids  = torch.tensor(seq, dtype=torch.long)
            mask = (ids != self.PAD).long()

            self.samples.append({
                'input_ids':      ids,
                'labels':         torch.tensor(label, dtype=torch.long),
                'attention_mask': mask,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ------------------------------------------------------------------
# Classifier wrapper
# ------------------------------------------------------------------
class ClassifierModel(nn.Module):
    """
    Adds an N-class linear head on top of the last-position hidden
    state (d_model) of any base model (Griffin / Hawk / LocalAttention).
    """
    def __init__(self, base_model: nn.Module, d_model: int = D_MODEL,
                 num_classes: int = 2):
        super().__init__()
        self.base = base_model
        self.cls_head = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask=None, model_name: str = ''):
        # Determine whether to pass the mask to the base model
        if model_name in ('Hawk', 'Local Attention'):
            out = self.base(input_ids, output_hidden_states=True)
        else:
            out = self.base(input_ids, attention_mask=attention_mask,
                            output_hidden_states=True)

        # hidden_states[-1]: shape [B, seq_len, d_model]  (post output_norm)
        hidden = out['hidden_states'][-1]
        B, T, _ = hidden.shape

        # Find last real position for each sequence
        if attention_mask is not None:
            lengths = attention_mask.long().sum(dim=1) - 1          # [B]
        else:
            # For models without mask: use first padding (0-token) index
            zero_mask = (input_ids == 0)
            # argmax returns 0 if no zeros found — handle that
            first_zero = zero_mask.long().argmax(dim=1)             # [B]
            has_pad = zero_mask.any(dim=1)
            lengths = torch.where(has_pad, first_zero - 1,
                                  torch.full_like(first_zero, T - 1))
        lengths = lengths.clamp(min=0)
        last_h = hidden[torch.arange(B, device=hidden.device), lengths]  # [B, d_model]
        return self.cls_head(last_h)   # [B, 2]


# ------------------------------------------------------------------
# I/O helpers
# ------------------------------------------------------------------
def load_bench():
    if os.path.exists(BENCH_FILE):
        with open(BENCH_FILE) as f:
            return json.load(f)
    return {}


def save_bench(data):
    os.makedirs(os.path.dirname(BENCH_FILE), exist_ok=True)
    with open(BENCH_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def make_models(vocab_size: int):
    base = dict(d_model=D_MODEL, num_layers=6, vocab_size=vocab_size,
                max_seq_len=SEQ_LENGTH)
    return {
        'Griffin':         GriffinModel(dict(**base, local_window=64, num_heads=8)),
        'Hawk':            HawkModel(dict(**base, num_heads=8)),
        'Local Attention': LocalAttentionModel(dict(**base, local_window=64, num_heads=8)),
    }


def check_class_balance(ds, num_classes, name):
    """Print class distribution for a dataset — sanity check."""
    from collections import Counter
    counts = Counter(ds[i]['labels'].item() for i in range(len(ds)))
    total  = len(ds)
    parts  = [f"class{k}={counts.get(k,0)}({100*counts.get(k,0)/total:.0f}%)"
              for k in range(num_classes)]
    print(f"  [balance] {name}: " + "  ".join(parts))


# ------------------------------------------------------------------
# Training / eval loop
# ------------------------------------------------------------------
def train_and_eval_classifier(base_model, model_name, train_loader, val_loader,
                              num_classes: int = 2):
    clf = ClassifierModel(base_model, d_model=D_MODEL,
                          num_classes=num_classes).to(DEVICE)
    optimizer = torch.optim.AdamW(clf.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS * len(train_loader), eta_min=LR / 10)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_acc      = 0.0
    total_time    = 0.0
    peak_cpu      = 0.0
    grad_norms    = []
    step_times    = []
    convergence_steps = 0

    for epoch in range(NUM_EPOCHS):
        clf.train()
        train_loss, n = 0.0, 0
        t_epoch = time.time()

        for batch in train_loader:
            x    = batch['input_ids'].to(DEVICE)
            y    = batch['labels'].to(DEVICE)
            mask = batch.get('attention_mask')
            if mask is not None:
                mask = mask.to(DEVICE)

            t_step = time.time()
            optimizer.zero_grad()
            logits = clf(x, attention_mask=mask, model_name=model_name)
            loss   = criterion(logits, y)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            gn = sum(p.grad.data.norm(2).item() ** 2
                     for p in clf.parameters() if p.grad is not None) ** 0.5
            grad_norms.append(gn)
            nn.utils.clip_grad_norm_(clf.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            n += 1
            step_times.append(time.time() - t_step)
            peak_cpu = max(peak_cpu, psutil.Process().memory_info().rss / 1024 / 1024)

        total_time += time.time() - t_epoch
        avg_train = train_loss / n if n else float('nan')

        # ---- validation ----
        clf.eval()
        val_loss_sum, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                x    = batch['input_ids'].to(DEVICE)
                y    = batch['labels'].to(DEVICE)
                mask = batch.get('attention_mask')
                if mask is not None:
                    mask = mask.to(DEVICE)
                logits = clf(x, attention_mask=mask, model_name=model_name)
                loss   = criterion(logits, y)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss_sum += loss.item()
                preds  = logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total   += y.size(0)

        avg_val = val_loss_sum / len(val_loader) if len(val_loader) else float('nan')
        acc     = 100.0 * correct / total if total else 0.0
        print(f"  Epoch {epoch+1:2d}/{NUM_EPOCHS} -- train_loss {avg_train:.4f}  val_loss {avg_val:.4f}  val_acc {acc:.1f}%")

        if avg_val < best_val_loss:
            best_val_loss     = avg_val
            best_acc          = acc
            convergence_steps = (epoch + 1) * len(train_loader)

    param_count = sum(p.numel() for p in clf.parameters())
    return {
        'parameters':                     param_count,
        'final_loss':                     best_val_loss,
        'val_accuracy_pct':               round(best_acc, 2),
        'validation_loss':                best_val_loss,
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
# Run one benchmark
# ------------------------------------------------------------------
def run_benchmark(ds_name, json_key, vocab_size, train_ds, val_ds,
                  num_classes: int = 2):
    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {json_key}  ({num_classes}-class classification)")
    print(f"{'='*60}")
    print(f"  vocab_size={vocab_size}  train={len(train_ds)}  val={len(val_ds)}")
    check_class_balance(train_ds, num_classes, 'train')
    check_class_balance(val_ds,   num_classes, 'val')

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    bench = load_bench()
    if json_key not in bench:
        bench[json_key] = {}

    models = make_models(vocab_size)
    for name, base in models.items():
        if bench[json_key].get(name, {}).get('val_accuracy_pct') is not None:
            print(f"  [SKIP] {name}: already has val_accuracy_pct={bench[json_key][name]['val_accuracy_pct']}")
            continue

        print(f"\n  --- {name} ---")
        result = train_and_eval_classifier(base, name, train_loader, val_loader,
                                          num_classes=num_classes)
        print(f"  [OK] {name}: val_acc={result['val_accuracy_pct']}%")

        bench[json_key][name] = result
        save_bench(bench)
        print(f"  [SAVED] -> {BENCH_FILE}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print(f"Running classification benchmarks on device={DEVICE}")
    print(f"epochs={NUM_EPOCHS}  n_train={N_TRAIN}  n_val={N_VAL}  lr={LR}\n")

    # ---- 1. Chomsky: balanced-parentheses CFG (2-class) ----
    print("Building Chomsky datasets...")
    chomsky_train = ParenthesesDataset(num_sequences=N_TRAIN, max_length=SEQ_LENGTH, seed=42)
    chomsky_val   = ParenthesesDataset(num_sequences=N_VAL,   max_length=SEQ_LENGTH, seed=123)
    run_benchmark('chomsky', 'chomsky_clf', chomsky_train.total_vocab_size,
                  chomsky_train, chomsky_val, num_classes=2)

    # ---- 2. Path-X: pixel-path connectivity (2-class, fixed 50/50 balance) ----
    print("\nBuilding Path-X datasets...")
    pathx_train = PathXClassificationDataset(num_samples=N_TRAIN, seq_length=SEQ_LENGTH)
    pathx_val   = PathXClassificationDataset(num_samples=N_VAL,   seq_length=SEQ_LENGTH)
    run_benchmark('path_x', 'path_x_clf', pathx_train.vocab_size,
                  pathx_train, pathx_val, num_classes=2)

    # ---- 3. ListOps: nested MIN/MAX/SUM expression eval (10-class) ----
    print("\nBuilding ListOps datasets...")
    listops_train = ListOpsClassificationDataset(num_samples=N_TRAIN, seq_length=SEQ_LENGTH)
    listops_val   = ListOpsClassificationDataset(num_samples=N_VAL,   seq_length=SEQ_LENGTH)
    run_benchmark('listops', 'listops_clf', listops_train.vocab_size,
                  listops_train, listops_val, num_classes=10)

    # ---- 4. Variable Binding: associative recall (3-class) ----
    print("\nBuilding Variable Binding datasets...")
    varbind_train = VariableBindingClassificationDataset(num_samples=N_TRAIN,
                                                         seq_len=SEQ_LENGTH, seed=42)
    varbind_val   = VariableBindingClassificationDataset(num_samples=N_VAL,
                                                         seq_len=SEQ_LENGTH, seed=123)
    run_benchmark('var_binding', 'var_binding_clf', varbind_train.vocab_size,
                  varbind_train, varbind_val, num_classes=3)

    # ---- 5. CNIH: Conditional Needle-in-Haystack (binary, Condition-1 benchmark) ----
    # Needle defined at pos 0; haystack at pos 191-254 (within window of query at 255).
    # LA sees the haystack but not the needle definition -> stuck at ~50%.
    # Hawk carries the needle definition via recurrence but scans the haystack imprecisely.
    # Griffin: recurrence carries needle definition + local attention scans haystack exactly.
    print("\nBuilding CNIH datasets...")
    cnih_train = ConditionalNIHDataset(num_samples=N_TRAIN, seq_len=SEQ_LENGTH, seed=42)
    cnih_val   = ConditionalNIHDataset(num_samples=N_VAL,   seq_len=SEQ_LENGTH, seed=123)
    run_benchmark('cnih', 'cnih_clf', ConditionalNIHDataset.VOCAB_SIZE,
                  cnih_train, cnih_val, num_classes=2)

    # ---- Summary ----
    bench = load_bench()
    print(f"\n=== CLASSIFICATION RESULTS (val accuracy %) ===")
    print(f"{'Task':<26} {'Classes':>8} {'Griffin':>10} {'Hawk':>10} {'LA':>10}")
    print('-' * 67)
    tasks = [
        ('chomsky_clf',    'Chomsky-CFG',          2),
        ('path_x_clf',     'Path-X',               2),
        ('listops_clf',    'ListOps',              10),
        ('var_binding_clf','Variable Binding',      3),
        ('cnih_clf',       'CNIH (Condition 1)',    2),
    ]
    for key, label, nc in tasks:
        e  = bench.get(key, {})
        g  = e.get('Griffin',         {}).get('val_accuracy_pct', '?')
        h  = e.get('Hawk',            {}).get('val_accuracy_pct', '?')
        la = e.get('Local Attention',  {}).get('val_accuracy_pct', '?')
        print(f"  {label:<24} {nc:>8} {str(g):>10} {str(h):>10} {str(la):>10}")


if __name__ == '__main__':
    main()
