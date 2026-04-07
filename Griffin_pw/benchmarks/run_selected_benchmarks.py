"""Run ONLY the datasets used in notebooks/model_analysis.ipynb.

This is a self-contained runner so you don't have to touch the existing benchmark
scripts.

Default output:
    - results/complete_benchmark.json

You can also write separate JSON files via --out (e.g. results/benchmark_run1.json).

Optional:
    python benchmarks/run_selected_benchmarks.py --device cuda

Run a specific dataset (quick test):
    python benchmarks/run_selected_benchmarks.py --list-tasks
    python benchmarks/run_selected_benchmarks.py --tasks MQAR

Notes:
- Perplexity is computed token-weighted (masked) as exp(NLL/token).
- Memory is reported as process RSS (CPU) and CUDA peak allocated (GPU), separately.
- Results are written after EVERY model, so the run is crash-safe.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Allow imports when run from Griffin_pw/benchmarks/
REPO_ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(REPO_ROOT))

from data.unified_datasets import UnifiedDataset
from data.copy.copy_dataset import CopyDataset
from data.chomsky.chomsky_dataset import ParenthesesDataset
from data.path_x import PathXDataset
from data.listops.listops_dataset import ListopsDataset
from models.griffin import GriffinModel
from models.hawk import HawkModel
from models.local_attention import LocalAttentionModel


MODELS = ("Griffin", "Hawk", "Local Attention")


# -----------------------------------------------------------------------------
# Datasets (exactly the notebook set)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class LmTask:
    key: str
    group: str
    make_loaders: Any  # callable returning (train_loader, val_loader, vocab_size, n_train_samples)


@dataclass(frozen=True)
class ClfTask:
    key: str
    group: str
    num_classes: int
    make_loaders: Any


GROUP_1 = "group_1_recurrence_dominant"
GROUP_2 = "group_2_local_precision_dominant"


# -----------------------------------------------------------------------------
# Classification dataset wrappers (copied from run_classification_benchmarks.py)
# -----------------------------------------------------------------------------
class PathXClassificationDataset(Dataset):
    """Binary label = 1 if path connected, else 0 (balanced-ish)."""

    def __init__(self, num_samples: int, seq_length: int, seed: int, difficulty: int = 1):
        # PathX embeds raw position indices as token ids, so vocab must cover 0..seq_length-1
        vocab_size = seq_length + 20
        gen = PathXDataset(num_samples=num_samples, seq_length=seq_length, vocab_size=vocab_size, difficulty=difficulty, seed=seed)
        self.vocab_size = vocab_size
        self.seq_length = seq_length

        # PathXDataset has its own RNG; set torch seed for deterministic padding/masks
        torch.manual_seed(seed)

        self.samples = []
        for _ in range(num_samples):
            s = gen.generate_path_sample()
            inp = s["input_ids"]
            label = int(s["is_connected"])
            ids = torch.tensor(inp, dtype=torch.long)
            mask = (ids != 0).long()
            self.samples.append({
                "input_ids": ids,
                "labels": torch.tensor(label, dtype=torch.long),
                "attention_mask": mask,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


class ListOpsClassificationDataset(Dataset):
    """10-class label = expression result digit (0..9)."""

    def __init__(self, num_samples: int, seq_length: int, seed: int, max_depth: int = 3):
        base = ListopsDataset(num_sequences=num_samples, max_length=seq_length, max_depth=max_depth, seed=seed)
        self.vocab_size = base.total_vocab_size
        self.seq_length = seq_length

        # Re-generate so we capture scalar result directly
        import random as _random

        _random.seed(seed)

        self.samples = []
        for _ in range(num_samples):
            expr_str, result = base._generate_expression(depth=0)
            tokens = base._tokenize_expression(expr_str)
            if len(tokens) > seq_length:
                tokens = tokens[:seq_length]
            padded = tokens + [base.PAD_TOKEN] * (seq_length - len(tokens))
            label = int(result) % 10

            ids = torch.tensor(padded, dtype=torch.long)
            mask = (ids != base.PAD_TOKEN).long()
            self.samples.append({
                "input_ids": ids,
                "labels": torch.tensor(label, dtype=torch.long),
                "attention_mask": mask,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


class ClassifierModel(nn.Module):
    """Linear head on last-position hidden state of base model."""

    def __init__(self, base_model: nn.Module, d_model: int, num_classes: int):
        super().__init__()
        self.base = base_model
        self.cls_head = nn.Linear(d_model, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], model_name: str) -> torch.Tensor:
        if model_name in ("Hawk", "Local Attention"):
            out = self.base(input_ids, output_hidden_states=True)
        else:
            out = self.base(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        hidden = out["hidden_states"][-1]  # [B, T, d_model]
        bsz, seq_len, _ = hidden.shape

        if attention_mask is not None:
            lengths = attention_mask.long().sum(dim=1) - 1
        else:
            zero_mask = input_ids == 0
            first_zero = zero_mask.long().argmax(dim=1)
            has_pad = zero_mask.any(dim=1)
            lengths = torch.where(has_pad, first_zero - 1, torch.full_like(first_zero, seq_len - 1))
        lengths = lengths.clamp(min=0)

        last_h = hidden[torch.arange(bsz, device=hidden.device), lengths]
        return self.cls_head(last_h)


# -----------------------------------------------------------------------------
# Model + metrics helpers
# -----------------------------------------------------------------------------
def make_models(vocab_size: int, seq_len: int, d_model: int, num_layers: int, num_heads: int, local_window: int) -> Dict[str, nn.Module]:
    base = dict(d_model=d_model, num_layers=num_layers, vocab_size=vocab_size, max_seq_len=seq_len)
    return {
        "Griffin": GriffinModel(dict(**base, local_window=local_window, num_heads=num_heads)),
        "Hawk": HawkModel(dict(**base, num_heads=num_heads)),
        "Local Attention": LocalAttentionModel(dict(**base, local_window=local_window, num_heads=num_heads)),
    }


def _forward_lm(model: nn.Module, model_name: str, x: torch.Tensor, attention_mask: Optional[torch.Tensor]):
    if model_name in ("Hawk", "Local Attention"):
        return model(x)
    return model(x, attention_mask=attention_mask)


def _choose_lm_loss_mask(attention_mask: Optional[torch.Tensor], labels: torch.Tensor) -> Optional[torch.Tensor]:
    """Choose a token-loss mask that matches where the targets actually are.

    Most LM-style datasets align targets with the non-padding input region, so
    `attention_mask` works.

    But some synthetic datasets (notably Copy) have targets in a different span
    than the non-padding input tokens. In those cases, using `attention_mask`
    silently masks out the hard region and makes perplexity meaningless.

    Heuristic:
    - Prefer attention_mask when it overlaps well with non-zero labels.
    - Fall back to label-based mask (labels != 0) when overlap is poor.
    """
    if attention_mask is None:
        return None

    attn = attention_mask.bool()
    # Some datasets use torch-style ignore_index=-100 (e.g., MQAR answer-only).
    # Others use PAD=0 in labels (e.g., Copy) and rely on masking.
    uses_ignore_index = int(labels.min().item()) < 0
    ignore_idx = -100 if uses_ignore_index else 0
    label_mask = (labels != ignore_idx)

    # Critical: for ignore_index datasets, loss must be computed ONLY on
    # supervised positions. Using attention_mask (often all-ones) would dilute
    # the loss by averaging many zero-loss ignored tokens.
    if uses_ignore_index:
        return label_mask

    attn_sum = int(attn.sum().item())
    label_sum = int(label_mask.sum().item())
    if attn_sum == 0:
        return attn
    if label_sum == 0:
        # Nothing to learn from labels; keep the default.
        return attn

    overlap = int((attn & label_mask).sum().item())
    # Key question: are the *labeled* positions inside the attention mask?
    label_coverage = overlap / label_sum if label_sum else 1.0

    # If many labeled positions are outside attention_mask (Copy-style), fall
    # back to label-based masking for the loss.
    if label_coverage < 0.5:
        return label_mask

    return attn


def _choose_lm_forward_mask(attention_mask: Optional[torch.Tensor], labels: torch.Tensor) -> Optional[torch.Tensor]:
    """Choose the attention mask passed into the model forward.

    For most datasets, the input attention mask is correct.

    For Copy-style datasets, the *targets* live in a region where input_ids are
    PAD=0, so attention_mask is typically 0 there. Passing that mask into
    Griffin would effectively prevent the model from producing meaningful hidden
    states/logits for the target region (while Hawk/LocalAttention ignore masks).

    Heuristic:
    - If attention_mask overlaps poorly with non-zero labels, disable forward
      masking by returning None.
    """
    if attention_mask is None:
        return None

    attn = attention_mask.bool()
    ignore_idx = -100 if int(labels.min().item()) < 0 else 0
    label_mask = (labels != ignore_idx)

    attn_sum = int(attn.sum().item())
    label_sum = int(label_mask.sum().item())
    if attn_sum == 0:
        return None
    if label_sum == 0:
        return attn

    overlap = int((attn & label_mask).sum().item())
    label_coverage = overlap / label_sum if label_sum else 1.0

    # Only disable the forward attention mask when the labeled positions are
    # mostly outside the input mask (Copy-style targets on PAD region).
    if label_coverage < 0.5:
        return None

    return attn


def _masked_xent_loss(logits: torch.Tensor, labels: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, float]:
    vocab = logits.size(-1)
    ignore_idx = -100 if int(labels.min().item()) < 0 else None
    if ignore_idx is None:
        per_tok = F.cross_entropy(logits.view(-1, vocab), labels.view(-1), reduction="none")
    else:
        per_tok = F.cross_entropy(logits.view(-1, vocab), labels.view(-1), reduction="none", ignore_index=ignore_idx)

    # Safety: if the dataset uses ignore_index labels (e.g., MQAR answer-only)
    # and the caller didn't provide an explicit mask, we must NOT average over
    # ignored tokens (they contribute 0 and would dilute the loss).
    if ignore_idx is not None and mask is None:
        mask = (labels != ignore_idx)

    if mask is None:
        return per_tok.mean(), float(labels.numel())

    flat = mask.view(-1).float()
    per_tok = per_tok * flat
    denom = float(flat.sum().item())
    denom = denom if denom > 0 else 1.0
    return per_tok.sum() / denom, denom


@dataclass
class MemStats:
    cpu_start_mb: float
    cpu_peak_mb: float
    gpu_start_mb: float
    gpu_peak_mb: float

    @property
    def cpu_delta_mb(self) -> float:
        return self.cpu_peak_mb - self.cpu_start_mb

    @property
    def gpu_delta_mb(self) -> float:
        return self.gpu_peak_mb - self.gpu_start_mb


def _mem_begin(device: str) -> Tuple[psutil.Process, MemStats]:
    proc = psutil.Process()
    cpu_start = proc.memory_info().rss / 1024 / 1024

    gpu_start = 0.0
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        gpu_start = torch.cuda.memory_allocated() / 1024 / 1024

    return proc, MemStats(cpu_start_mb=cpu_start, cpu_peak_mb=cpu_start, gpu_start_mb=gpu_start, gpu_peak_mb=gpu_start)


def _mem_begin_phase(proc: psutil.Process, device: str) -> MemStats:
    """Start a new measurement phase (e.g., validation/inference) using the
    current process/cuda allocations as the baseline.

    Unlike `_mem_begin`, this reuses the existing `psutil.Process`.
    """
    cpu_start = proc.memory_info().rss / 1024 / 1024
    gpu_start = 0.0
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        gpu_start = torch.cuda.memory_allocated() / 1024 / 1024
    return MemStats(cpu_start_mb=cpu_start, cpu_peak_mb=cpu_start, gpu_start_mb=gpu_start, gpu_peak_mb=gpu_start)


def _mem_update(proc: psutil.Process, mem: MemStats, device: str) -> MemStats:
    cpu_peak = max(mem.cpu_peak_mb, proc.memory_info().rss / 1024 / 1024)
    gpu_peak = mem.gpu_peak_mb
    if device.startswith("cuda"):
        gpu_peak = max(gpu_peak, torch.cuda.max_memory_allocated() / 1024 / 1024)
    return MemStats(mem.cpu_start_mb, cpu_peak, mem.gpu_start_mb, gpu_peak)

def _rss_mb(proc: psutil.Process) -> float:
    """Resident Set Size (RSS) in MB.

    This matches the common definition used in profiling code (e.g. StreamingLLM).
    """
    return float(proc.memory_info().rss) / 1024.0 / 1024.0


def _summarize_rss_mb(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"mean": float("nan"), "median": float("nan")}
    arr = np.asarray(samples, dtype=np.float64)
    return {"mean": float(arr.mean()), "median": float(np.median(arr))}


def train_eval_lm(
    *,
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_train_samples: int,
    device: str,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    num_epochs: int,
) -> Dict[str, Any]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader), eta_min=lr / 10)

    proc, mem_train = _mem_begin(device)
    mem_val = _mem_begin_phase(proc, device)

    rss_samples_train: list[float] = []
    rss_samples_val: list[float] = []

    best_val_loss = float("inf")
    last_val_loss = float("nan")
    last_val_token_acc = float("nan")
    last_val_seq_acc = float("nan")
    best_val_token_acc = float("nan")
    best_val_seq_acc = float("nan")
    total_time = 0.0
    grad_norms = []
    step_times = []
    convergence_steps = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_nll = 0.0
        epoch_tokens = 0.0
        n_steps = 0
        t_epoch = time.time()

        for batch in train_loader:
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device).bool()

            # Use the *input* attention_mask for the model, but choose a
            # potentially different mask for the *loss* (e.g., Copy targets).
            forward_mask = _choose_lm_forward_mask(attention_mask, y)
            loss_mask = _choose_lm_loss_mask(attention_mask, y)
            if loss_mask is not None:
                loss_mask = loss_mask.to(device).bool()

            t_step = time.time()
            optimizer.zero_grad(set_to_none=True)

            out = _forward_lm(model, model_name, x, forward_mask)
            logits = out["logits"] if isinstance(out, dict) else out
            loss, n_tokens = _masked_xent_loss(logits, y, loss_mask)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            gn = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    gn += p.grad.data.norm(2).item() ** 2
            grad_norms.append(gn ** 0.5)

            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_nll += float(loss.item()) * n_tokens
            epoch_tokens += n_tokens
            n_steps += 1
            step_times.append(time.time() - t_step)
            mem_train = _mem_update(proc, mem_train, device)
            rss_samples_train.append(_rss_mb(proc))

        total_time += time.time() - t_epoch
        avg_train = epoch_nll / epoch_tokens if epoch_tokens else float("nan")

        # Validation (token-weighted)
        model.eval()
        val_nll = 0.0
        val_tokens = 0.0
        val_tok_correct = 0.0
        val_tok_total = 0.0
        val_seq_correct = 0.0
        val_seq_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["input_ids"].to(device)
                y = batch["labels"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device).bool()

                forward_mask = _choose_lm_forward_mask(attention_mask, y)
                loss_mask = _choose_lm_loss_mask(attention_mask, y)
                if loss_mask is not None:
                    loss_mask = loss_mask.to(device).bool()

                out = _forward_lm(model, model_name, x, forward_mask)
                logits = out["logits"] if isinstance(out, dict) else out
                loss, n_tokens = _masked_xent_loss(logits, y, loss_mask)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                val_nll += float(loss.item()) * n_tokens
                val_tokens += n_tokens

                # --- accuracy on supervised tokens ---
                # For Copy tasks this corresponds to the copied target span.
                # For MQAR it corresponds to the answer tokens.
                ignore_idx = -100 if int(y.min().item()) < 0 else 0
                if loss_mask is not None:
                    acc_mask = loss_mask.to(device).bool()
                else:
                    acc_mask = (y != ignore_idx)

                if acc_mask.any():
                    preds = logits.argmax(dim=-1)
                    tok_correct = ((preds == y) & acc_mask).sum().item()
                    tok_total = acc_mask.sum().item()
                    val_tok_correct += float(tok_correct)
                    val_tok_total += float(tok_total)

                    # Sequence accuracy: all supervised tokens correct.
                    seq_correct = (((preds == y) | (~acc_mask)).all(dim=1)).sum().item()
                    val_seq_correct += float(seq_correct)
                    val_seq_total += float(y.size(0))

                # Track peak memory during validation/inference as well.
                mem_val = _mem_update(proc, mem_val, device)
                rss_samples_val.append(_rss_mb(proc))

        avg_val = val_nll / val_tokens if val_tokens else float("nan")
        last_val_loss = float(avg_val)
        ppl = float(torch.exp(torch.tensor(avg_val)).item())

        if val_tok_total > 0:
            last_val_token_acc = 100.0 * (val_tok_correct / val_tok_total)
        else:
            last_val_token_acc = float("nan")

        if val_seq_total > 0:
            last_val_seq_acc = 100.0 * (val_seq_correct / val_seq_total)
        else:
            last_val_seq_acc = float("nan")

        print(f"    epoch {epoch+1}/{num_epochs} train={avg_train:.4f} val={avg_val:.4f} ppl={ppl:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            convergence_steps = (epoch + 1) * len(train_loader)
            best_val_token_acc = last_val_token_acc
            best_val_seq_acc = last_val_seq_acc

    param_count = sum(p.numel() for p in model.parameters())

    gpu_peak_mb_train = mem_train.gpu_peak_mb
    if device.startswith("cuda"):
        gpu_peak_mb_train = max(gpu_peak_mb_train, torch.cuda.max_memory_allocated() / 1024 / 1024)

    gpu_peak_mb_val = mem_val.gpu_peak_mb
    if device.startswith("cuda"):
        gpu_peak_mb_val = max(gpu_peak_mb_val, torch.cuda.max_memory_allocated() / 1024 / 1024)

    # Overall (back-compat) peaks across train+val
    cpu_peak_mb = max(mem_train.cpu_peak_mb, mem_val.cpu_peak_mb)
    gpu_peak_mb = max(gpu_peak_mb_train, gpu_peak_mb_val)

    # Refresh mem snapshots with final peaks
    mem_train = MemStats(mem_train.cpu_start_mb, mem_train.cpu_peak_mb, mem_train.gpu_start_mb, gpu_peak_mb_train)
    mem_val = MemStats(mem_val.cpu_start_mb, mem_val.cpu_peak_mb, mem_val.gpu_start_mb, gpu_peak_mb_val)
    mem = MemStats(mem_train.cpu_start_mb, cpu_peak_mb, mem_train.gpu_start_mb, gpu_peak_mb)

    rss_train = _summarize_rss_mb(rss_samples_train)
    rss_val = _summarize_rss_mb(rss_samples_val)
    rss_all = _summarize_rss_mb(rss_samples_train + rss_samples_val)

    latency_sec_per_step = float(np.mean(step_times)) if step_times else 0.0
    throughput_samples_per_sec = (n_train_samples / total_time) if total_time else 0.0

    best_val_ppl = float(torch.exp(torch.tensor(best_val_loss)).item())
    last_val_ppl = float(torch.exp(torch.tensor(last_val_loss)).item()) if not np.isnan(last_val_loss) else float("nan")

    return {
        "parameters": int(param_count),
        # "final" refers to the last validation pass, while "best_*" is the best
        # epoch by validation loss.
        "final_loss": float(last_val_loss),
        "perplexity": float(last_val_ppl),
        "validation_loss": float(last_val_loss),
        "validation_perplexity": float(last_val_ppl),
        "best_validation_loss": float(best_val_loss),
        "best_validation_perplexity": float(best_val_ppl),
        # Accuracy over supervised tokens (Copy target span / MQAR answers).
        "validation_token_accuracy_pct": float(round(last_val_token_acc, 2)) if last_val_token_acc == last_val_token_acc else float("nan"),
        "validation_sequence_accuracy_pct": float(round(last_val_seq_acc, 2)) if last_val_seq_acc == last_val_seq_acc else float("nan"),
        "best_validation_token_accuracy_pct": float(round(best_val_token_acc, 2)) if best_val_token_acc == best_val_token_acc else float("nan"),
        "best_validation_sequence_accuracy_pct": float(round(best_val_seq_acc, 2)) if best_val_seq_acc == best_val_seq_acc else float("nan"),
        "training_time_hours": total_time / 3600.0,
        "convergence_steps": int(convergence_steps),
        "peak_gradient_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "memory_efficiency_mb_per_param": (mem.cpu_peak_mb / param_count) if param_count else 0.0,
        "latency_sec_per_step": latency_sec_per_step,
        "throughput_samples_per_sec": throughput_samples_per_sec,
        "cpu_mem_start_mb": mem.cpu_start_mb,
        "cpu_mem_peak_mb": mem.cpu_peak_mb,
        "cpu_mem_delta_mb": mem.cpu_delta_mb,
        # RSS summaries (StreamingLLM-style)
        "cpu_rss_mean_mb": rss_all["mean"],
        "cpu_rss_median_mb": rss_all["median"],
        "gpu_mem_start_mb": mem.gpu_start_mb,
        "gpu_mem_peak_mb": mem.gpu_peak_mb,
        "gpu_mem_delta_mb": mem.gpu_delta_mb,
        "cpu_mem_start_mb_train": mem_train.cpu_start_mb,
        "cpu_mem_peak_mb_train": mem_train.cpu_peak_mb,
        "cpu_mem_delta_mb_train": mem_train.cpu_delta_mb,
        "cpu_rss_mean_mb_train": rss_train["mean"],
        "cpu_rss_median_mb_train": rss_train["median"],
        "gpu_mem_start_mb_train": mem_train.gpu_start_mb,
        "gpu_mem_peak_mb_train": mem_train.gpu_peak_mb,
        "gpu_mem_delta_mb_train": mem_train.gpu_delta_mb,
        "cpu_mem_start_mb_val": mem_val.cpu_start_mb,
        "cpu_mem_peak_mb_val": mem_val.cpu_peak_mb,
        "cpu_mem_delta_mb_val": mem_val.cpu_delta_mb,
        "cpu_rss_mean_mb_val": rss_val["mean"],
        "cpu_rss_median_mb_val": rss_val["median"],
        "gpu_mem_start_mb_val": mem_val.gpu_start_mb,
        "gpu_mem_peak_mb_val": mem_val.gpu_peak_mb,
        "gpu_mem_delta_mb_val": mem_val.gpu_delta_mb,
        "model_size_mb": (param_count * 4) / 1024 / 1024,
        "flops_per_token": 0,
    }


def train_eval_clf(
    *,
    base_model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    n_train_samples: int,
    device: str,
    d_model: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    num_epochs: int,
) -> Dict[str, Any]:
    clf = ClassifierModel(base_model, d_model=d_model, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader), eta_min=lr / 10)
    criterion = nn.CrossEntropyLoss()

    proc, mem_train = _mem_begin(device)
    mem_val = _mem_begin_phase(proc, device)

    rss_samples_train: list[float] = []
    rss_samples_val: list[float] = []

    best_val_loss = float("inf")
    last_val_loss = float("nan")
    last_acc = 0.0
    best_acc = 0.0
    total_time = 0.0
    grad_norms = []
    step_times = []
    convergence_steps = 0

    for epoch in range(num_epochs):
        clf.train()
        train_loss = 0.0
        n_steps = 0
        t_epoch = time.time()

        for batch in train_loader:
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            mask = batch.get("attention_mask")
            if mask is not None:
                mask = mask.to(device)

            t_step = time.time()
            optimizer.zero_grad(set_to_none=True)
            logits = clf(x, attention_mask=mask, model_name=model_name)
            loss = criterion(logits, y)
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            gn = 0.0
            for p in clf.parameters():
                if p.grad is not None:
                    gn += p.grad.data.norm(2).item() ** 2
            grad_norms.append(gn ** 0.5)

            nn.utils.clip_grad_norm_(clf.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            train_loss += float(loss.item())
            n_steps += 1
            step_times.append(time.time() - t_step)
            mem_train = _mem_update(proc, mem_train, device)
            rss_samples_train.append(_rss_mb(proc))

        total_time += time.time() - t_epoch
        avg_train = train_loss / n_steps if n_steps else float("nan")

        # Validation
        clf.eval()
        val_loss_sum = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["input_ids"].to(device)
                y = batch["labels"].to(device)
                mask = batch.get("attention_mask")
                if mask is not None:
                    mask = mask.to(device)

                logits = clf(x, attention_mask=mask, model_name=model_name)
                loss = criterion(logits, y)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss_sum += float(loss.item())
                preds = logits.argmax(dim=-1)
                correct += int((preds == y).sum().item())
                total += int(y.size(0))

                # Track peak memory during validation/inference as well.
                mem_val = _mem_update(proc, mem_val, device)
                rss_samples_val.append(_rss_mb(proc))

        avg_val = val_loss_sum / len(val_loader) if len(val_loader) else float("nan")
        acc = 100.0 * correct / total if total else 0.0
        last_val_loss = float(avg_val)
        last_acc = float(acc)
        print(f"    epoch {epoch+1}/{num_epochs} train_loss={avg_train:.4f} val_loss={avg_val:.4f} acc={acc:.1f}%")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_acc = acc
            convergence_steps = (epoch + 1) * len(train_loader)

    param_count = sum(p.numel() for p in clf.parameters())

    gpu_peak_mb_train = mem_train.gpu_peak_mb
    if device.startswith("cuda"):
        gpu_peak_mb_train = max(gpu_peak_mb_train, torch.cuda.max_memory_allocated() / 1024 / 1024)

    gpu_peak_mb_val = mem_val.gpu_peak_mb
    if device.startswith("cuda"):
        gpu_peak_mb_val = max(gpu_peak_mb_val, torch.cuda.max_memory_allocated() / 1024 / 1024)

    cpu_peak_mb = max(mem_train.cpu_peak_mb, mem_val.cpu_peak_mb)
    gpu_peak_mb = max(gpu_peak_mb_train, gpu_peak_mb_val)

    mem_train = MemStats(mem_train.cpu_start_mb, mem_train.cpu_peak_mb, mem_train.gpu_start_mb, gpu_peak_mb_train)
    mem_val = MemStats(mem_val.cpu_start_mb, mem_val.cpu_peak_mb, mem_val.gpu_start_mb, gpu_peak_mb_val)
    mem = MemStats(mem_train.cpu_start_mb, cpu_peak_mb, mem_train.gpu_start_mb, gpu_peak_mb)

    rss_train = _summarize_rss_mb(rss_samples_train)
    rss_val = _summarize_rss_mb(rss_samples_val)
    rss_all = _summarize_rss_mb(rss_samples_train + rss_samples_val)

    latency_sec_per_step = float(np.mean(step_times)) if step_times else 0.0
    throughput_samples_per_sec = (n_train_samples / total_time) if total_time else 0.0

    classification_perplexity = float(torch.exp(torch.tensor(last_val_loss)).item()) if not np.isnan(last_val_loss) else float("nan")
    best_classification_perplexity = float(torch.exp(torch.tensor(best_val_loss)).item()) if not np.isnan(best_val_loss) else float("nan")

    return {
        "parameters": int(param_count),
        "final_loss": float(last_val_loss),
        "classification_perplexity": float(classification_perplexity),
        "val_accuracy_pct": float(round(last_acc, 2)),
        "validation_loss": float(last_val_loss),
        "validation_classification_perplexity": float(classification_perplexity),
        "best_validation_loss": float(best_val_loss),
        "best_validation_classification_perplexity": float(best_classification_perplexity),
        "best_val_accuracy_pct": float(round(best_acc, 2)),
        "training_time_hours": total_time / 3600.0,
        "convergence_steps": int(convergence_steps),
        "peak_gradient_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "memory_efficiency_mb_per_param": (mem.cpu_peak_mb / param_count) if param_count else 0.0,
        "latency_sec_per_step": latency_sec_per_step,
        "throughput_samples_per_sec": throughput_samples_per_sec,
        "cpu_mem_start_mb": mem.cpu_start_mb,
        "cpu_mem_peak_mb": mem.cpu_peak_mb,
        "cpu_mem_delta_mb": mem.cpu_delta_mb,
        "cpu_rss_mean_mb": rss_all["mean"],
        "cpu_rss_median_mb": rss_all["median"],
        "gpu_mem_start_mb": mem.gpu_start_mb,
        "gpu_mem_peak_mb": mem.gpu_peak_mb,
        "gpu_mem_delta_mb": mem.gpu_delta_mb,
        "cpu_mem_start_mb_train": mem_train.cpu_start_mb,
        "cpu_mem_peak_mb_train": mem_train.cpu_peak_mb,
        "cpu_mem_delta_mb_train": mem_train.cpu_delta_mb,
        "cpu_rss_mean_mb_train": rss_train["mean"],
        "cpu_rss_median_mb_train": rss_train["median"],
        "gpu_mem_start_mb_train": mem_train.gpu_start_mb,
        "gpu_mem_peak_mb_train": mem_train.gpu_peak_mb,
        "gpu_mem_delta_mb_train": mem_train.gpu_delta_mb,
        "cpu_mem_start_mb_val": mem_val.cpu_start_mb,
        "cpu_mem_peak_mb_val": mem_val.cpu_peak_mb,
        "cpu_mem_delta_mb_val": mem_val.cpu_delta_mb,
        "cpu_rss_mean_mb_val": rss_val["mean"],
        "cpu_rss_median_mb_val": rss_val["median"],
        "gpu_mem_start_mb_val": mem_val.gpu_start_mb,
        "gpu_mem_peak_mb_val": mem_val.gpu_peak_mb,
        "gpu_mem_delta_mb_val": mem_val.gpu_delta_mb,
        "model_size_mb": (param_count * 4) / 1024 / 1024,
        "flops_per_token": 0,
    }


# -----------------------------------------------------------------------------
# Task factories
# -----------------------------------------------------------------------------
def _make_unified_lm_loaders(*, ds_name: str, seq_len: int, n_train: int, n_val: int, batch_size: int, seed: int) -> Tuple[DataLoader, DataLoader, int, int]:
    # UnifiedDataset seeds internally via torch/numpy/random; set seeds here for reproducibility.
    torch.manual_seed(seed)
    np.random.seed(seed)

    # For MQAR specifically, ensure train/val are generated with different seeds
    # to avoid accidental leakage (and to keep the task meaningful).
    if ds_name.strip().lower() == "mqar":
        train_ds = UnifiedDataset(ds_name, num_samples=n_train, seq_length=seq_len, seed=seed)
        val_ds = UnifiedDataset(ds_name, num_samples=n_val, seq_length=seq_len, seed=seed + 1000)
    else:
        train_ds = UnifiedDataset(ds_name, num_samples=n_train, seq_length=seq_len)
        val_ds = UnifiedDataset(ds_name, num_samples=n_val, seq_length=seq_len)
    vocab_size = int(getattr(train_ds, "vocab_size", 0) or train_ds.dataset.total_vocab_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, vocab_size, n_train


def _make_copy_loaders(*, copy_len: int, seq_len: int, n_train: int, n_val: int, batch_size: int, seed: int) -> Tuple[DataLoader, DataLoader, int, int]:
    train_ds = CopyDataset(num_sequences=n_train, seq_len=seq_len, copy_length=copy_len, seed=seed)
    val_ds = CopyDataset(num_sequences=n_val, seq_len=seq_len, copy_length=copy_len, seed=seed + 1000)
    vocab_size = int(train_ds.total_vocab_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, vocab_size, n_train


def _make_pg19_loaders(
    *,
    seq_len: int,
    n_train: int,
    n_val: int,
    batch_size: int,
    seed: int,
    pg19_root: Optional[Path],
    pg19_zip: Optional[Path],
    cache_dir: Path,
) -> Tuple[DataLoader, DataLoader, int, int]:
    from data.pg19.pg19_dataset import PG19ByteLMDataset, PG19ZipByteLMDataset, ensure_pg19_extracted, is_pg19_root

    if pg19_root is not None:
        root = pg19_root.expanduser().resolve()
        if not is_pg19_root(root):
            raise RuntimeError(
                f"PG-19 root does not look valid: {root}. "
                "Expected metadata.csv + train/validation/test folders with .txt books."
            )
        train_ds = PG19ByteLMDataset(root=root, split="train", seq_len=seq_len, num_samples=n_train, seed=seed)
        val_ds = PG19ByteLMDataset(root=root, split="validation", seq_len=seq_len, num_samples=n_val, seed=seed + 1000)
        vocab_size = int(train_ds.total_vocab_size)
    elif pg19_zip is not None:
        # Default: read directly from ZIP to avoid huge extraction cost.
        train_ds = PG19ZipByteLMDataset(zip_path=pg19_zip, split="train", seq_len=seq_len, num_samples=n_train, seed=seed)
        val_ds = PG19ZipByteLMDataset(zip_path=pg19_zip, split="validation", seq_len=seq_len, num_samples=n_val, seed=seed + 1000)
        vocab_size = int(train_ds.total_vocab_size)
    else:
        raise RuntimeError(
            "PG-19 is enabled but no source was provided. "
            "Pass --pg19-root <extracted_pg19_dir> or --pg19-zip <pg19_zip_path>."
        )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, vocab_size, n_train


def _make_chomsky_loaders(*, seq_len: int, n_train: int, n_val: int, batch_size: int, seed: int) -> Tuple[DataLoader, DataLoader, int, int]:
    train_ds = ParenthesesDataset(num_sequences=n_train, max_length=seq_len, seed=seed)
    val_ds = ParenthesesDataset(num_sequences=n_val, max_length=seq_len, seed=seed + 1000)
    vocab_size = int(train_ds.total_vocab_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, vocab_size, n_train


def _make_pathx_loaders(
    *,
    seq_len: int,
    n_train: int,
    n_val: int,
    batch_size: int,
    seed: int,
    difficulty: int = 1,
    source: str = "synthetic",
    official_dir: Optional[Path] = None,
    official_cache_dir: Optional[Path] = None,
    download_official: bool = False,
    variant: str = "pathx",
    split_name: str = "hard",
    allow_truncate: bool = False,
    local_dir: Optional[Path] = None,
    local_force_regen: bool = False,
) -> Tuple[DataLoader, DataLoader, int, int]:
    if source == "synthetic":
        train_ds = PathXClassificationDataset(num_samples=n_train, seq_length=seq_len, seed=seed, difficulty=difficulty)
        val_ds = PathXClassificationDataset(num_samples=n_val, seq_length=seq_len, seed=seed + 1000, difficulty=difficulty)
        vocab_size = int(train_ds.vocab_size)
    elif source == "lra":
        from data.official.pathfinder_lra_dataset import LRAPathfinderClassificationDataset
        from data.official.lra_release import LRA_RELEASE_ENV_ARCHIVE_PATH, ensure_lra_release_extracted

        if official_dir is None:
            has_env_archive = bool(os.environ.get(LRA_RELEASE_ENV_ARCHIVE_PATH))
            if not (download_official or has_env_archive):
                raise RuntimeError(
                    "Path-X official source requested but no data directory provided. "
                    "Either pass --pathx-official-dir <path> (folder containing extracted LRA release), "
                    f"or set {LRA_RELEASE_ENV_ARCHIVE_PATH} to a local lra_release.gz, "
                    "or pass --download-official to download the LRA release archive."
                )
            cache_dir = official_cache_dir or (REPO_ROOT / "data_cache")
            paths = ensure_lra_release_extracted(cache_dir / "lra_release")
            official_dir = paths.extract_dir

        train_ds = LRAPathfinderClassificationDataset(
            split="train",
            seq_length=seq_len,
            official_root=official_dir,
            variant=variant,
            difficulty=split_name,
            max_samples=n_train,
            strict_seq_length=not allow_truncate,
        )
        val_ds = LRAPathfinderClassificationDataset(
            split="val",
            seq_length=seq_len,
            official_root=official_dir,
            variant=variant,
            difficulty=split_name,
            max_samples=n_val,
            strict_seq_length=not allow_truncate,
        )
        vocab_size = int(train_ds.vocab_size)
    elif source == "local":
        # Local Pathfinder32-hard proxy generator (no external downloads).
        from data.local.pathfinder32_hard_generator import ensure_pathfinder32_hard_generated
        from data.official.pathfinder_lra_dataset import LRAPathfinderClassificationDataset

        cache_dir = (local_dir or (REPO_ROOT / "data_cache" / "pathfinder32_hard_local")).expanduser().resolve()
        ensure_pathfinder32_hard_generated(
            cache_dir,
            n_train=n_train,
            n_val=n_val,
            n_test=max(n_val, 200),
            seed=seed,
            force=bool(local_force_regen),
        )

        train_ds = LRAPathfinderClassificationDataset(
            split="train",
            seq_length=seq_len,
            official_root=cache_dir,
            variant="pathfinder32",
            difficulty=split_name,
            max_samples=n_train,
            # Local generator always produces fixed 32x32; keep strict.
            strict_seq_length=True,
        )
        val_ds = LRAPathfinderClassificationDataset(
            split="val",
            seq_length=seq_len,
            official_root=cache_dir,
            variant="pathfinder32",
            difficulty=split_name,
            max_samples=n_val,
            strict_seq_length=True,
        )
        vocab_size = int(train_ds.vocab_size)
    else:
        raise ValueError(f"Unknown pathx source: {source}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, vocab_size, n_train


def _make_listops_loaders(
    *,
    seq_len: int,
    n_train: int,
    n_val: int,
    batch_size: int,
    seed: int,
    max_depth: int = 3,
    source: str = "synthetic",
    official_dir: Optional[Path] = None,
    official_cache_dir: Optional[Path] = None,
    download_official: bool = False,
    official_task: str = "basic",
) -> Tuple[DataLoader, DataLoader, int, int]:
    if source == "synthetic":
        train_ds = ListOpsClassificationDataset(num_samples=n_train, seq_length=seq_len, seed=seed, max_depth=max_depth)
        val_ds = ListOpsClassificationDataset(num_samples=n_val, seq_length=seq_len, seed=seed + 1000, max_depth=max_depth)
        vocab_size = int(train_ds.vocab_size)
    elif source == "lra":
        from data.official.listops_lra_dataset import LRAListOpsClassificationDataset
        from data.official.lra_release import LRA_RELEASE_ENV_ARCHIVE_PATH, ensure_lra_release_extracted

        if official_dir is None:
            has_env_archive = bool(os.environ.get(LRA_RELEASE_ENV_ARCHIVE_PATH))
            if not (download_official or has_env_archive):
                raise RuntimeError(
                    "ListOps official source requested but no data directory provided. "
                    "Either pass --listops-official-dir <path> (folder containing basic_train.tsv/basic_val.tsv), "
                    f"or set {LRA_RELEASE_ENV_ARCHIVE_PATH} to a local lra_release.gz, "
                    "or pass --download-official to download the LRA release archive."
                )
            cache_dir = official_cache_dir or (REPO_ROOT / "data_cache")
            paths = ensure_lra_release_extracted(cache_dir / "lra_release")
            official_dir = paths.extract_dir

        train_ds = LRAListOpsClassificationDataset(
            split="train",
            seq_length=seq_len,
            official_root=official_dir,
            task=official_task,
            max_samples=n_train,
        )
        val_ds = LRAListOpsClassificationDataset(
            split="val",
            seq_length=seq_len,
            official_root=official_dir,
            task=official_task,
            max_samples=n_val,
        )
        vocab_size = int(train_ds.vocab_size)
    else:
        raise ValueError(f"Unknown listops source: {source}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, vocab_size, n_train


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Run the notebook benchmark subset and write results/complete_benchmark.json")
    parser.add_argument("--run-id", type=int, choices=(1, 2), default=1, help="Seed variant (default: 1)")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output JSON path (default: results/complete_benchmark.json). "
            "Use this to write separate files, e.g. results/benchmark_run1.json"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--only",
        default="all",
        choices=("all", "c1", "c2"),
        help="Run only C1 tasks, only C2 tasks, or all tasks",
    )

    parser.add_argument(
        "--tasks",
        default=None,
        help=(
            "Comma-separated task keys to run (e.g. MQAR,PG19,copy_len100_seq256,path_x_clf). "
            "By default, runs all tasks enabled by your flags."
        ),
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print available task keys and exit",
    )

    # Keep these defaults aligned with existing scripts
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-val", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=144)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--local-window", type=int, default=64)

    # C2 difficulty knobs (defaults match previous behavior)
    parser.add_argument("--pathx-seq-len", type=int, default=256)
    parser.add_argument("--pathx-difficulty", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument(
        "--pathx-source",
        choices=("synthetic", "lra", "local"),
        default="synthetic",
        help="Path-X dataset source: synthetic proxy, official LRA release (if available), or local Pathfinder32 generator",
    )
    parser.add_argument(
        "--pathx-variant",
        choices=("pathx", "pathfinder32", "pathfinder64", "pathfinder128", "pathfinder256"),
        default="pathx",
        help=(
            "Which LRA Pathfinder variant to load when --pathx-source=lra. "
            "Note: 'pathx' is treated as an alias for the common longer-setting (pathfinder128)."
        ),
    )
    parser.add_argument(
        "--pathx-split",
        choices=("easy", "intermediate", "hard"),
        default="hard",
        help="Pathfinder difficulty split (LRA uses easy/intermediate/hard; most benchmarking uses hard)",
    )
    parser.add_argument(
        "--pathx-allow-truncate",
        action="store_true",
        help="Allow truncating/padding official Pathfinder inputs to --pathx-seq-len (faster but not apples-to-apples)",
    )
    parser.add_argument(
        "--pathx-official-dir",
        default=None,
        help="Path to extracted LRA release directory (or a folder containing Pathfinder/Path-X split files)",
    )
    parser.add_argument(
        "--pathx-local-dir",
        default=None,
        help="Directory to cache locally generated Pathfinder32 hard NPZ splits when --pathx-source=local",
    )
    parser.add_argument(
        "--pathx-local-force-regen",
        action="store_true",
        help="Force re-generating the local Pathfinder32 dataset cache",
    )
    parser.add_argument("--listops-seq-len", type=int, default=256)
    parser.add_argument("--listops-max-depth", type=int, default=3)
    parser.add_argument(
        "--listops-source",
        choices=("synthetic", "lra"),
        default="synthetic",
        help="ListOps dataset source: current synthetic generator or official LRA TSV",
    )
    parser.add_argument(
        "--listops-official-dir",
        default=None,
        help="Path to a folder containing LRA ListOps TSV files (e.g. basic_train.tsv/basic_val.tsv/basic_test.tsv)",
    )
    parser.add_argument(
        "--listops-task",
        default="basic",
        help="LRA ListOps task name (default: basic; corresponds to <task>_train.tsv)",
    )
    parser.add_argument(
        "--official-cache-dir",
        default=str(REPO_ROOT / "data_cache"),
        help="Cache directory used for downloading official datasets (when enabled)",
    )
    parser.add_argument(
        "--download-official",
        action="store_true",
        help="Download the official LRA release archive into --official-cache-dir if needed",
    )

    # PG-19 (official split) options (byte-level LM loader)
    parser.add_argument(
        "--pg19-seq-len",
        type=int,
        default=1024,
        help="Sequence length for PG-19 byte-level LM task (default: 1024)",
    )
    parser.add_argument(
        "--pg19-root",
        default=None,
        help="Path to extracted PG-19 folder (must contain metadata.csv and train/validation/test subfolders)",
    )
    parser.add_argument(
        "--pg19-zip",
        default=None,
        help=(
            "Path to a PG-19 zip (official split layout). "
            "If provided, the benchmark will read books directly from the zip (no full extraction)."
        ),
    )

    args = parser.parse_args()

    if args.list_tasks:
        print("Available task keys:")
        for k in [
            "MQAR",
            "MQAR_seq512",
            "MQAR_seq1024",
            "PG19",
            "copy_len100_seq256",
            "path_x_clf",
            "chomsky_clf",
            "listops_clf",
        ]:
            print(f"  - {k}")
        print("\nNote: PG19 is optional and will be skipped unless --pg19-root or --pg19-zip is provided.")
        return 0

    listops_official_dir = Path(args.listops_official_dir).expanduser().resolve() if args.listops_official_dir else None
    pathx_official_dir = Path(args.pathx_official_dir).expanduser().resolve() if args.pathx_official_dir else None
    pathx_local_dir = Path(args.pathx_local_dir).expanduser().resolve() if args.pathx_local_dir else None
    official_cache_dir = Path(args.official_cache_dir).expanduser().resolve() if args.official_cache_dir else None
    pg19_root = Path(args.pg19_root).expanduser().resolve() if args.pg19_root else None
    pg19_zip = Path(args.pg19_zip).expanduser().resolve() if args.pg19_zip else None

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU")
        device = "cpu"

    out_path = Path(args.out) if args.out else (REPO_ROOT / "results" / "complete_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Per-run seed: make run1/run2 independent-ish
    seed = 42 + int(args.run_id)

    def mk_unified(ds_name: str, seq_len: int):
        return _make_unified_lm_loaders(ds_name=ds_name, seq_len=seq_len, n_train=args.n_train, n_val=args.n_val, batch_size=args.batch_size, seed=seed)

    tasks_lm = [
        LmTask("MQAR", GROUP_1, lambda: mk_unified("mqar", 256)),
        LmTask("MQAR_seq512", GROUP_1, lambda: mk_unified("mqar", 512)),
        LmTask("MQAR_seq1024", GROUP_1, lambda: mk_unified("mqar", 1024)),
    ]

    # PG-19 (official split) byte-level LM task
    # Make PG-19 optional so this runner stays usable without local PG-19 data.
    if pg19_root is not None or pg19_zip is not None:
        tasks_lm.append(
            LmTask(
                "PG19",
                GROUP_1,
                lambda: _make_pg19_loaders(
                    seq_len=args.pg19_seq_len,
                    n_train=args.n_train,
                    n_val=args.n_val,
                    batch_size=max(4, args.batch_size // 2),
                    seed=seed,
                    pg19_root=pg19_root,
                    pg19_zip=pg19_zip,
                    cache_dir=official_cache_dir or (REPO_ROOT / "data_cache"),
                ),
            )
        )
    else:
        print("PG19 disabled (no --pg19-root / --pg19-zip provided).")

    # C2 copy
    tasks_lm.append(
        LmTask(
            "copy_len100_seq256",
            GROUP_2,
            lambda: _make_copy_loaders(
                copy_len=100,
                seq_len=256,
                n_train=args.n_train,
                n_val=args.n_val,
                batch_size=args.batch_size,
                seed=seed,
            ),
        )
    )

    tasks_lm = tuple(tasks_lm)

    tasks_clf = (
        ClfTask(
            "path_x_clf",
            GROUP_2,
            2,
            lambda: _make_pathx_loaders(
                seq_len=args.pathx_seq_len,
                n_train=args.n_train,
                n_val=args.n_val,
                batch_size=args.batch_size,
                seed=seed,
                difficulty=args.pathx_difficulty,
                source=args.pathx_source,
                official_dir=pathx_official_dir,
                official_cache_dir=official_cache_dir,
                download_official=bool(args.download_official),
                variant=args.pathx_variant,
                split_name=args.pathx_split,
                allow_truncate=bool(args.pathx_allow_truncate),
                local_dir=pathx_local_dir,
                local_force_regen=bool(args.pathx_local_force_regen),
            ),
        ),
        ClfTask("chomsky_clf", GROUP_2, 2, lambda: _make_chomsky_loaders(seq_len=256, n_train=args.n_train, n_val=args.n_val, batch_size=args.batch_size, seed=seed)),
        ClfTask(
            "listops_clf",
            GROUP_2,
            10,
            lambda: _make_listops_loaders(
                seq_len=args.listops_seq_len,
                n_train=args.n_train,
                n_val=args.n_val,
                batch_size=args.batch_size,
                seed=seed,
                max_depth=args.listops_max_depth,
                source=args.listops_source,
                official_dir=listops_official_dir,
                official_cache_dir=official_cache_dir,
                download_official=bool(args.download_official),
                official_task=args.listops_task,
            ),
        ),
    )

    # --- Optional: restrict to a subset of task keys ---
    all_task_keys = [t.key for t in tasks_lm] + [t.key for t in tasks_clf]
    selected_keys = None
    if args.tasks:
        requested = [s.strip() for s in str(args.tasks).split(",") if s.strip()]
        if not requested:
            raise SystemExit("--tasks was provided but no task keys were parsed")

        # Resolve case-insensitively for convenience.
        key_by_lower = {k.lower(): k for k in all_task_keys}
        resolved: list[str] = []
        unknown: list[str] = []
        for r in requested:
            if r in all_task_keys:
                resolved.append(r)
                continue
            k = key_by_lower.get(r.lower())
            if k is None:
                unknown.append(r)
            else:
                resolved.append(k)

        if unknown:
            avail = ", ".join(all_task_keys)
            raise SystemExit(f"Unknown task key(s): {unknown}. Available: {avail}")

        selected_keys = set(resolved)

    # Build a config blob we can persist into the JSON for reproducibility.
    cfg_blob = {
        "run_id": int(args.run_id),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "seed": seed,
        "config": {
            "n_train": args.n_train,
            "n_val": args.n_val,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "d_model": args.d_model,
            "layers": args.layers,
            "heads": args.heads,
            "local_window": args.local_window,
            "pathx_source": args.pathx_source,
            "pathx_variant": args.pathx_variant,
            "pathx_split": args.pathx_split,
            "pathx_allow_truncate": bool(args.pathx_allow_truncate),
            "listops_source": args.listops_source,
            "listops_task": args.listops_task,
            "pg19_seq_len": args.pg19_seq_len,
            "pg19_root": str(pg19_root) if pg19_root else None,
            "pg19_zip": str(pg19_zip) if pg19_zip else None,
        },
    }

    results: Dict[str, Any]
    if out_path.exists() and not args.overwrite:
        try:
            results = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise SystemExit(f"Failed to read existing output JSON: {out_path} ({e})")
        if not isinstance(results, dict):
            raise SystemExit(f"Existing output JSON is not an object: {out_path}")
        # Keep existing content; update/attach meta for this run.
        prev_cfg = (results.get("_meta") or {}).get("config")
        if prev_cfg and isinstance(prev_cfg, dict) and prev_cfg != cfg_blob["config"]:
            print("WARNING: output JSON exists and config differs; resuming may mix runs.")
            print("         Consider using --overwrite or --out to start a clean file.")
        results["_meta"] = cfg_blob
    else:
        results = {"_meta": cfg_blob}

    results.setdefault(GROUP_1, {"_section": "Notebook subset C1"})
    results.setdefault(GROUP_2, {"_section": "Notebook subset C2"})

    def save() -> None:
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    save()
    print(f"Writing results to: {out_path}")

    # --- LM tasks ---
    for task in tasks_lm:
        if selected_keys is not None and task.key not in selected_keys:
            continue
        if args.only == "c1" and task.group != GROUP_1:
            continue
        if args.only == "c2" and task.group != GROUP_2:
            continue
        print(f"\n{'='*72}\nTASK {task.key} ({task.group})\n{'='*72}")
        train_loader, val_loader, vocab_size, n_train_samples = task.make_loaders()

        results[task.group].setdefault(task.key, {})
        save()

        models = make_models(
            vocab_size=vocab_size,
            seq_len=next(iter(train_loader))["input_ids"].shape[1],
            d_model=args.d_model,
            num_layers=args.layers,
            num_heads=args.heads,
            local_window=args.local_window,
        )

        for model_name in MODELS:
            print(f"\n  --- {model_name} ---")
            existing = results.get(task.group, {}).get(task.key, {}).get(model_name)
            if isinstance(existing, dict) and any(k in existing for k in ("perplexity", "final_loss", "latency_sec_per_step", "throughput_samples_per_sec")):
                print("    (skip: already present in JSON)")
                continue
            metrics = train_eval_lm(
                model=models[model_name],
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                n_train_samples=n_train_samples,
                device=device,
                lr=args.lr,
                weight_decay=args.weight_decay,
                grad_clip=args.grad_clip,
                num_epochs=args.epochs,
            )
            results[task.group][task.key][model_name] = metrics
            save()

    # --- Classification tasks ---
    for task in tasks_clf:
        if selected_keys is not None and task.key not in selected_keys:
            continue
        if args.only == "c1":
            continue
        print(f"\n{'='*72}\nTASK {task.key} ({task.group})\n{'='*72}")
        train_loader, val_loader, vocab_size, n_train_samples = task.make_loaders()

        results[task.group].setdefault(task.key, {})
        save()

        base_models = make_models(
            vocab_size=vocab_size,
            seq_len=next(iter(train_loader))["input_ids"].shape[1],
            d_model=args.d_model,
            num_layers=args.layers,
            num_heads=args.heads,
            local_window=args.local_window,
        )

        for model_name in MODELS:
            print(f"\n  --- {model_name} ---")
            existing = results.get(task.group, {}).get(task.key, {}).get(model_name)
            if isinstance(existing, dict) and any(k in existing for k in ("val_accuracy_pct", "final_loss", "latency_sec_per_step", "throughput_samples_per_sec")):
                print("    (skip: already present in JSON)")
                continue
            metrics = train_eval_clf(
                base_model=base_models[model_name],
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                num_classes=task.num_classes,
                n_train_samples=n_train_samples,
                device=device,
                d_model=args.d_model,
                lr=args.lr,
                weight_decay=args.weight_decay,
                grad_clip=args.grad_clip,
                num_epochs=args.epochs,
            )
            results[task.group][task.key][model_name] = metrics
            save()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
