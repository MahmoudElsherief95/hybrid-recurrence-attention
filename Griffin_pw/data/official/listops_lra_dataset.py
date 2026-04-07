from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ListOpsVocab:
    pad: int = 0
    open_paren: int = 11
    close_paren: int = 12
    min_op: int = 13
    max_op: int = 14
    med_op: int = 15
    sm_op: int = 16

    @property
    def vocab_size(self) -> int:
        # token ids are sparse; we set vocab_size to max_id+1
        return self.sm_op + 1


_TOKEN_RE = re.compile(r"\(|\)|\]|\[[A-Za-z]+|\d+")


def _tokenize_listops_source(source: str) -> List[str]:
    # LRA ListOps has tokens like '(', ')', '[MIN', '[MAX', '[MED', '[SM', ']', and digits.
    # We normalize them into a small token set.
    tokens = _TOKEN_RE.findall(source)
    norm: List[str] = []
    for tok in tokens:
        if tok == "(":
            norm.append("(")
        elif tok == ")" or tok == "]":
            norm.append(")")
        elif tok.startswith("["):
            norm.append(tok[1:])  # [MIN -> MIN
        else:
            norm.append(tok)
    return norm


def _encode_tokens(tokens: List[str], vocab: ListOpsVocab) -> List[int]:
    token_to_id: Dict[str, int] = {
        "(": vocab.open_paren,
        ")": vocab.close_paren,
        "MIN": vocab.min_op,
        "MAX": vocab.max_op,
        "MED": vocab.med_op,
        "SM": vocab.sm_op,
    }
    # digits 0..9 -> 1..10
    for d in range(10):
        token_to_id[str(d)] = d + 1

    ids: List[int] = []
    for tok in tokens:
        t = tok.strip()
        if not t:
            continue
        if t in token_to_id:
            ids.append(token_to_id[t])
        else:
            # unknown tokens are ignored (keeps loader robust to minor format variants)
            continue
    return ids


def _find_listops_tsv_dir(root: Path, task: str) -> Path:
    # Common layouts:
    # - <root>/{task}_train.tsv
    # - <root>/listops/{task}_train.tsv
    # - <root>/**/listops/**/{task}_train.tsv
    direct = root / f"{task}_train.tsv"
    if direct.exists():
        return root

    candidates = list(root.glob(f"**/{task}_train.tsv"))
    if candidates:
        # pick the shallowest
        candidates.sort(key=lambda p: len(p.parts))
        return candidates[0].parent

    raise FileNotFoundError(
        f"Could not find '{task}_train.tsv' under {root}. "
        "Point --listops-official-dir at a folder containing basic_train.tsv/basic_val.tsv/basic_test.tsv."
    )


def _read_listops_tsv(path: Path) -> List[Tuple[str, int]]:
    rows: List[Tuple[str, int]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        # expected header: Source, Target
        for parts in reader:
            if not parts:
                continue
            if len(parts) < 2:
                continue
            src, tgt = parts[0], parts[1]
            try:
                y = int(tgt)
            except ValueError:
                continue
            rows.append((src, y))
    return rows


class LRAListOpsClassificationDataset(Dataset):
    """Official LRA ListOps from TSV (10-class classification)."""

    def __init__(
        self,
        *,
        split: str,
        seq_length: int,
        official_root: Path,
        task: str = "basic",
        max_samples: Optional[int] = None,
        vocab: Optional[ListOpsVocab] = None,
    ):
        split = split.lower()
        if split not in ("train", "val", "valid", "validation", "test"):
            raise ValueError(f"Unsupported split: {split}")
        split_file = "val" if split in ("val", "valid", "validation") else split

        self.seq_length = int(seq_length)
        self.vocab = vocab or ListOpsVocab()
        self.vocab_size = self.vocab.vocab_size

        tsv_dir = _find_listops_tsv_dir(official_root, task=task)
        tsv_path = tsv_dir / f"{task}_{split_file}.tsv"
        if not tsv_path.exists():
            raise FileNotFoundError(f"Missing ListOps TSV: {tsv_path}")

        rows = _read_listops_tsv(tsv_path)
        if max_samples is not None:
            rows = rows[: int(max_samples)]

        self.samples: List[Dict[str, torch.Tensor]] = []
        pad = self.vocab.pad
        for src, y in rows:
            toks = _tokenize_listops_source(src)
            ids = _encode_tokens(toks, vocab=self.vocab)
            if len(ids) > self.seq_length:
                ids = ids[: self.seq_length]
            padded = ids + [pad] * (self.seq_length - len(ids))

            input_ids = torch.tensor(padded, dtype=torch.long)
            attention_mask = (input_ids != pad).long()
            label = int(y) % 10
            self.samples.append(
                {
                    "input_ids": input_ids,
                    "labels": torch.tensor(label, dtype=torch.long),
                    "attention_mask": attention_mask,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]
