from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PG19Paths:
    root: Path
    train_dir: Path
    val_dir: Path
    test_dir: Path


def _resolve_split_dir(root: Path, split: str) -> Path:
    """Handle both layouts:

    - <root>/train/train/*.txt (Kaggle zip)
    - <root>/train/*.txt
    """

    split = split.lower()
    cand1 = root / split / split
    if cand1.exists():
        return cand1
    cand2 = root / split
    if cand2.exists():
        return cand2
    raise FileNotFoundError(f"Missing PG-19 split directory for '{split}' under: {root}")


def is_pg19_root(root: Path) -> bool:
    root = root.expanduser().resolve()
    if not root.exists():
        return False
    if not (root / "metadata.csv").exists():
        return False
    for split in ("train", "validation", "test"):
        try:
            d = _resolve_split_dir(root, split)
        except FileNotFoundError:
            return False
        if not any(d.glob("*.txt")):
            return False
    return True


def ensure_pg19_extracted(cache_dir: Path, *, zip_path: Path) -> PG19Paths:
    """Ensure the PG-19 zip is extracted to a local directory.

    This expects the official split layout inside the zip:
      metadata.csv
      train/train/*.txt
      validation/validation/*.txt
      test/test/*.txt

    If already extracted, it is reused.
    """

    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    zip_path = zip_path.expanduser().resolve()
    if not zip_path.exists():
        raise FileNotFoundError(f"PG-19 zip not found: {zip_path}")

    root = cache_dir / "pg19_official"
    if is_pg19_root(root):
        return PG19Paths(
            root=root,
            train_dir=_resolve_split_dir(root, "train"),
            val_dir=_resolve_split_dir(root, "validation"),
            test_dir=_resolve_split_dir(root, "test"),
        )

    # Extract (large; one-time).
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(root)

    if not is_pg19_root(root):
        raise RuntimeError(
            "Extracted PG-19 folder does not look like PG-19. "
            f"Expected metadata.csv + train/validation/test .txt files under {root}."
        )

    return PG19Paths(
        root=root,
        train_dir=_resolve_split_dir(root, "train"),
        val_dir=_resolve_split_dir(root, "validation"),
        test_dir=_resolve_split_dir(root, "test"),
    )


class PG19ZipByteLMDataset(Dataset):
    """PG-19 byte-level LM dataset that reads directly from the official ZIP.

    This avoids a multi-GB extraction step. It is slower per sample than reading
    from extracted files, but is usually the best trade-off for quick benchmarks.

    Expected zip layout (Kaggle mirror / official split):
      metadata.csv
      train/train/<book_id>.txt
      validation/validation/<book_id>.txt
      test/test/<book_id>.txt
    """

    PAD_TOKEN = 0

    def __init__(
        self,
        *,
        zip_path: Path,
        split: str,
        seq_len: int,
        num_samples: int,
        seed: int,
        min_bytes_per_book: int = 4096,
    ):
        self.zip_path = zip_path.expanduser().resolve()
        if not self.zip_path.exists():
            raise FileNotFoundError(f"PG-19 zip not found: {self.zip_path}")

        self.split = split.lower()
        if self.split not in ("train", "validation", "test"):
            raise ValueError(f"Unsupported PG-19 split: {split}")

        self.seq_len = int(seq_len)
        self.num_samples = int(num_samples)
        self.seed = int(seed)

        self.total_vocab_size = 257

        # Keep the ZIP open for speed (num_workers=0 in our runner).
        self._zip = zipfile.ZipFile(self.zip_path, "r")
        names = self._zip.namelist()

        prefix = f"{self.split}/{self.split}/"
        txts = [n for n in names if n.startswith(prefix) and n.lower().endswith(".txt")]
        if not txts:
            # Fallback: support layout without duplicated split folder.
            prefix2 = f"{self.split}/"
            txts = [n for n in names if n.startswith(prefix2) and n.lower().endswith(".txt")]
            prefix = prefix2

        if not txts:
            raise RuntimeError(f"No .txt files found in PG-19 zip for split '{self.split}'")

        # Filter usable books by compressed size info.
        usable: List[Tuple[str, int]] = []
        for n in txts:
            try:
                info = self._zip.getinfo(n)
            except KeyError:
                continue
            # file_size is uncompressed size
            if info.file_size >= max(min_bytes_per_book, self.seq_len + 1):
                usable.append((n, int(info.file_size)))

        if not usable:
            raise RuntimeError(
                f"No PG-19 books large enough for seq_len={self.seq_len} in split '{self.split}'."
            )

        self._books = usable

    def __len__(self) -> int:
        return self.num_samples

    def _sample_book_and_offset(self, *, idx: int) -> Tuple[str, int]:
        rng = np.random.default_rng(self.seed + idx)
        name, size = self._books[int(rng.integers(0, len(self._books)))]
        max_off = max(0, size - (self.seq_len + 1))
        off = int(rng.integers(0, max_off + 1)) if max_off > 0 else 0
        return name, off

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        name, off = self._sample_book_and_offset(idx=idx)
        with self._zip.open(name, "r") as f:
            data = f.read()  # bytes

        if len(data) < self.seq_len + 1:
            raise RuntimeError(f"Book too short in zip: {name}")

        # Slice after reading (ZipExtFile does not support seek efficiently).
        if off + self.seq_len + 1 <= len(data):
            chunk = data[off : off + self.seq_len + 1]
        else:
            # fallback: start
            chunk = data[: self.seq_len + 1]

        arr = np.frombuffer(chunk, dtype=np.uint8).astype(np.int64, copy=False)
        toks = arr + 1

        x = torch.tensor(toks[: self.seq_len], dtype=torch.long)
        y = torch.tensor(toks[1 : self.seq_len + 1], dtype=torch.long)
        attention_mask = torch.ones_like(x, dtype=torch.long)
        return {"input_ids": x, "labels": y, "attention_mask": attention_mask}

    def __del__(self):
        try:
            self._zip.close()
        except Exception:
            pass


class PG19ByteLMDataset(Dataset):
    """PG-19 language modeling dataset (byte-level).

    This is an *official-split* PG-19 loader, but uses a byte-level modeling
    scheme so it works with the existing integer-token models without requiring
    external tokenizers.

    Tokens:
    - PAD = 0 (not used in samples)
    - byte b in [0..255] is mapped to token id (b + 1) in [1..256]
    - vocab_size = 257

    Each sample is a random contiguous chunk from a random book.
    """

    PAD_TOKEN = 0

    def __init__(
        self,
        *,
        root: Path,
        split: str,
        seq_len: int,
        num_samples: int,
        seed: int,
        min_bytes_per_book: int = 4096,
    ):
        self.root = root.expanduser().resolve()
        self.split = split.lower()
        self.seq_len = int(seq_len)
        self.num_samples = int(num_samples)
        self.seed = int(seed)

        split_dir = _resolve_split_dir(self.root, self.split)
        files = sorted(split_dir.glob("*.txt"))
        if not files:
            raise FileNotFoundError(f"No .txt files found for PG-19 split '{split}' in {split_dir}")

        # Filter to books large enough to sample seq_len+1 bytes.
        usable: List[Tuple[Path, int]] = []
        for p in files:
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if size >= max(min_bytes_per_book, self.seq_len + 1):
                usable.append((p, int(size)))

        if not usable:
            raise RuntimeError(
                f"No PG-19 books large enough for seq_len={self.seq_len}. "
                f"Checked {len(files)} files under {split_dir}."
            )

        self._books = usable
        self.total_vocab_size = 257
        self._rng = np.random.default_rng(self.seed)

    def __len__(self) -> int:
        return self.num_samples

    def _sample_book_and_offset(self, *, idx: int) -> Tuple[Path, int]:
        # Deterministic per index for reproducibility.
        rng = np.random.default_rng(self.seed + idx)
        book_path, size = self._books[int(rng.integers(0, len(self._books)))]
        max_off = max(0, size - (self.seq_len + 1))
        off = int(rng.integers(0, max_off + 1)) if max_off > 0 else 0
        return book_path, off

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        book_path, off = self._sample_book_and_offset(idx=idx)

        with book_path.open("rb") as f:
            f.seek(off)
            chunk = f.read(self.seq_len + 1)

        # If we somehow hit EOF, fall back to start.
        if len(chunk) < self.seq_len + 1:
            with book_path.open("rb") as f:
                f.seek(0)
                chunk = f.read(self.seq_len + 1)

        if len(chunk) < self.seq_len + 1:
            raise RuntimeError(f"Book too short after fallback: {book_path}")

        arr = np.frombuffer(chunk, dtype=np.uint8).astype(np.int64, copy=False)
        # byte -> token_id in [1..256]
        toks = arr + 1

        x = torch.tensor(toks[: self.seq_len], dtype=torch.long)
        y = torch.tensor(toks[1 : self.seq_len + 1], dtype=torch.long)
        attention_mask = torch.ones_like(x, dtype=torch.long)

        return {"input_ids": x, "labels": y, "attention_mask": attention_mask}
