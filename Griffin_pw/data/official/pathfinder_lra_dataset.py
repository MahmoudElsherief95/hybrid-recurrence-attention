from __future__ import annotations

import gzip
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PathfinderLoaded:
    x: np.ndarray  # (N, ...)
    y: np.ndarray  # (N,)
    source_path: Path


def _normalize_variant_and_split(variant: str, difficulty: str) -> Tuple[str, str]:
    """Normalize user-friendly aliases to LRA task naming.

    LRA's image task registry uses task names like:
      pathfinder32_hard, pathfinder128_easy, ...

    In LRA tables/papers, "Path" is commonly Pathfinder-32 (hard) and
    "Path-X" corresponds to a longer Pathfinder setting, most commonly
    Pathfinder-128 (hard).
    """

    v = variant.strip().lower().replace("-", "").replace("_", "")
    d = difficulty.strip().lower()
    if d in ("inter", "intermediate"):
        d = "intermediate"
    if d not in ("easy", "intermediate", "hard"):
        raise ValueError(f"Unsupported difficulty split: {difficulty}")

    # Aliases
    if v in ("path", "pathfinder", "pathfinder32"):
        return "pathfinder32", d
    if v in ("pathx", "path-x", "path_x"):
        # Path-X in LRA refers to the longer pathfinder setting; 128 is the
        # common public benchmark setting.
        return "pathfinder128", d
    if v in ("pathfinder64", "pathfinder128", "pathfinder256"):
        return v, d

    raise ValueError(f"Unsupported variant: {variant}")


def _load_pickle(path: Path) -> Any:
    if path.suffixes[-1].lower() == ".gz":
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    with path.open("rb") as f:
        return pickle.load(f)


def _load_np(path: Path) -> Any:
    # NOTE: allow_pickle=True because some upstream dumps store tuples/dicts.
    if path.suffix.lower() == ".npz":
        return np.load(path, allow_pickle=True)
    return np.load(path, allow_pickle=True)


def _coerce_xy(obj: Any, *, path: Path) -> PathfinderLoaded:
    """Coerce a variety of common LRA dump formats into (x, y) ndarrays.

    Supported shapes:
    - dict-like with keys (x/inputs/images/data) and (y/labels/targets)
    - tuple/list of length 2: (x, y)
    - list of dicts: [{"input_ids"/"x"/"inputs": ..., "label"/"y"/"labels": ...}, ...]
    - np.lib.npyio.NpzFile with similar keys
    """

    # NpzFile behaves like a dict for keys/values.
    if hasattr(obj, "files") and hasattr(obj, "__getitem__"):
        keys = [str(k) for k in getattr(obj, "files", [])]
        d = {k: obj[k] for k in keys}
        obj = d

    x: Any = None
    y: Any = None

    if isinstance(obj, dict):
        for k in ("x", "inputs", "images", "data", "input_ids"):
            if k in obj:
                x = obj[k]
                break
        for k in ("y", "labels", "targets", "target", "label"):
            if k in obj:
                y = obj[k]
                break

        # Some dumps store nested dicts.
        if x is None and "train" in obj and isinstance(obj["train"], (dict, tuple, list)):
            return _coerce_xy(obj["train"], path=path)

    elif isinstance(obj, (tuple, list)) and len(obj) == 2:
        x, y = obj[0], obj[1]

    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
        xs: List[Any] = []
        ys: List[Any] = []
        for item in obj:
            xi = None
            yi = None
            for k in ("x", "inputs", "images", "data", "input_ids"):
                if k in item:
                    xi = item[k]
                    break
            for k in ("y", "labels", "targets", "label"):
                if k in item:
                    yi = item[k]
                    break
            if xi is None or yi is None:
                continue
            xs.append(xi)
            ys.append(yi)
        if xs and ys:
            x = np.stack(xs)
            y = np.asarray(ys)

    if x is None or y is None:
        raise ValueError(
            "Unsupported Pathfinder/Path-X dump format. "
            f"path={path} type={type(obj)} keys={list(obj.keys()) if isinstance(obj, dict) else None}"
        )

    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    # Labels sometimes come as one-hot.
    if y_arr.ndim > 1:
        y_arr = y_arr.argmax(axis=-1)

    # Ensure (N, ...) and (N,)
    if x_arr.ndim == 1:
        x_arr = x_arr[None, :]
    if y_arr.ndim == 0:
        y_arr = y_arr[None]

    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(f"Mismatched x/y lengths: x={x_arr.shape} y={y_arr.shape} path={path}")

    # Coerce integer tokens.
    if np.issubdtype(x_arr.dtype, np.floating):
        x_arr = np.rint(x_arr).astype(np.int64)
    else:
        x_arr = x_arr.astype(np.int64, copy=False)

    y_arr = y_arr.astype(np.int64, copy=False)

    return PathfinderLoaded(x=x_arr, y=y_arr, source_path=path)


def _find_split_file(root: Path, *, keyword: str, difficulty: str, split: str) -> Path:
    """Find a single file for (keyword, split) under root.

    Tries a handful of common upstream naming patterns and falls back to rglob.
    """

    split = split.lower()
    if split in ("valid", "validation"):
        split = "val"

    keyword_l = keyword.lower()
    difficulty_l = difficulty.lower()

    # Prefer filenames that mention both variant and difficulty.
    patterns = [
        f"*{keyword_l}*{difficulty_l}*{split}*.npz",
        f"*{keyword_l}*{difficulty_l}*{split}*.npy",
        f"*{keyword_l}*{difficulty_l}*{split}*.pkl",
        f"*{keyword_l}*{difficulty_l}*{split}*.pickle",
        f"*{keyword_l}*{difficulty_l}*{split}*.pkl.gz",
        f"*{keyword_l}*{difficulty_l}*{split}*.pickle.gz",
        # Back-compat: some dumps don't include difficulty in the filename.
        f"*{keyword_l}*{split}*.npz",
        f"*{keyword_l}*{split}*.npy",
        f"*{keyword_l}*{split}*.pkl",
        f"*{keyword_l}*{split}*.pickle",
        f"*{keyword_l}*{split}*.pkl.gz",
        f"*{keyword_l}*{split}*.pickle.gz",
    ]

    candidates: List[Path] = []
    for pat in patterns:
        candidates.extend(root.rglob(pat))

    # If those failed (e.g., split name appears before keyword), do a broader search.
    if not candidates:
        for ext in ("*.npz", "*.npy", "*.pkl", "*.pickle", "*.pkl.gz", "*.pickle.gz"):
            for p in root.rglob(ext):
                name = p.name.lower()
                if keyword_l in name and split in name:
                    candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            f"Could not find LRA Pathfinder/Path-X files for keyword='{keyword}' difficulty='{difficulty}' split='{split}' under {root}. "
            "Tip: run `python -m Griffin_pw.utils.inspect_lra_release` with LRA_RELEASE_ARCHIVE_PATH set to see filenames."
        )

    # Prefer shallowest paths to avoid picking cached copies.
    if split == "":
        # Prefer a base file that doesn't already look pre-split.
        def _looks_presplit(p: Path) -> bool:
            n = p.name.lower()
            return any(tok in n for tok in ("train", "val", "valid", "validation", "test"))

        candidates.sort(key=lambda p: (_looks_presplit(p), len(p.parts), len(p.name)))
    else:
        candidates.sort(key=lambda p: (len(p.parts), len(p.name)))
    return candidates[0]


def load_lra_pathfinder_split(official_root: Path, *, variant: str, difficulty: str, split: str) -> PathfinderLoaded:
    """Load Pathfinder variants from an extracted LRA release directory.

    This function is file-layout tolerant: it searches for a file containing
    both the variant keyword and the split name.

    Note: LRA's preprocessing is defined in `lra_benchmarks/image/input_pipeline.py`:
    grayscale pixels are cast to int and flattened for transformer models.
    """

    keyword, difficulty_key = _normalize_variant_and_split(variant, difficulty)

    official_root = official_root.expanduser().resolve()

    # The release archives seen in the wild sometimes use either "pathx" or
    # "pathfinder128" naming. If the user asked for Path-X, try both.
    keywords = [keyword]
    raw = variant.strip().lower().replace("-", "").replace("_", "")
    if raw in ("pathx", "path-x", "path_x") and keyword != "pathx":
        keywords = ["pathx", keyword]

    split_path: Optional[Path] = None
    last_err: Optional[Exception] = None
    for kw in keywords:
        try:
            split_path = _find_split_file(official_root, keyword=kw, difficulty=difficulty_key, split=split)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e

    if split_path is None:
        # Some releases store a single file per difficulty (e.g. "...hard...")
        # and rely on TFDS slicing to define train/val/test:
        #   train = hard[:80%]
        #   val   = hard[80%:90%]
        #   test  = hard[90%:]
        # Mirror that behavior here.
        split_l = split.lower()
        if split_l in ("train", "val", "valid", "validation", "test"):
            base_split = ""  # search without requiring split token
            base_path: Optional[Path] = None
            for kw in keywords:
                try:
                    base_path = _find_split_file(official_root, keyword=kw, difficulty=difficulty_key, split=base_split)
                    break
                except Exception as e:  # noqa: BLE001
                    last_err = e
            if base_path is None:
                assert last_err is not None
                raise last_err

            if base_path.suffix.lower() in (".pkl", ".pickle") or base_path.name.lower().endswith((".pkl.gz", ".pickle.gz")):
                obj = _load_pickle(base_path)
            else:
                obj = _load_np(base_path)
            loaded = _coerce_xy(obj, path=base_path)

            n = int(loaded.x.shape[0])
            i0 = 0
            i1 = int(0.8 * n)
            i2 = int(0.9 * n)
            if split_l == "train":
                xs, ys = loaded.x[i0:i1], loaded.y[i0:i1]
            elif split_l in ("val", "valid", "validation"):
                xs, ys = loaded.x[i1:i2], loaded.y[i1:i2]
            else:
                xs, ys = loaded.x[i2:n], loaded.y[i2:n]

            return PathfinderLoaded(x=np.asarray(xs), y=np.asarray(ys), source_path=base_path)

        assert last_err is not None
        raise last_err

    if split_path.suffix.lower() in (".pkl", ".pickle") or split_path.name.lower().endswith((".pkl.gz", ".pickle.gz")):
        obj = _load_pickle(split_path)
    else:
        obj = _load_np(split_path)

    return _coerce_xy(obj, path=split_path)


class LRAPathfinderClassificationDataset(Dataset):
    """Official-ish LRA Pathfinder/Path-X classification dataset loader.

    This expects the extracted `lra_release.gz` tree (or a manually provided folder).

    It returns:
      - input_ids: int64 tokens of length `seq_length` (flattened if image-like)
      - labels: int64 scalar (0/1)
      - attention_mask: 1 for non-pad tokens

    Notes:
    - For image-like arrays, we flatten per-sample.
    - If the official data's native sequence is longer than `seq_length`, we truncate.
    - If shorter, we right-pad with `pad_token`.
    """

    def __init__(
        self,
        *,
        split: str,
        seq_length: int,
        official_root: Path,
        variant: str = "pathx",
        difficulty: str = "hard",
        max_samples: Optional[int] = None,
        # IMPORTANT: pixel value 0 is a valid token (background).
        # Use a pad token that does NOT collide with pixel values.
        pad_token: int = 256,
        strict_seq_length: bool = True,
    ):
        variant_key, difficulty_key = _normalize_variant_and_split(variant, difficulty)
        loaded = load_lra_pathfinder_split(official_root, variant=variant_key, difficulty=difficulty_key, split=split)

        x = loaded.x
        y = loaded.y

        if max_samples is not None:
            n = int(max_samples)
            x = x[:n]
            y = y[:n]

        # LRA expects grayscale image inputs that are later flattened for
        # transformer models. We flatten here to expose 1D `input_ids`.
        # Common shapes:
        #  - (N, H, W)
        #  - (N, H, W, 1)
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)

        self.seq_length = int(seq_length)
        self.pad_token = int(pad_token)

        # If we can infer the native length, enforce apples-to-apples by default.
        native_len = int(x.shape[1]) if x.ndim == 2 else None
        if strict_seq_length and native_len is not None and native_len != self.seq_length:
            raise ValueError(
                "Official LRA Pathfinder inputs have a fixed length (flattened pixels). "
                f"Got seq_length={self.seq_length} but native_len={native_len} for variant={variant_key} difficulty={difficulty_key}. "
                "Set --pathx-seq-len to the native length (e.g., 1024 for 32x32, 16384 for 128x128) or pass strict_seq_length=False."
            )

        # LRA treats pixel values as categorical tokens (0..255) for transformer models.
        # Ensure vocab covers both pixel range and our pad token.
        max_tok = int(np.max(x)) if x.size else 0
        self.vocab_size = max(257, max_tok + 1, self.pad_token + 1)

        self.samples: List[Dict[str, torch.Tensor]] = []
        for i in range(x.shape[0]):
            row = x[i]
            ids = row.tolist()
            if len(ids) != self.seq_length:
                # Allow non-strict mode to truncate/pad; strict mode validated above.
                if len(ids) > self.seq_length:
                    ids = ids[: self.seq_length]
                else:
                    ids = ids + [self.pad_token] * (self.seq_length - len(ids))

            input_ids = torch.tensor(ids, dtype=torch.long)
            attention_mask = (input_ids != self.pad_token).long()

            label = int(y[i])
            # Most sources are binary; keep robust.
            if label not in (0, 1):
                label = 1 if label != 0 else 0

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
