from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PathfinderGenConfig:
    resolution: int = 32
    # Controls the length/complexity of the main path.
    contour_length: int = 14
    # Number of distractor paths (hardness knob).
    distractor_paths: int = 20
    # Max steps for each distractor path.
    distractor_length: int = 5
    # Thickness in pixels for drawing.
    thickness: int = 1
    # Whether to add low-amplitude random noise.
    add_noise: bool = True
    # Noise intensity in [0, 255].
    noise_max: int = 10


def _clip_int(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else hi if x > hi else x


def _random_walk_path(
    rng: np.random.Generator,
    *,
    size: int,
    steps: int,
    avoid: Optional[np.ndarray] = None,
    max_tries: int = 200,
) -> Optional[List[Tuple[int, int]]]:
    """Generate a self-avoiding-ish random walk path on a size×size grid."""

    if steps < 2:
        return None

    for _ in range(max_tries):
        y = int(rng.integers(2, size - 2))
        x = int(rng.integers(2, size - 2))
        path = [(y, x)]
        visited = {path[0]}

        ok = True
        for _k in range(steps - 1):
            # 4-neighborhood
            neighbors = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
            rng.shuffle(neighbors)

            chosen = None
            for ny, nx in neighbors:
                if ny <= 1 or ny >= size - 2 or nx <= 1 or nx >= size - 2:
                    continue
                if (ny, nx) in visited:
                    continue
                if avoid is not None and avoid[ny, nx]:
                    continue
                chosen = (ny, nx)
                break

            if chosen is None:
                ok = False
                break

            y, x = chosen
            path.append((y, x))
            visited.add((y, x))

        if ok:
            return path

    return None


def _draw_thick(points: List[Tuple[int, int]], *, size: int, thickness: int) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.uint8)
    t = max(1, int(thickness))
    r = t // 2
    for y, x in points:
        y0 = _clip_int(y - r, 0, size - 1)
        y1 = _clip_int(y + r, 0, size - 1)
        x0 = _clip_int(x - r, 0, size - 1)
        x1 = _clip_int(x + r, 0, size - 1)
        img[y0 : y1 + 1, x0 : x1 + 1] = 255
    return img


def _place_marker(img: np.ndarray, y: int, x: int, *, radius: int = 1) -> None:
    size = img.shape[0]
    rr = max(1, int(radius))
    for dy in range(-rr, rr + 1):
        for dx in range(-rr, rr + 1):
            if dy * dy + dx * dx > rr * rr:
                continue
            yy = y + dy
            xx = x + dx
            if 0 <= yy < size and 0 <= xx < size:
                img[yy, xx] = 255


def generate_pathfinder32_hard_sample(
    rng: np.random.Generator,
    *,
    cfg: PathfinderGenConfig,
) -> Tuple[np.ndarray, int]:
    """Generate a Pathfinder-like 32×32 sample.

    Returns: (image_uint8, label)
      label=1 if start/end markers are connected by the same target path.

    This is a *local proxy* for Pathfinder "hard" (curv_contour_length_14).
    """

    size = int(cfg.resolution)

    # We generate balanced-ish classes by sampling label uniformly.
    label = int(rng.integers(0, 2))

    # Target path length: scale contour_length to steps.
    # Keep within grid; longer than typical short-range.
    steps = int(max(8, cfg.contour_length * 3))

    # Build image from target + distractors.
    avoid = np.zeros((size, size), dtype=bool)

    target = _random_walk_path(rng, size=size, steps=steps, avoid=None)
    if target is None:
        # fallback: trivial line
        target = [(size // 2, i) for i in range(2, size - 2)]

    target_img = _draw_thick(target, size=size, thickness=cfg.thickness)
    avoid |= target_img > 0

    # For negative examples, generate a second independent path and put end marker on it.
    other = None
    if label == 0:
        other = _random_walk_path(rng, size=size, steps=steps, avoid=avoid)
        if other is None:
            other = [(i, size // 2) for i in range(2, size - 2)]
        other_img = _draw_thick(other, size=size, thickness=cfg.thickness)
    else:
        other_img = np.zeros_like(target_img)

    img = np.maximum(target_img, other_img)
    avoid |= other_img > 0

    # Add distractor paths (shorter), avoiding markers/path overlap as much as possible.
    for _ in range(int(cfg.distractor_paths)):
        d_steps = int(max(3, cfg.distractor_length))
        d_path = _random_walk_path(rng, size=size, steps=d_steps, avoid=None)
        if not d_path:
            continue
        d_img = _draw_thick(d_path, size=size, thickness=max(1, cfg.thickness))
        img = np.maximum(img, d_img)

    # Choose start/end markers.
    start = target[0]
    if label == 1:
        end = target[-1]
    else:
        assert other is not None
        end = other[-1]

    _place_marker(img, start[0], start[1], radius=1)
    _place_marker(img, end[0], end[1], radius=1)

    if cfg.add_noise and cfg.noise_max > 0:
        noise = rng.integers(0, int(cfg.noise_max) + 1, size=(size, size), dtype=np.uint8)
        img = np.maximum(img, noise)

    return img.astype(np.uint8, copy=False), label


def generate_splits_to_npz(
    out_dir: Path,
    *,
    cfg: PathfinderGenConfig,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int = 0,
) -> None:
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    def _gen(n: int, seed_off: int) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed + seed_off)
        xs = np.zeros((n, cfg.resolution, cfg.resolution, 1), dtype=np.uint8)
        ys = np.zeros((n,), dtype=np.int64)
        for i in range(n):
            img, y = generate_pathfinder32_hard_sample(rng, cfg=cfg)
            xs[i, :, :, 0] = img
            ys[i] = y
        return xs, ys

    x_tr, y_tr = _gen(int(n_train), 0)
    x_va, y_va = _gen(int(n_val), 1_000_000)
    x_te, y_te = _gen(int(n_test), 2_000_000)

    # Naming matches our loader search patterns (variant + difficulty + split).
    np.savez_compressed(out_dir / "pathfinder32_hard_train.npz", inputs=x_tr, labels=y_tr)
    np.savez_compressed(out_dir / "pathfinder32_hard_val.npz", inputs=x_va, labels=y_va)
    np.savez_compressed(out_dir / "pathfinder32_hard_test.npz", inputs=x_te, labels=y_te)


def ensure_pathfinder32_hard_generated(
    out_dir: Path,
    *,
    cfg: Optional[PathfinderGenConfig] = None,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int = 0,
    force: bool = False,
) -> Path:
    """Ensure cached npz splits exist; generate them if missing."""

    out_dir = out_dir.expanduser().resolve()
    train_p = out_dir / "pathfinder32_hard_train.npz"
    val_p = out_dir / "pathfinder32_hard_val.npz"
    test_p = out_dir / "pathfinder32_hard_test.npz"

    if not force and train_p.exists() and val_p.exists() and test_p.exists():
        return out_dir

    generate_splits_to_npz(
        out_dir,
        cfg=cfg or PathfinderGenConfig(),
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        seed=seed,
    )
    return out_dir
