"""Synthetic Path-X (Pathfinder-inspired) generator.

This implementation is intentionally a *proxy* task: it generates a 1D token
sequence with a long-range dependency that resembles the Pathfinder/Path-X
connectivity problem.

If you want the official upstream data used in Long Range Arena (LRA), use the
loader in `Griffin_pw/data/official/pathfinder_lra_dataset.py` and run the
benchmark runner with `--pathx-source lra`.
"""

import torch
import random
from typing import List, Dict, Tuple
import numpy as np

class PathXDataset:
    """Simplified Path-X dataset for sequence models."""

    def __init__(self, num_samples: int = 1000, seq_length: int = 128, vocab_size: int = 50, difficulty: int = 1, seed: int | None = None):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.difficulty = int(difficulty)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Special tokens
        self.start_token = vocab_size - 4
        self.end_token = vocab_size - 3
        self.query_token = vocab_size - 2
        self.separator = vocab_size - 1

    def generate_path_sample(self) -> Dict:
        """
        Generate a path connectivity sample with balanced classes (50/50).

        Connected samples: a chain of waypoints actually bridges pos1 -> pos2.
        Disconnected samples: a short decoy path starts from pos1 but stops
                              well before pos2, so no chain reaches pos2.

        The long-range dependency (pos1 near the start, pos2 near the end) is
        what makes this hard for Local Attention (window=64) but tractable for
        recurrent models.
        """
        # Two distant anchor positions
        pos1 = random.randint(5, self.seq_length // 4)
        pos2 = random.randint(3 * self.seq_length // 4, self.seq_length - 10)

        is_connected = random.random() < 0.5

        path_pixels = []
        if is_connected:
            # Build a chain of waypoints that strictly advances from pos1 to pos2.
            # Use ~8-16 hops so the gap (≥128 tokens) is always covered.
            # Increase hops with difficulty to reduce trivial shortcuts.
            num_hops = random.randint(8, 16) if self.difficulty <= 1 else random.randint(16, 32)
            # Evenly spaced waypoints with small jitter
            step = (pos2 - pos1) / (num_hops + 1)
            current_pos = pos1
            for k in range(1, num_hops + 1):
                target_pos = int(pos1 + k * step)
                jitter = random.randint(-2, 2)
                next_pos = max(current_pos + 1,
                               min(pos2 - 1, target_pos + jitter))
                if next_pos != current_pos:
                    path_pixels.append(next_pos)
                    current_pos = next_pos
            # Make sure the chain terminates at pos2
            path_pixels.append(pos2)
        else:
            # Disconnected: short decoy path from pos1, but stops before
            # getting within 30 tokens of pos2 (never reaches it).
            stop_limit = max(pos1 + 5,
                             min(pos1 + (pos2 - pos1) // 3,
                                 pos2 - 30))
            current_pos = pos1
            for _ in range(random.randint(3, 8)):
                next_pos = min(current_pos + random.randint(1, 5), stop_limit)
                if next_pos != current_pos:
                    path_pixels.append(next_pos)
                    current_pos = next_pos
                if current_pos >= stop_limit:
                    break

            # Hard mode: add a few far-away "decoy" pixels that increase noise
            # without accidentally creating a full chain to pos2.
            if self.difficulty >= 2:
                for _ in range(random.randint(3, 8)):
                    dp = random.randint(0, self.seq_length - 1)
                    if dp not in (pos1, pos2):
                        path_pixels.append(dp)

        # Build the flat sequence
        sequence = [0] * self.seq_length
        sequence[pos1] = 1   # start pixel
        sequence[pos2] = 2   # end pixel
        for pos in path_pixels:
            if 0 <= pos < self.seq_length and pos not in (pos1, pos2):
                sequence[pos] = 3   # path pixel

        # Add distractor pixels (noise)
        if self.difficulty <= 1:
            num_distractors = random.randint(5, 15)
        else:
            num_distractors = random.randint(20, 60)
        for _ in range(num_distractors):
            dp = random.randint(0, self.seq_length - 1)
            if sequence[dp] == 0:
                sequence[dp] = 4   # distractor

        # Convert to token sequence
        token_sequence = [self.start_token]
        for i, pixel in enumerate(sequence):
            if pixel > 0:
                token_sequence.extend([pixel + 10, i])  # pixel_type_token + position

        # Always include query at the end.
        query_suffix = [self.query_token, pos1, pos2]
        token_sequence.extend(query_suffix)

        target = [1 if is_connected else 0, self.end_token]

        # Ensure the query suffix is preserved even when the token sequence is long.
        if len(token_sequence) > self.seq_length:
            # Keep as much of the prefix as fits, but always keep the last 3 query tokens.
            keep = self.seq_length - len(query_suffix)
            token_sequence = token_sequence[:keep] + query_suffix

        # Pad sequences
        input_padded = token_sequence + [0] * max(0, self.seq_length - len(token_sequence))
        target_padded = target + [0] * max(0, self.seq_length - len(target))

        return {
            "input_ids": input_padded[:self.seq_length],
            "target_ids": target_padded[:self.seq_length],
            "start_pos": pos1,
            "end_pos": pos2,
            "path_pixels": path_pixels,
            "is_connected": is_connected,
            "connection_type": "path_connected" if is_connected else "not_connected"
        }

    def generate_dataset(self) -> List[Dict]:
        """Generate the full dataset."""
        return [self.generate_path_sample() for _ in range(self.num_samples)]