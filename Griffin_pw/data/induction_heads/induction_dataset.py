"""
Induction Heads Dataset
Tests the model's ability to learn and apply induction patterns.
This is particularly challenging for Local Attention as it requires
recognizing patterns that span beyond the local window.
"""

import torch
import random
from typing import List, Dict, Tuple

class InductionHeadsDataset:
    """Dataset for induction head tasks that test pattern recognition across distances."""

    def __init__(self, num_samples: int = 1000, seq_length: int = 100, vocab_size: int = 50):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        # Special tokens for induction patterns
        self.pattern_start = vocab_size - 2
        self.pattern_end = vocab_size - 1

    def generate_induction_sample(self) -> Dict:
        """Generate a sample that requires induction head capabilities."""
        # Create a sequence with repeated patterns
        pattern_length = random.randint(2, 4)
        num_repetitions = random.randint(3, 6)

        # Generate base pattern
        base_pattern = [random.randint(1, self.vocab_size - 3) for _ in range(pattern_length)]

        # Build sequence with repetitions and variations
        sequence = []
        pattern_positions = []

        for i in range(num_repetitions):
            if i > 0:
                # Add some distractor tokens between patterns
                distractor_length = random.randint(1, 3)
                distractors = [random.randint(1, self.vocab_size - 3) for _ in range(distractor_length)]
                sequence.extend(distractors)

            # Add pattern
            pattern_positions.append(len(sequence))
            sequence.extend(base_pattern)

            # Sometimes vary the pattern slightly (but maintain core structure)
            if random.random() < 0.3 and i > 0:
                varied_pattern = base_pattern.copy()
                # Change one element
                change_idx = random.randint(0, len(varied_pattern) - 1)
                varied_pattern[change_idx] = random.randint(1, self.vocab_size - 3)
                sequence.extend(varied_pattern)

        # Add induction query: partial pattern that should complete
        query_start = base_pattern[:random.randint(1, len(base_pattern) - 1)]
        sequence.extend(query_start)

        # Target: complete the pattern
        target = base_pattern[len(query_start):] + [0]  # Add end token

        # Pad sequences
        input_padded = sequence + [0] * (self.seq_length - len(sequence))
        target_padded = target + [0] * (self.seq_length - len(target))

        return {
            "input_ids": input_padded[:self.seq_length],
            "target_ids": target_padded[:self.seq_length],
            "base_pattern": base_pattern,
            "pattern_positions": pattern_positions,
            "num_repetitions": num_repetitions,
            "query_completion": target[:-1]  # Remove end token
        }

    def generate_dataset(self) -> List[Dict]:
        """Generate the full dataset."""
        return [self.generate_induction_sample() for _ in range(self.num_samples)]