"""
Selective Copy Dataset
Tests ability to selectively copy specific elements while ignoring others.
This exposes Local Attention limitations when relevant information is separated by distractors.
"""

import torch
import random
from typing import List, Dict, Tuple

class SelectiveCopyDataset:
    """Dataset for selective copying tasks that challenge local attention."""

    def __init__(self, num_samples: int = 1000, seq_length: int = 100, vocab_size: int = 100):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        # Special tokens
        self.copy_token = vocab_size - 3  # Token that indicates what to copy
        self.separator = vocab_size - 2   # Separates sections
        self.query_token = vocab_size - 1 # Query token

    def generate_sample(self) -> Dict:
        """Generate a selective copy sample."""
        # Create a sequence with multiple items
        num_items = random.randint(3, 8)
        items = []
        positions = []

        # Generate random items
        for i in range(num_items):
            item_length = random.randint(2, 5)
            item = [random.randint(1, self.vocab_size - 4) for _ in range(item_length)]
            items.append(item)
            positions.append(len(positions) + sum(len(x) for x in items[:-1]))

        # Choose which item to copy (not the first or last to make it challenging)
        copy_index = random.randint(1, num_items - 2)

        # Build input sequence
        input_seq = []
        for i, item in enumerate(items):
            if i > 0:
                input_seq.append(self.separator)
            input_seq.extend(item)

        # Add copy instruction
        input_seq.append(self.copy_token)
        input_seq.append(copy_index)  # Which item to copy
        input_seq.append(self.query_token)

        # Target is the selected item
        target = items[copy_index] + [0]  # Add end token

        # Pad sequences
        input_padded = input_seq + [0] * (self.seq_length - len(input_seq))
        target_padded = target + [0] * (self.seq_length - len(target))

        return {
            "input_ids": input_padded[:self.seq_length],
            "target_ids": target_padded[:self.seq_length],
            "copy_index": copy_index,
            "num_items": num_items,
            "selected_item": items[copy_index]
        }

    def generate_dataset(self) -> List[Dict]:
        """Generate the full dataset."""
        return [self.generate_sample() for _ in range(self.num_samples)]