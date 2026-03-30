"""
Sequential-NIAH Dataset
Tests ability to maintain multiple pieces of information over time.
Local Attention should struggle as earlier needles get "forgotten" when new content arrives.
"""

import torch
import random
from typing import List, Dict, Tuple
import json

class SequentialNIAHDataset:
    """Sequential NIAH - multiple needles that must be remembered over time."""

    def __init__(self, num_samples: int = 1000, seq_length: int = 256, vocab_size: int = 100):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        # Special tokens
        self.needle_start = vocab_size - 5
        self.needle_end = vocab_size - 4
        self.query_token = vocab_size - 3
        self.separator = vocab_size - 2
        self.end_token = vocab_size - 1

    def generate_sequential_niah_sample(self) -> Dict:
        """Generate a sequential NIAH sample with multiple needles."""
        # Create multiple needles that appear over time
        num_needles = random.randint(3, 6)
        needles = []

        # Generate needle content
        for i in range(num_needles):
            needle_length = random.randint(3, 8)
            needle_content = [random.randint(1, self.vocab_size - 6) for _ in range(needle_length)]
            needles.append({
                "id": i,
                "content": needle_content,
                "topic": f"topic_{i}",
                "importance": random.randint(1, 5)
            })

        # Build the sequence with needles appearing at different times
        sequence = []
        needle_positions = {}

        # Add initial context
        initial_context = [random.randint(1, self.vocab_size - 6) for _ in range(random.randint(10, 20))]
        sequence.extend(initial_context)
        sequence.append(self.separator)

        # Add needles at different positions
        for i, needle in enumerate(needles):
            # Add some distractor content before each needle
            distractor_length = random.randint(5, 15)
            distractors = [random.randint(1, self.vocab_size - 6) for _ in range(distractor_length)]
            sequence.extend(distractors)
            sequence.append(self.separator)

            # Record needle start position
            needle_start_pos = len(sequence)
            needle_positions[needle["id"]] = needle_start_pos

            # Add needle
            sequence.append(self.needle_start)
            sequence.extend(needle["content"])
            sequence.append(self.needle_end)
            sequence.append(self.separator)

        # Add final distractors
        final_distractors = [random.randint(1, self.vocab_size - 6) for _ in range(random.randint(10, 20))]
        sequence.extend(final_distractors)

        # Query: ask about a specific needle (not necessarily the most recent)
        query_needle_id = random.choice([n["id"] for n in needles[:-1]])  # Avoid most recent
        sequence.append(self.query_token)
        sequence.append(query_needle_id)

        # Target: the content of the queried needle
        target_needle = next(n for n in needles if n["id"] == query_needle_id)
        target = target_needle["content"] + [self.end_token]

        # Pad sequences
        input_padded = sequence + [0] * (self.seq_length - len(sequence))
        target_padded = target + [0] * (self.seq_length - len(target))

        return {
            "input_ids": input_padded[:self.seq_length],
            "target_ids": target_padded[:self.seq_length],
            "needles": needles,
            "query_needle_id": query_needle_id,
            "expected_answer": target_needle["content"],
            "needle_positions": needle_positions,
            "num_needles": num_needles,
            "sequence_length": len(sequence)
        }

    def generate_dataset(self) -> List[Dict]:
        """Generate the full dataset."""
        return [self.generate_sequential_niah_sample() for _ in range(self.num_samples)]