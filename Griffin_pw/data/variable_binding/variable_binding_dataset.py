"""
Variable Binding Dataset
Tests the model's ability to track variable bindings across long sequences.
This requires maintaining associations between variables and values,
which is challenging for models with limited receptive fields.
"""

import torch
import random
from typing import List, Dict, Tuple

class VariableBindingDataset:
    """Dataset for variable binding tasks that test associative memory across distances."""

    def __init__(self, num_samples: int = 1000, seq_length: int = 100, vocab_size: int = 50):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        # Special tokens
        self.bind_token = vocab_size - 4    # Indicates variable binding
        self.query_token = vocab_size - 3   # Query for bound value
        self.separator = vocab_size - 2     # Separates sections
        self.end_token = vocab_size - 1     # End of sequence

    def generate_binding_sample(self) -> Dict:
        """Generate a variable binding sample."""
        # Create multiple variable bindings
        num_variables = random.randint(3, 6)
        variables = list(range(1, num_variables + 1))  # Variable IDs
        values = [random.randint(10, self.vocab_size - 5) for _ in range(num_variables)]

        # Create binding sequence
        bindings = list(zip(variables, values))
        random.shuffle(bindings)  # Randomize binding order

        sequence = []
        binding_positions = {}

        # Add variable bindings
        for var, val in bindings:
            sequence.extend([self.bind_token, var, val])
            binding_positions[var] = len(sequence) - 1  # Position of value

        # Add distractor operations
        num_distractors = random.randint(2, 4)
        for _ in range(num_distractors):
            # Add some operations that don't affect bindings
            op_length = random.randint(2, 4)
            operation = [random.randint(1, self.vocab_size - 5) for _ in range(op_length)]
            sequence.extend([self.separator] + operation)

        # Query: ask for value of a specific variable
        query_var = random.choice(variables)
        sequence.extend([self.query_token, query_var])

        # Target: the bound value
        target_value = values[variables.index(query_var)]
        target = [target_value, self.end_token]

        # Pad sequences
        input_padded = sequence + [0] * (self.seq_length - len(sequence))
        target_padded = target + [0] * (self.seq_length - len(target))

        return {
            "input_ids": input_padded[:self.seq_length],
            "target_ids": target_padded[:self.seq_length],
            "bindings": dict(zip(variables, values)),
            "query_variable": query_var,
            "expected_value": target_value,
            "binding_positions": binding_positions,
            "num_variables": num_variables
        }

    def generate_dataset(self) -> List[Dict]:
        """Generate the full dataset."""
        return [self.generate_binding_sample() for _ in range(self.num_samples)]