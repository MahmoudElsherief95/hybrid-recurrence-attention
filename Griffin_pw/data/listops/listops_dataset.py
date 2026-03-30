"""
Listops Dataset Implementation
Long Range Arena benchmark task for hierarchical structure understanding
This dataset is particularly challenging for attention-based models
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from typing import List, Tuple, Dict, Optional
import re


class ListopsDataset(Dataset):
    """
    Listops Dataset from Long Range Arena benchmark.

    This dataset tests hierarchical structure understanding through mathematical
    expressions with nested operations (MIN, MAX, SUM mod 10). It's extremely
    challenging for attention-based models due to the tree structure and
    long-range dependencies.

    Format: Nested expressions like "[MIN 3 4 [MAX 1 2]]" -> evaluate to result
    """

    def __init__(
        self,
        num_sequences: int = 10000,
        max_length: int = 256,
        max_depth: int = 3,  # capped: depth 10 + width 5 exceeds 256 tokens
        vocab_size: int = 15,  # 0-9 digits + operations + brackets
        seed: Optional[int] = 42
    ):
        """
        Initialize Listops dataset.

        Args:
            num_sequences: Number of sequences to generate
            max_length: Maximum sequence length
            max_depth: Maximum nesting depth
            vocab_size: Size of vocabulary
            seed: Random seed for reproducibility
        """
        self.num_sequences = num_sequences
        self.max_length = max_length
        self.max_depth = max_depth
        self.vocab_size = vocab_size

        # Special tokens
        self.PAD_TOKEN = 0
        self.OPEN_BRACKET = 11
        self.CLOSE_BRACKET = 12
        self.MIN_OP = 13
        self.MAX_OP = 14
        self.SUM_OP = 15
        self.SM_OP = 16  # SUM modulo 10

        # Create token mappings
        self.token_to_idx = {
            '(': self.OPEN_BRACKET,
            ')': self.CLOSE_BRACKET,
            'MIN': self.MIN_OP,
            'MAX': self.MAX_OP,
            'SUM': self.SUM_OP,
            'SM': self.SM_OP,
            'PAD': self.PAD_TOKEN
        }
        # Add digits 0-9
        for i in range(10):
            self.token_to_idx[str(i)] = i + 1

        self.idx_to_token = {v: k for k, v in self.token_to_idx.items()}
        self.total_vocab_size = len(self.token_to_idx)

        # Set random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Generate dataset
        self.sequences = []
        self.labels = []
        self._generate_dataset()

    def _generate_expression(self, depth: int = 0) -> Tuple[str, int]:
        """
        Recursively generate a Listops expression and its result.

        Args:
            depth: Current nesting depth

        Returns:
            expression: String representation
            result: Integer result of the expression
        """
        if depth >= self.max_depth or random.random() < 0.1:  # Leaf node
            value = random.randint(0, 9)
            return str(value), value

        # Choose operation
        operations = ['MIN', 'MAX', 'SUM', 'SM']
        op = random.choice(operations)

        # Generate operands (2-5 operands for variety)
        num_operands = random.randint(2, 5)
        operands = []
        operand_strs = []

        for _ in range(num_operands):
            expr_str, expr_val = self._generate_expression(depth + 1)
            operand_strs.append(expr_str)
            operands.append(expr_val)

        # Compute result based on operation
        if op == 'MIN':
            result = min(operands)
        elif op == 'MAX':
            result = max(operands)
        elif op == 'SUM':
            result = sum(operands)
        elif op == 'SM':
            result = sum(operands) % 10

        # Build expression string
        expr_parts = [op] + operand_strs
        expression = f"({' '.join(expr_parts)})"

        return expression, result

    def _tokenize_expression(self, expression: str) -> List[int]:
        """Tokenize expression string into token indices."""
        # Split by spaces and brackets
        tokens = re.findall(r'\(|\)|\w+', expression)
        token_indices = []

        for token in tokens:
            if token in self.token_to_idx:
                token_indices.append(self.token_to_idx[token])
            else:
                # Unknown token, skip or handle
                continue

        return token_indices

    def _generate_dataset(self):
        """Generate the complete dataset."""
        generated = 0
        attempts = 0
        max_attempts = self.num_sequences * 10  # Allow more attempts
        
        while generated < self.num_sequences and attempts < max_attempts:
            attempts += 1
            try:
                expression, result = self._generate_expression()
                token_indices = self._tokenize_expression(expression)
                
                # Truncate if too long instead of skipping
                if len(token_indices) > self.max_length:
                    token_indices = token_indices[:self.max_length]
                
                # Pad sequence
                padded_sequence = token_indices + [self.PAD_TOKEN] * (self.max_length - len(token_indices))
                
                # Target: LM-style — same as input shifted left by 1 (next-token prediction).
                # The model learns to predict the closing ')' and the digit result token.
                target_result = (result % 10) + 1  # digit 1-10
                # Replace last real token with the result digit (marks end-of-expression)
                last_tok = len(token_indices) - 1
                result_indices = list(token_indices)
                if last_tok >= 0:
                    result_indices[last_tok] = target_result
                padded_result = result_indices + [self.PAD_TOKEN] * (self.max_length - len(result_indices))
                target_sequence = padded_result[:self.max_length]
                
                self.sequences.append(padded_sequence)
                self.labels.append(target_sequence)
                generated += 1
                
            except Exception as e:
                # Skip failed generations
                continue

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        label = self.labels[idx]

        input_ids = torch.tensor(sequence, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)
        attention_mask = (input_ids != self.PAD_TOKEN).long()

        return {
            'input_ids': input_ids,
            'labels': label_tensor,
            'attention_mask': attention_mask
        }

    def create_dataloader(self, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)


def create_listops_datasets(
    num_train: int = 50000,
    num_val: int = 10000,
    max_length: int = 1024,
    max_depth: int = 10,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for Listops task.

    Args:
        num_train: Number of training sequences
        num_val: Number of validation sequences
        max_length: Maximum sequence length
        max_depth: Maximum nesting depth
        batch_size: Batch size for dataloaders

    Returns:
        train_dataloader, val_dataloader
    """
    train_dataset = ListopsDataset(
        num_sequences=num_train,
        max_length=max_length,
        max_depth=max_depth,
        seed=42
    )
    val_dataset = ListopsDataset(
        num_sequences=num_val,
        max_length=max_length,
        max_depth=max_depth,
        seed=123
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader