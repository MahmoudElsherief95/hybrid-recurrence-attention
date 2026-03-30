"""
Copy Task Dataset Implementation
Dataset for testing long-range memory and exact recall capabilities
This task is particularly challenging for attention-based models with long sequences
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from typing import List, Tuple, Dict, Optional


class CopyDataset(Dataset):
    """
    Copy Task Dataset.

    This dataset tests a model's ability to remember and exactly reproduce
    a sequence of tokens after a delimiter. It's particularly challenging
    for attention-based models as the required context length increases.

    Format: [sequence_to_copy, DELIMITER, padding, ...] -> [padding, sequence_to_copy, ...]
    """

    def __init__(
        self,
        num_sequences: int = 10000,
        seq_len: int = 512,
        vocab_size: int = 1000,
        copy_length: int = 100,  # > local_window=64 so LA cannot see prefix within its window
        seed: Optional[int] = 42
    ):
        """
        Initialize Copy dataset.

        Args:
            num_sequences: Number of sequences to generate
            seq_len: Maximum sequence length
            vocab_size: Size of vocabulary (excluding special tokens)
            copy_length: Length of the sequence to copy
            seed: Random seed for reproducibility
        """
        self.num_sequences = num_sequences
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.copy_length = copy_length

        # Special tokens
        self.PAD_TOKEN = 0
        self.DELIMITER = vocab_size + 1  # Marks end of input sequence
        self.total_vocab_size = vocab_size + 2

        # Set random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Generate dataset
        self.sequences = []
        self.labels = []
        self._generate_dataset()

    def _generate_dataset(self):
        """Generate the complete dataset."""
        for _ in range(self.num_sequences):
            sequence, labels = self._generate_sequence()
            self.sequences.append(sequence)
            self.labels.append(labels)

    def _generate_sequence(self) -> Tuple[List[int], List[int]]:
        """
        Generate a single copy task sequence.

        Returns:
            sequence: Input sequence [copy_seq, DELIMITER, PAD, ...]
            labels: Target labels [PAD, ..., copy_seq, ...]
        """
        # Generate random sequence to copy
        copy_seq = [random.randint(1, self.vocab_size) for _ in range(self.copy_length)]

        # Input: copy sequence + delimiter + padding
        input_seq = copy_seq + [self.DELIMITER] + [self.PAD_TOKEN] * (self.seq_len - self.copy_length - 1)

        # Target: padding + copy sequence + padding
        target_seq = [self.PAD_TOKEN] * (self.copy_length + 1) + copy_seq + [self.PAD_TOKEN] * (self.seq_len - 2 * self.copy_length - 1)

        # Truncate to seq_len
        input_seq = input_seq[:self.seq_len]
        target_seq = target_seq[:self.seq_len]

        return input_seq, target_seq

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


def create_copy_datasets(
    num_train: int = 50000,
    num_val: int = 10000,
    seq_len: int = 512,
    vocab_size: int = 1000,
    copy_length: int = 32,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for copy task.

    Args:
        num_train: Number of training sequences
        num_val: Number of validation sequences
        seq_len: Maximum sequence length
        vocab_size: Vocabulary size
        copy_length: Length of sequence to copy
        batch_size: Batch size for dataloaders

    Returns:
        train_dataloader, val_dataloader
    """
    train_dataset = CopyDataset(
        num_sequences=num_train,
        seq_len=seq_len,
        vocab_size=vocab_size,
        copy_length=copy_length,
        seed=42
    )
    val_dataset = CopyDataset(
        num_sequences=num_val,
        seq_len=seq_len,
        vocab_size=vocab_size,
        copy_length=copy_length,
        seed=123
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader