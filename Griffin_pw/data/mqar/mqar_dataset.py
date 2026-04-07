"""Multi-Query Associative Recall (MQAR) dataset.

This implementation is adapted to match the common MQAR benchmark semantics
used in Zoology (HazyResearch):

- LM-style alignment: the model predicts a value token immediately after a
    query key appears.
- Answer-only supervision: labels are `IGNORE_INDEX` everywhere except at the
    positions where the value should be predicted.
- Long-range gaps: query positions are sampled from a power-law-like gap
    distribution.
- No padding shortcut: by default, non-query placeholders are replaced with
    random tokens so there is no large constant PAD region.

The dataset returns tensors shaped `[seq_len]` for both `input_ids` and
`labels`, with labels using the standard `ignore_index=-100` convention.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class MQARDataset(Dataset):
    """
    Multi-Query Associative Recall (MQAR) Dataset.
    
    This dataset tests a model's ability to recall information based on queries.
    It consists of sequences with key-value pairs followed by queries that need
    to be answered based on the stored information.
    
    Format: [key1, value1, key2, value2, ..., query1, answer1, query2, answer2, ...]
    """
    
    def __init__(
        self,
        num_sequences: int = 10000,
        seq_len: int = 512,
        vocab_size: int = 1000,
        num_kv_pairs: int = 8,
        num_passes: int = 1,
        power_a: float = 0.01,
        random_non_queries: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Initialize MQAR dataset.
        
        Args:
            num_sequences: Number of sequences to generate
            seq_len: Maximum sequence length
            vocab_size: Size of vocabulary (excluding special tokens)
            num_kv_pairs: Number of key-value pairs to store
            num_queries: Number of queries to test
            noise_ratio: Ratio of noise tokens between kv pairs
            gap_tokens: Number of filler tokens inserted between KV section and queries.
                        Set to > local_window (e.g. 150) to force long-range recall.
                        Models that rely on attention alone fail when gap > local_window.
            seed: Random seed for reproducibility
        """
        self.num_sequences = int(num_sequences)
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.num_kv_pairs = int(num_kv_pairs)
        self.num_passes = int(num_passes)
        self.power_a = float(power_a)
        self.random_non_queries = bool(random_non_queries)

        # Standard tokens
        self.PAD_TOKEN = 0
        self.IGNORE_INDEX = -100

        # Ensure vocabulary is large enough for disjoint key/value pools.
        if self.vocab_size <= self.seq_len:
            self.vocab_size = int(self.seq_len * 2 + 1)
        self.total_vocab_size = int(self.vocab_size)

        # RNG
        # Do NOT reset global RNGs to a constant seed by default; that can make
        # train/val identical when wrappers construct two datasets back-to-back.
        # Instead, keep a per-dataset Generator for reproducible generation.
        self._seed = None if seed is None else int(seed)
        self._rng = np.random.default_rng(self._seed)
        if self._seed is not None:
            torch.manual_seed(self._seed)
        
        # Generate dataset
        self.sequences: List[List[int]] = []
        self.labels: List[List[int]] = []
        self._generate_dataset()
    
    def _generate_dataset(self):
        """Generate the complete dataset."""
        for _ in range(self.num_sequences):
            # Use a stable per-sample seed derived from the dataset RNG.
            sample_seed = int(self._rng.integers(0, 2**32 - 1))
            sequence, labels = self._generate_sequence(sample_seed=sample_seed)
            self.sequences.append(sequence)
            self.labels.append(labels)
    
    def _generate_sequence(self, *, sample_seed: int) -> Tuple[List[int], List[int]]:
        """Generate a single MQAR example (inputs, labels).

        This is a multi-query associative recall construction:
        - The prefix stores key/value pairs (possibly repeated via num_passes).
        - The suffix contains sparse query keys placed at power-law gaps.
        - The model must predict the associated value token immediately after
          seeing the query key.

        Returns:
            sequence: input_ids of length seq_len
            labels: length seq_len with IGNORE_INDEX except supervised positions
        """

        T = int(self.seq_len)
        K = int(self.num_kv_pairs)
        P = int(max(1, self.num_passes))

        context_size = K * 2 * P
        # We need at least 2 tokens per query (key + value slot).
        if context_size + 2 * K > T:
            raise ValueError(
                f"MQAR config too large for seq_len={T}: context={context_size}, queries={2*K}"
            )

        # Split vocab into disjoint key/value pools.
        key_hi = self.vocab_size // 2
        rng = np.random.default_rng(int(sample_seed))
        keys = rng.choice(np.arange(1, key_hi, dtype=np.int64), size=K, replace=False)
        values = rng.choice(np.arange(key_hi, self.vocab_size, dtype=np.int64), size=K, replace=False)

        # Build KV prefix, repeating pairs for num_passes.
        kvs = np.zeros((context_size,), dtype=np.int64)
        kvs[0::2] = np.tile(keys, P)
        kvs[1::2] = np.tile(values, P)

        tail_len = T - context_size
        # We place queries at even offsets so the answer is the next token.
        space = tail_len // 2

        a = float(self.power_a)
        x = np.arange(1, space + 1, dtype=np.float64)
        p = a * np.power(x, a - 1.0)
        p = p / p.sum() if p.sum() > 0 else np.full_like(p, 1.0 / len(p))

        bins = rng.choice(np.arange(space, dtype=np.int64), size=K, replace=False, p=p)
        query_offsets = bins * 2

        tail = np.zeros((tail_len,), dtype=np.int64)
        tail[query_offsets] = keys

        full = np.concatenate([kvs, tail], axis=0)
        assert full.shape[0] == T

        # Labels use IGNORE_INDEX everywhere, except at the position of the
        # query key (LM-style alignment predicts the next token).
        labels_plus = np.full((T + 1,), self.IGNORE_INDEX, dtype=np.int64)
        labels_plus[context_size + query_offsets + 1] = values

        full_plus = np.concatenate([full, np.array([self.PAD_TOKEN], dtype=np.int64)], axis=0)
        inputs = full_plus[:-1]
        labels = labels_plus[1:]

        if self.random_non_queries:
            zeros = inputs == 0
            if np.any(zeros):
                inputs[zeros] = rng.integers(1, self.vocab_size, size=int(zeros.sum()), dtype=np.int64)

        return inputs.tolist(), labels.tolist()
    
    def __len__(self) -> int:
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sequence.
        
        Args:
            idx: Index of sequence
            
        Returns:
            Dictionary with input_ids and labels
        """
        sequence = self.sequences[idx]
        labels = self.labels[idx]
        
        # Convert to tensors
        input_ids = torch.tensor(sequence, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        # MQAR is generated without padding regions (and may contain zeros only
        # when random_non_queries=False). Keep the mask all-ones to avoid
        # accidental masking of valid tokens.
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'labels': labels_tensor,
            'attention_mask': attention_mask
        }
    
    def get_vocab_size(self) -> int:
        """Get total vocabulary size including special tokens."""
        return self.total_vocab_size
    
    def create_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> DataLoader:
        """
        Create a DataLoader for this dataset.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            
        Returns:
            DataLoader instance
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def evaluate_recall_accuracy(
        self,
        model,
        device: str = 'cuda',
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Evaluate model's recall accuracy on MQAR task.
        
        Args:
            model: Trained model to evaluate
            device: Device to run evaluation on
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with accuracy metrics
        """
        model.eval()
        dataloader = self.create_dataloader(batch_size=batch_size, shuffle=False)
        
        total_queries = 0
        correct_queries = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Forward pass
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                
                # Supervised positions are exactly those with labels != IGNORE_INDEX.
                supervised = labels != self.IGNORE_INDEX
                for i in range(input_ids.shape[0]):
                    for pos in torch.where(supervised[i])[0].tolist():
                        predicted = int(torch.argmax(logits[i, pos, :]).item())
                        true_token = int(labels[i, pos].item())
                        total_queries += 1
                        if predicted == true_token:
                            correct_queries += 1
        
        accuracy = correct_queries / total_queries if total_queries > 0 else 0.0
        
        return {
            'recall_accuracy': accuracy,
            'total_queries': total_queries,
            'correct_queries': correct_queries
        }


def create_mqar_datasets(
    train_size: int = 8000,
    val_size: int = 1000,
    test_size: int = 1000,
    **kwargs
) -> Tuple[MQARDataset, MQARDataset, MQARDataset]:
    """
    Create train, validation, and test MQAR datasets.
    
    Args:
        train_size: Size of training set
        val_size: Size of validation set
        test_size: Size of test set
        **kwargs: Additional arguments for MQARDataset
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Create datasets with different seeds for diversity
    train_dataset = MQARDataset(num_sequences=train_size, seed=42, **kwargs)
    val_dataset = MQARDataset(num_sequences=val_size, seed=123, **kwargs)
    test_dataset = MQARDataset(num_sequences=test_size, seed=456, **kwargs)
    
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # Example usage
    dataset = MQARDataset(
        num_sequences=100,
        seq_len=256,
        vocab_size=500,
        num_kv_pairs=8,
        num_queries=2
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Vocabulary size: {dataset.get_vocab_size()}")
    
    # Show a sample
    sample = dataset[0]
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    print(f"Attention mask shape: {sample['attention_mask'].shape}")
    
    # Show first few tokens
    print(f"First 20 input tokens: {sample['input_ids'][:20].tolist()}")
    print(f"First 20 label tokens: {sample['labels'][:20].tolist()}")
