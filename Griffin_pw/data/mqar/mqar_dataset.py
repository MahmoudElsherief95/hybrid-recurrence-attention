"""
Multi-Query Associative Recall (MQAR) Dataset Implementation
Dataset for testing associative memory capabilities of sequence models
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from typing import List, Tuple, Dict, Optional
import math


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
        num_kv_pairs: int = 16,
        num_queries: int = 4,
        noise_ratio: float = 0.1,
        seed: Optional[int] = 42
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
            seed: Random seed for reproducibility
        """
        self.num_sequences = num_sequences
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_kv_pairs = num_kv_pairs
        self.num_queries = num_queries
        self.noise_ratio = noise_ratio
        
        # Special tokens
        self.PAD_TOKEN = 0
        self.SEP_TOKEN = vocab_size + 1  # Separator between kv pairs and queries
        self.QUERY_TOKEN = vocab_size + 2  # Query marker
        self.ANSWER_TOKEN = vocab_size + 3  # Answer marker
        self.total_vocab_size = vocab_size + 4
        
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
        Generate a single MQAR sequence.
        
        Returns:
            sequence: Input sequence with kv pairs and queries
            labels: Target labels for each position
        """
        sequence = []
        labels = []
        
        # Generate key-value pairs
        kv_pairs = {}
        keys_used = set()
        
        for _ in range(self.num_kv_pairs):
            # Generate unique key
            while True:
                key = random.randint(1, self.vocab_size)
                if key not in keys_used:
                    keys_used.add(key)
                    break
            
            # Generate value
            value = random.randint(1, self.vocab_size)
            kv_pairs[key] = value
            
            # Add key-value pair to sequence
            sequence.extend([key, value])
            labels.extend([key, value])  # During storage, predict next token
            
            # Add noise tokens with some probability
            if random.random() < self.noise_ratio:
                noise_length = random.randint(1, 3)
                for _ in range(noise_length):
                    noise_token = random.randint(1, self.vocab_size)
                    sequence.append(noise_token)
                    labels.append(noise_token)
        
        # Add separator
        sequence.append(self.SEP_TOKEN)
        labels.append(self.SEP_TOKEN)
        
        # Generate queries
        query_keys = random.sample(list(keys_used), min(self.num_queries, len(keys_used)))
        
        for query_key in query_keys:
            # Add query marker and key
            sequence.extend([self.QUERY_TOKEN, query_key])
            labels.extend([self.QUERY_TOKEN, query_key])
            
            # Add answer marker and correct value
            correct_value = kv_pairs[query_key]
            sequence.extend([self.ANSWER_TOKEN, correct_value])
            labels.extend([self.ANSWER_TOKEN, correct_value])
        
        # Pad or truncate to desired length
        if len(sequence) < self.seq_len:
            # Pad with PAD_TOKEN
            pad_length = self.seq_len - len(sequence)
            sequence.extend([self.PAD_TOKEN] * pad_length)
            labels.extend([self.PAD_TOKEN] * pad_length)
        else:
            # Truncate
            sequence = sequence[:self.seq_len]
            labels = labels[:self.seq_len]
        
        return sequence, labels
    
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
        
        # Create attention mask (1 for non-padding tokens)
        attention_mask = (input_ids != self.PAD_TOKEN).long()
        
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
                
                # Find answer positions
                answer_positions = (input_ids == self.ANSWER_TOKEN)
                
                # Get predictions at answer positions
                for i in range(input_ids.shape[0]):
                    seq_answer_positions = torch.where(answer_positions[i])[0]
                    
                    for pos in seq_answer_positions:
                        if pos + 1 < input_ids.shape[1]:  # Ensure we have a next token
                            predicted_token = torch.argmax(logits[i, pos, :]).item()
                            true_token = labels[i, pos + 1].item()
                            
                            if true_token != self.PAD_TOKEN:  # Skip padding
                                total_queries += 1
                                if predicted_token == true_token:
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
