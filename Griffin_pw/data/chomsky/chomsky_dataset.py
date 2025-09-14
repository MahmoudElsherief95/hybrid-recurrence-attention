"""
Chomsky Hierarchy Dataset Implementation
Datasets for testing different levels of computational complexity
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from typing import List, Tuple, Dict, Optional, Union
import string


class ChomskyDataset(Dataset):
    """
    Base class for Chomsky Hierarchy datasets.
    Tests different levels of computational complexity.
    """
    
    def __init__(
        self,
        num_sequences: int = 10000,
        max_length: int = 256,
        seed: Optional[int] = 42
    ):
        self.num_sequences = num_sequences
        self.max_length = max_length
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Character vocabulary
        self.chars = list(string.ascii_lowercase) + ['(', ')', '[', ']', '{', '}']
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.chars)
        
        # Special tokens
        self.PAD_TOKEN = self.vocab_size
        self.SOS_TOKEN = self.vocab_size + 1  # Start of sequence
        self.EOS_TOKEN = self.vocab_size + 2  # End of sequence
        self.total_vocab_size = self.vocab_size + 3
        
        self.sequences = []
        self.labels = []
        self._generate_dataset()
    
    def _generate_dataset(self):
        """Generate dataset - to be implemented by subclasses."""
        raise NotImplementedError
    
    def _encode_sequence(self, sequence: str) -> List[int]:
        """Encode string sequence to token indices."""
        return [self.char_to_idx[char] for char in sequence if char in self.char_to_idx]
    
    def _decode_sequence(self, indices: List[int]) -> str:
        """Decode token indices to string sequence."""
        return ''.join([self.idx_to_char.get(idx, '') for idx in indices])
    
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


class ParenthesesDataset(ChomskyDataset):
    """
    Context-Free Grammar: Balanced Parentheses
    Tests ability to handle nested structures.
    
    Examples:
    - Valid: "()", "(())", "((()))", "()(())"
    - Invalid: "(()", "())", "()("
    """
    
    def __init__(self, num_sequences: int = 10000, max_length: int = 64, **kwargs):
        self.paren_types = ['()', '[]', '{}']
        super().__init__(num_sequences, max_length, **kwargs)
    
    def _generate_balanced_sequence(self, length: int) -> str:
        """Generate a balanced parentheses sequence."""
        if length == 0 or length % 2 != 0:
            return ""
        
        sequence = []
        stack = []
        pairs = length // 2
        
        # Choose parentheses types
        paren_type = random.choice(self.paren_types)
        open_paren, close_paren = paren_type[0], paren_type[1]
        
        for _ in range(length):
            if len(stack) == 0 or (len(stack) < pairs and random.random() < 0.6):
                # Add opening parenthesis
                sequence.append(open_paren)
                stack.append(open_paren)
            else:
                # Add closing parenthesis
                sequence.append(close_paren)
                stack.pop()
        
        # Ensure all are closed
        while stack:
            sequence.append(close_paren)
            stack.pop()
        
        return ''.join(sequence)
    
    def _generate_unbalanced_sequence(self, length: int) -> str:
        """Generate an unbalanced parentheses sequence."""
        paren_type = random.choice(self.paren_types)
        open_paren, close_paren = paren_type[0], paren_type[1]
        
        sequence = []
        for _ in range(length):
            sequence.append(random.choice([open_paren, close_paren]))
        
        # Ensure it's actually unbalanced
        seq_str = ''.join(sequence)
        if self._is_balanced(seq_str):
            # Force imbalance
            if len(sequence) > 0:
                sequence[0] = close_paren
        
        return ''.join(sequence)
    
    def _is_balanced(self, sequence: str) -> bool:
        """Check if a parentheses sequence is balanced."""
        stack = []
        pairs = {'(': ')', '[': ']', '{': '}'}
        
        for char in sequence:
            if char in pairs:
                stack.append(char)
            elif char in pairs.values():
                if not stack:
                    return False
                if pairs[stack.pop()] != char:
                    return False
        
        return len(stack) == 0
    
    def _generate_dataset(self):
        """Generate dataset of balanced and unbalanced parentheses."""
        for _ in range(self.num_sequences):
            # Generate sequence length
            seq_len = random.randint(4, min(self.max_length - 2, 32))  # Leave room for special tokens
            
            # 50% balanced, 50% unbalanced
            if random.random() < 0.5:
                seq_str = self._generate_balanced_sequence(seq_len)
                label = 1  # Balanced
            else:
                seq_str = self._generate_unbalanced_sequence(seq_len)
                label = 0  # Unbalanced
            
            # Encode sequence
            encoded = [self.SOS_TOKEN] + self._encode_sequence(seq_str) + [self.EOS_TOKEN]
            
            # Pad to max length
            if len(encoded) < self.max_length:
                encoded.extend([self.PAD_TOKEN] * (self.max_length - len(encoded)))
            else:
                encoded = encoded[:self.max_length]
            
            self.sequences.append(encoded)
            self.labels.append(label)


class ABnDataset(ChomskyDataset):
    """
    Context-Sensitive Grammar: a^n b^n
    Tests ability to count and match equal quantities.
    
    Examples:
    - Valid: "ab", "aabb", "aaabbb"
    - Invalid: "aab", "abb", "aabbb"
    """
    
    def _generate_dataset(self):
        """Generate dataset of a^n b^n sequences."""
        for _ in range(self.num_sequences):
            # Generate n (number of a's and b's)
            n = random.randint(1, min(self.max_length // 2 - 1, 16))
            
            # 70% valid, 30% invalid
            if random.random() < 0.7:
                # Valid: a^n b^n
                seq_str = 'a' * n + 'b' * n
                label = 1
            else:
                # Invalid: random number of a's and b's
                n_a = random.randint(1, n + 2)
                n_b = random.randint(1, n + 2)
                if n_a == n_b:  # Ensure it's actually invalid
                    n_b += 1
                seq_str = 'a' * n_a + 'b' * n_b
                label = 0
            
            # Encode sequence
            encoded = [self.SOS_TOKEN] + self._encode_sequence(seq_str) + [self.EOS_TOKEN]
            
            # Pad to max length
            if len(encoded) < self.max_length:
                encoded.extend([self.PAD_TOKEN] * (self.max_length - len(encoded)))
            else:
                encoded = encoded[:self.max_length]
            
            self.sequences.append(encoded)
            self.labels.append(label)


class ABCnDataset(ChomskyDataset):
    """
    Context-Sensitive Grammar: a^n b^n c^n
    Tests ability to count and match three equal quantities.
    
    Examples:
    - Valid: "abc", "aabbcc", "aaabbbccc"
    - Invalid: "aabbc", "abbc", "aabbbc"
    """
    
    def _generate_dataset(self):
        """Generate dataset of a^n b^n c^n sequences."""
        for _ in range(self.num_sequences):
            # Generate n
            n = random.randint(1, min(self.max_length // 3 - 1, 10))
            
            # 70% valid, 30% invalid
            if random.random() < 0.7:
                # Valid: a^n b^n c^n
                seq_str = 'a' * n + 'b' * n + 'c' * n
                label = 1
            else:
                # Invalid: random numbers
                n_a = random.randint(1, n + 1)
                n_b = random.randint(1, n + 1)
                n_c = random.randint(1, n + 1)
                
                # Ensure not all equal
                if n_a == n_b == n_c:
                    n_c += 1
                
                seq_str = 'a' * n_a + 'b' * n_b + 'c' * n_c
                label = 0
            
            # Encode sequence
            encoded = [self.SOS_TOKEN] + self._encode_sequence(seq_str) + [self.EOS_TOKEN]
            
            # Pad to max length
            if len(encoded) < self.max_length:
                encoded.extend([self.PAD_TOKEN] * (self.max_length - len(encoded)))
            else:
                encoded = encoded[:self.max_length]
            
            self.sequences.append(encoded)
            self.labels.append(label)


class RegularLanguageDataset(ChomskyDataset):
    """
    Regular Grammar: (ab)*
    Tests ability to recognize regular patterns.
    
    Examples:
    - Valid: "", "ab", "abab", "ababab"
    - Invalid: "a", "ba", "aba", "abba"
    """
    
    def _generate_dataset(self):
        """Generate dataset of (ab)* sequences."""
        for _ in range(self.num_sequences):
            # Generate number of repetitions
            n_reps = random.randint(0, min(self.max_length // 2 - 1, 10))
            
            # 70% valid, 30% invalid
            if random.random() < 0.7:
                # Valid: (ab)^n
                seq_str = 'ab' * n_reps
                label = 1
            else:
                # Invalid: break the pattern
                if n_reps == 0:
                    seq_str = random.choice(['a', 'b', 'ba'])
                else:
                    seq_str = 'ab' * n_reps
                    # Insert error
                    if len(seq_str) > 0:
                        error_pos = random.randint(0, len(seq_str) - 1)
                        seq_list = list(seq_str)
                        seq_list[error_pos] = random.choice(['c', 'd'])
                        seq_str = ''.join(seq_list)
                label = 0
            
            # Encode sequence
            encoded = [self.SOS_TOKEN] + self._encode_sequence(seq_str) + [self.EOS_TOKEN]
            
            # Pad to max length
            if len(encoded) < self.max_length:
                encoded.extend([self.PAD_TOKEN] * (self.max_length - len(encoded)))
            else:
                encoded = encoded[:self.max_length]
            
            self.sequences.append(encoded)
            self.labels.append(label)


def create_chomsky_datasets(
    dataset_type: str = "parentheses",
    train_size: int = 8000,
    val_size: int = 1000,
    test_size: int = 1000,
    **kwargs
) -> Tuple[ChomskyDataset, ChomskyDataset, ChomskyDataset]:
    """
    Create train, validation, and test datasets for Chomsky hierarchy tasks.
    
    Args:
        dataset_type: Type of dataset ("parentheses", "abn", "abcn", "regular")
        train_size: Size of training set
        val_size: Size of validation set
        test_size: Size of test set
        **kwargs: Additional arguments for dataset
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    dataset_classes = {
        "parentheses": ParenthesesDataset,
        "abn": ABnDataset,
        "abcn": ABCnDataset,
        "regular": RegularLanguageDataset
    }
    
    if dataset_type not in dataset_classes:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    DatasetClass = dataset_classes[dataset_type]
    
    train_dataset = DatasetClass(num_sequences=train_size, seed=42, **kwargs)
    val_dataset = DatasetClass(num_sequences=val_size, seed=123, **kwargs)
    test_dataset = DatasetClass(num_sequences=test_size, seed=456, **kwargs)
    
    return train_dataset, val_dataset, test_dataset


def evaluate_classification_accuracy(
    model,
    dataset: ChomskyDataset,
    device: str = 'cuda',
    batch_size: int = 32
) -> float:
    """
    Evaluate classification accuracy on Chomsky hierarchy task.
    
    Args:
        model: Trained model
        dataset: Dataset to evaluate on
        device: Device to run on
        batch_size: Batch size
        
    Returns:
        Classification accuracy
    """
    model.eval()
    dataloader = dataset.create_dataloader(batch_size=batch_size, shuffle=False)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            
            # Get predictions at the last position
            predictions = torch.argmax(logits[:, -1, :], dim=-1)
            
            correct += (predictions == labels).sum().item()
            total += labels.shape[0]
    
    return correct / total if total > 0 else 0.0


if __name__ == "__main__":
    # Test datasets
    datasets = {
        "Parentheses": ParenthesesDataset(100),
        "A^n B^n": ABnDataset(100),
        "A^n B^n C^n": ABCnDataset(100),
        "Regular (ab)*": RegularLanguageDataset(100)
    }
    
    for name, dataset in datasets.items():
        print(f"\n{name} Dataset:")
        print(f"Size: {len(dataset)}")
        print(f"Vocab size: {dataset.total_vocab_size}")
        
        sample = dataset[0]
        print(f"Sample input shape: {sample['input_ids'].shape}")
        print(f"Sample label: {sample['labels'].item()}")
        
        # Decode and show a few examples
        for i in range(3):
            sample = dataset[i]
            input_ids = sample['input_ids'].tolist()
            # Remove special tokens and padding for display
            clean_ids = [idx for idx in input_ids if idx not in [dataset.PAD_TOKEN, dataset.SOS_TOKEN, dataset.EOS_TOKEN]]
            decoded = dataset._decode_sequence(clean_ids)
            label = sample['labels'].item()
            print(f"  Example {i+1}: '{decoded}' -> {label}")
