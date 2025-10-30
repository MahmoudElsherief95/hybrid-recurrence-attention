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
    
    def __init__(self, num_sequences: int = 10000, max_length: int = 512, min_nesting: int = 32, **kwargs):
        self.paren_types = ['()', '[]', '{}']
        self.min_nesting = min_nesting
        super().__init__(num_sequences, max_length, **kwargs)

    def _generate_balanced_sequence(self, length: int) -> str:
        """Generate a deeply nested balanced parentheses sequence."""
        if length == 0 or length % 2 != 0:
            return ""
        max_nesting = max(1, length // 2)
        min_nesting = min(self.min_nesting, max_nesting)
        if min_nesting > max_nesting:
            min_nesting = max_nesting
        nesting = random.randint(min_nesting, max_nesting)
        paren_type = random.choice(self.paren_types)
        open_paren, close_paren = paren_type[0], paren_type[1]
        sequence = [open_paren] * nesting + [close_paren] * nesting
        remaining = length - 2 * nesting
        # Add random pairs and single parentheses to increase difficulty
        for _ in range(remaining // 2):
            if random.random() < 0.5:
                sequence.insert(nesting, open_paren)
                sequence.insert(nesting + 1, close_paren)
            else:
                sequence.insert(nesting, random.choice([open_paren, close_paren]))
        # Add random single parentheses to remaining positions
        for _ in range(remaining % 2):
            sequence.insert(nesting, random.choice([open_paren, close_paren]))
        return ''.join(sequence)

    def _generate_unbalanced_sequence(self, length: int) -> str:
        """Generate a highly non-trivial unbalanced parentheses sequence."""
        # Start with a deeply nested balanced sequence
        seq_str = self._generate_balanced_sequence(length)
        seq_list = list(seq_str)
        # Introduce many errors: remove, swap, or insert wrong parentheses
        num_errors = random.randint(max(8, length // 32), max(16, length // 16))
        for _ in range(num_errors):
            if len(seq_list) > 0:
                idx = random.randint(0, len(seq_list) - 1)
                action = random.choice(['remove', 'swap', 'insert'])
                if action == 'remove':
                    seq_list.pop(idx)
                elif action == 'swap':
                    seq_list[idx] = random.choice(['(', ')', '[', ']', '{', '}'])
                elif action == 'insert':
                    seq_list.insert(idx, random.choice(['(', ')', '[', ']', '{', '}']))
        return ''.join(seq_list)

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


class AnBnDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for a^n b^n classification (valid/invalid).
    Uses train/test splits from generate_an_bn_dataset.
    """
    def __init__(self, data, max_length=256, vocab=['a', 'b']):
        self.data = data
        self.vocab = vocab
        self.char_to_idx = {c: i for i, c in enumerate(vocab)}
        self.PAD_TOKEN = len(vocab)
        self.SOS_TOKEN = len(vocab) + 1
        self.EOS_TOKEN = len(vocab) + 2
        self.total_vocab_size = len(vocab) + 3
        self.max_length = max_length
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        seq, label = self.data[idx]
        encoded = [self.SOS_TOKEN] + [self.char_to_idx[c] for c in seq] + [self.EOS_TOKEN]
        if len(encoded) < self.max_length:
            encoded += [self.PAD_TOKEN] * (self.max_length - len(encoded))
        else:
            encoded = encoded[:self.max_length]
        input_ids = torch.tensor(encoded, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)
        attention_mask = (input_ids != self.PAD_TOKEN).long()
        return {
            'input_ids': input_ids,
            'labels': label_tensor,
            'attention_mask': attention_mask
        }

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


def generate_an_bn_dataset(train_n_range=(1, 10), test_n_range=(20, 40, 80, 160), num_train=5000, num_test=1000, vocab=['a', 'b'], hard_negatives=True):
    """
    Generate a^n b^n dataset with train/test splits and hard negatives.
    Returns: train_data, test_data (list of (sequence, label))
    """
    def make_positive(n):
        return ''.join(['a'] * n + ['b'] * n), 1
    def make_hard_negative(n):
        # Off-by-one, shuffled, extra/missing tokens
        options = [
            ''.join(['a'] * n + ['b'] * (n-1)),
            ''.join(['a'] * (n-1) + ['b'] * n),
            ''.join(['a'] * n + ['b'] * n + ['a']),
            ''.join(['b'] * n + ['a'] * n),
            ''.join(['a'] * n + ['b'] * n + ['b']),
        ]
        return random.choice(options), 0
    train_data = []
    for _ in range(num_train):
        n = random.randint(train_n_range[0], train_n_range[1])
        if random.random() < 0.5:
            train_data.append(make_positive(n))
        else:
            train_data.append(make_hard_negative(n))
    test_data = []
    for n in test_n_range:
        for _ in range(num_test // len(test_n_range)):
            if random.random() < 0.5:
                test_data.append(make_positive(n))
            else:
                test_data.append(make_hard_negative(n))
    return train_data, test_data

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

train_data, test_data = generate_an_bn_dataset()
print(f"\nGenerated A^n B^n Dataset:")
print(f"Train size: {len(train_data)}")
print(f"Test size: {len(test_data)}")
print(f"Sample train sequence: {train_data[0][0]} -> Label: {train_data[0][1]}")
print(f"Sample test sequence: {test_data[0][0]} -> Label: {test_data[0][1]}")
