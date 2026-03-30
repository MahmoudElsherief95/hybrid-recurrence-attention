"""
Unified Dataset Interface for All Benchmark Datasets
Provides consistent interface for testing all datasets with different models.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import sys
import os
sys.path.append(os.path.dirname(__file__))

# Import all dataset implementations
from data.chomsky.chomsky_dataset import ChomskyDataset, ParenthesesDataset
from data.copy.copy_dataset import CopyDataset
from data.listops.listops_dataset import ListopsDataset
from data.mqar.mqar_dataset import MQARDataset
from data.needle_in_haystack.optimal_niah_dataset import OptimizedNIAHDataset
from data.selective_copy import SelectiveCopyDataset
from data.induction_heads import InductionHeadsDataset
from data.variable_binding import VariableBindingDataset
from data.path_x import PathXDataset
from data.sequential_niah import SequentialNIAHDataset
from data.aan_retrieval import AANRetrievalDataset
import json
import tempfile


class UnifiedDataset(Dataset):
    """Wrapper to provide consistent PyTorch Dataset interface for all datasets."""

    def __init__(self, dataset_name: str, num_samples: int = 1000, seq_length: int = 256, **kwargs):
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.seq_length = seq_length

        # Initialize the appropriate dataset
        if dataset_name.lower() == 'chomsky':
            # Use ParenthesesDataset (concrete subclass) — base ChomskyDataset is abstract
            self.dataset = ParenthesesDataset(num_sequences=num_samples, max_length=seq_length, **kwargs)
            self.vocab_size = self.dataset.total_vocab_size
        elif dataset_name.lower() == 'copy':
            self.dataset = CopyDataset(num_sequences=num_samples, seq_len=seq_length, **kwargs)
            self.vocab_size = self.dataset.total_vocab_size
        elif dataset_name.lower() == 'listops':
            self.dataset = ListopsDataset(num_sequences=num_samples, max_length=seq_length, **kwargs)
            self.vocab_size = self.dataset.total_vocab_size
        elif dataset_name.lower() == 'mqar':
            self.dataset = MQARDataset(num_sequences=num_samples, seq_len=seq_length, **kwargs)
            self.vocab_size = self.dataset.total_vocab_size
        elif dataset_name.lower() == 'niah_optimal':
            # Generate data using OptimizedNIAHDataset and save to temp file
            generator = OptimizedNIAHDataset(num_samples=num_samples, context_length=seq_length, **kwargs)
            data = generator.generate_dataset()
            self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            json.dump(data, self.temp_file)
            self.temp_file.close()
            
            # Create a simple tokenizer
            class SimpleTokenizer:
                def __init__(self):
                    self.vocab = {chr(i): i for i in range(256)}
                    self.vocab.update({f'<{i}>': 256 + i for i in range(100)})
                    self.vocab_size = len(self.vocab)
                    
                def encode(self, text):
                    return [self.vocab.get(c, 0) for c in text[:seq_length]]
                    
                def decode(self, tokens):
                    reverse_vocab = {v: k for k, v in self.vocab.items()}
                    return ''.join(reverse_vocab.get(t, '?') for t in tokens)
            
            self.tokenizer = SimpleTokenizer()
            self.vocab_size = self.tokenizer.vocab_size
            
            # Load data for Dataset interface
            with open(self.temp_file.name, 'r') as f:
                self.data = json.load(f)
                
        elif dataset_name.lower() in ['selective_copy', 'induction_heads', 'variable_binding', 
                                    'path_x', 'sequential_niah', 'aan_retrieval']:
            # These are custom dataset classes with generate_sample methods
            dataset_class = {
                'selective_copy': SelectiveCopyDataset,
                'induction_heads': InductionHeadsDataset,
                'variable_binding': VariableBindingDataset,
                'path_x': PathXDataset,
                'sequential_niah': SequentialNIAHDataset,
                'aan_retrieval': AANRetrievalDataset
            }[dataset_name.lower()]
            
            # path_x embeds raw position indices as token ids, so vocab_size must
            # cover the full sequence length range (positions 0..seq_length-1)
            extra_kwargs = {}
            if dataset_name.lower() == 'path_x':
                extra_kwargs['vocab_size'] = seq_length + 20  # positions + pixel types
            self.dataset = dataset_class(num_samples=num_samples, seq_length=seq_length, **extra_kwargs, **kwargs)
            self.vocab_size = getattr(self.dataset, 'vocab_size', 100)
            # Find the correct generate method (each dataset class uses a different name)
            _gen_method = next(
                (getattr(self.dataset, m) for m in dir(self.dataset)
                 if m.startswith('generate_') and m != 'generate_dataset' and callable(getattr(self.dataset, m))),
                None
            )
            if _gen_method is None:
                raise AttributeError(f"No generate_* method found on {dataset_class.__name__}")
            # Generate all samples at once
            self.samples = [_gen_method() for _ in range(num_samples)]
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def __len__(self):
        if hasattr(self, 'data'):
            return len(self.data)
        elif hasattr(self, 'samples'):
            return len(self.samples)
        else:
            return len(self.dataset)

    def __getitem__(self, idx):
        if hasattr(self, 'data'):
            # NIAH case
            item = self.data[idx]
            text = item['text']
            tokens = self.tokenizer.encode(text)
            # Pad/truncate to seq_length
            if len(tokens) > self.seq_length:
                tokens = tokens[:self.seq_length]
            else:
                tokens.extend([0] * (self.seq_length - len(tokens)))
            
            return {
                'input_ids': torch.tensor(tokens, dtype=torch.long),
                'attention_mask': torch.ones(self.seq_length, dtype=torch.long),
                'labels': torch.tensor(tokens, dtype=torch.long)  # For LM task
            }
        elif hasattr(self, 'samples'):
            # Custom dataset case
            sample = self.samples[idx]
            # Convert to tensor format expected by models
            input_seq = sample.get('input_ids', sample.get('input', []))
            target_seq = sample.get('target_ids', sample.get('target', input_seq))
            
            # Pad/truncate
            if len(input_seq) > self.seq_length:
                input_seq = input_seq[:self.seq_length]
                target_seq = target_seq[:self.seq_length]
            else:
                input_seq.extend([0] * (self.seq_length - len(input_seq)))
                target_seq.extend([0] * (self.seq_length - len(target_seq)))
            
            return {
                'input_ids': torch.tensor(input_seq, dtype=torch.long),
                'attention_mask': torch.ones(self.seq_length, dtype=torch.long),
                'labels': torch.tensor(target_seq, dtype=torch.long)
            }
        else:
            # Standard Dataset case
            return self.dataset[idx]

    def __del__(self):
        # Clean up temp file for NIAH
        if hasattr(self, 'temp_file'):
            try:
                os.unlink(self.temp_file.name)
            except:
                pass


def get_dataset_info(dataset_name: str) -> dict:
    """Get information about a dataset for model configuration."""
    info = {
        'chomsky': {
            'description': 'Chomsky Hierarchy - syntactic complexity',
            'task_type': 'language_modeling',
            'vocab_size': 32  # 26 letters + 6 brackets + 3 special
        },
        'copy': {
            'description': 'Sequence copying task',
            'task_type': 'sequence_to_sequence',
            'vocab_size': 32
        },
        'listops': {
            'description': 'List operations parsing',
            'task_type': 'classification',
            'vocab_size': 32
        },
        'mqar': {
            'description': 'Multi-Query Answering Reasoning',
            'task_type': 'language_modeling',
            'vocab_size': 32
        },
        'niah_optimal': {
            'description': 'Needle in a Haystack - optimal challenging',
            'task_type': 'retrieval',
            'vocab_size': 32
        },
        'selective_copy': {
            'description': 'Selective copying with distractors',
            'task_type': 'selective_retrieval',
            'vocab_size': 100
        },
        'induction_heads': {
            'description': 'Pattern induction across sequences',
            'task_type': 'pattern_completion',
            'vocab_size': 50
        },
        'variable_binding': {
            'description': 'Variable binding and associative memory',
            'task_type': 'associative_memory',
            'vocab_size': 50
        },
        'path_x': {
            'description': 'Path-X pixel connectivity (LRA-inspired)',
            'task_type': 'connectivity',
            'vocab_size': 32
        },
        'sequential_niah': {
            'description': 'Sequential NIAH with multiple needles',
            'task_type': 'multi_retrieval',
            'vocab_size': 32
        },
        'aan_retrieval': {
            'description': 'AAN cross-document retrieval',
            'task_type': 'document_matching',
            'vocab_size': 100
        }
    }

    return info.get(dataset_name.lower(), {
        'description': 'Unknown dataset',
        'task_type': 'unknown',
        'vocab_size': 32
    })


def create_data_loader(dataset_name: str, batch_size: int = 32, num_samples: int = 1000,
                      seq_length: int = 256, **kwargs) -> DataLoader:
    """Create a DataLoader for the specified dataset."""
    dataset = UnifiedDataset(dataset_name, num_samples=num_samples,
                           seq_length=seq_length, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# List of all available datasets
ALL_DATASETS = [
    'chomsky', 'copy', 'listops', 'mqar',
    'selective_copy', 'induction_heads', 'variable_binding',
    'path_x', 'sequential_niah', 'aan_retrieval'
]


if __name__ == "__main__":
    # Quick test to ensure all datasets can be loaded
    print("Testing dataset loading...")

    for dataset_name in ALL_DATASETS:
        try:
            print(f"Loading {dataset_name}...")
            dataset = UnifiedDataset(dataset_name, num_samples=10)
            info = get_dataset_info(dataset_name)
            print(f"  ✓ {dataset_name}: {len(dataset)} samples, vocab_size={dataset.vocab_size}")
            print(f"    {info['description']}")

            # Test a sample
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"    Sample keys: {list(sample.keys()) if isinstance(sample, dict) else 'tensor'}")

        except Exception as e:
            print(f"  ✗ {dataset_name}: {str(e)}")

    print("\nAll datasets tested!")