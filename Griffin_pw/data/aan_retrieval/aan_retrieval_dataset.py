"""
AAN Retrieval Dataset
Tests cross-document information matching (from Long Range Arena)
Local Attention struggles when relevant connections are outside the local window.
"""

import torch
import random
from typing import List, Dict, Tuple
import json

class AANRetrievalDataset:
    """AAN Retrieval - cross-document information matching."""

    def __init__(self, num_samples: int = 1000, seq_length: int = 256, vocab_size: int = 100):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        # Special tokens
        self.doc_separator = vocab_size - 4
        self.query_token = vocab_size - 3
        self.match_token = vocab_size - 2
        self.end_token = vocab_size - 1

    def generate_retrieval_sample(self) -> Dict:
        """Generate a cross-document retrieval sample."""
        # Create two "documents" with related information
        # The task is to find connections between them

        # Generate related concepts
        num_concepts = random.randint(3, 6)
        concepts = []
        for i in range(num_concepts):
            concept_words = [random.randint(1, self.vocab_size - 5) for _ in range(random.randint(2, 4))]
            concepts.append({
                "id": i,
                "words": concept_words,
                "importance": random.randint(1, 3)
            })

        # Create Document 1
        doc1_content = []
        doc1_concept_positions = {}

        # Add some introductory content
        intro1 = [random.randint(1, self.vocab_size - 5) for _ in range(random.randint(5, 10))]
        doc1_content.extend(intro1)

        # Add concepts (some shared, some unique to doc1)
        doc1_concepts = random.sample(concepts, random.randint(2, len(concepts)))
        for concept in doc1_concepts:
            doc1_concept_positions[concept["id"]] = len(doc1_content)
            doc1_content.extend(concept["words"])
            # Add some context around the concept
            context = [random.randint(1, self.vocab_size - 5) for _ in range(random.randint(1, 3))]
            doc1_content.extend(context)

        # Create Document 2
        doc2_content = []
        doc2_concept_positions = {}

        # Add some introductory content
        intro2 = [random.randint(1, self.vocab_size - 5) for _ in range(random.randint(5, 10))]
        doc2_content.extend(intro2)

        # Add concepts (overlapping with doc1)
        doc2_concepts = random.sample(concepts, random.randint(2, len(concepts)))
        for concept in doc2_concepts:
            doc2_concept_positions[concept["id"]] = len(doc2_content)
            doc2_content.extend(concept["words"])
            # Add some context around the concept
            context = [random.randint(1, self.vocab_size - 5) for _ in range(random.randint(1, 3))]
            doc2_content.extend(context)

        # Build the full input sequence
        sequence = []
        sequence.extend(doc1_content)
        sequence.append(self.doc_separator)
        sequence.extend(doc2_content)

        # Query: find if two concepts are related (appear in both documents)
        # Choose two concepts - one that appears in both, one that doesn't
        shared_concepts = set(c["id"] for c in doc1_concepts) & set(c["id"] for c in doc2_concepts)
        unshared_concepts = set(c["id"] for c in concepts) - shared_concepts

        if shared_concepts and random.random() > 0.5:
            # Test shared concept
            query_concept_id = random.choice(list(shared_concepts))
            expected_answer = 1  # Connected
        else:
            # Test unshared concept
            query_concept_id = random.choice(concepts)["id"] if concepts else 0
            expected_answer = 1 if query_concept_id in shared_concepts else 0

        sequence.append(self.query_token)
        sequence.append(query_concept_id)

        # Target: 1 if concepts are connected across documents, 0 otherwise
        target = [expected_answer, self.end_token]

        # Pad sequences
        input_padded = sequence + [0] * (self.seq_length - len(sequence))
        target_padded = target + [0] * (self.seq_length - len(target))

        return {
            "input_ids": input_padded[:self.seq_length],
            "target_ids": target_padded[:self.seq_length],
            "doc1_concepts": [c["id"] for c in doc1_concepts],
            "doc2_concepts": [c["id"] for c in doc2_concepts],
            "shared_concepts": list(shared_concepts),
            "query_concept": query_concept_id,
            "expected_answer": expected_answer,
            "connection_type": "cross_document_match" if expected_answer else "no_match"
        }

    def generate_dataset(self) -> List[Dict]:
        """Generate the full dataset."""
        return [self.generate_retrieval_sample() for _ in range(self.num_samples)]