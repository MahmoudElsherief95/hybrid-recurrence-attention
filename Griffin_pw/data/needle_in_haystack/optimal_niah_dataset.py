"""
Optimized Needle-in-a-Haystack Dataset for Long-Range Tasks
Based on "Hidden in the Haystack: Smaller Needles are More Difficult for LLMs to Find"
https://arxiv.org/html/2505.18148v2

This dataset creates challenging NIAH tasks with:
- Multiple needle sizes (tiny/small/medium) - smaller = harder per paper
- Extensive position variation (6 positions)
- Topically relevant distractors
- Factual retrieval tasks
- Controlled for confounders (position, repetition, ratio)
"""

import json
import random
from typing import List, Dict, Tuple
import os

class OptimizedNIAHDataset:
    def __init__(self, num_samples: int = 1000, context_length: int = 4096, diverse_configs: bool = True):
        self.num_samples = num_samples
        self.context_length = context_length
        self.diverse_configs = diverse_configs

        # Diverse needle sizes for comprehensive testing
        if diverse_configs:
            self.needle_sizes = {
                'tiny': 5,      # Very hard - minimal information
                'small': 15,    # Hard - limited information
                'medium': 35,   # Moderate - reasonable information
                'large': 75     # Easy - comprehensive information
            }
            # Diverse context lengths
            self.context_lengths = [512, 1024, 2048, 4096]
            # Diverse distractor counts
            self.distractor_counts = [5, 10, 15, 20]
        else:
            # Original configuration
            self.needle_sizes = {
                'tiny': 10,
                'small': 25,
                'medium': 50
            }
            self.context_lengths = [context_length]
            self.distractor_counts = [15]

        # Extended position variations (8 positions for more comprehensive testing)
        self.positions = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]

        # Factual knowledge base for retrieval tasks
        self.facts = self._load_facts()

    def _load_facts(self) -> List[Dict]:
        """Load diverse factual information for retrieval tasks"""
        facts = [
            {
                "topic": "science",
                "fact": "The speed of light in vacuum is exactly 299,792,458 meters per second",
                "question": "What is the exact speed of light in vacuum?",
                "answer": "299,792,458 meters per second"
            },
            {
                "topic": "history",
                "fact": "The first moon landing occurred on July 20, 1969, when Apollo 11 astronauts Neil Armstrong and Buzz Aldrin walked on the lunar surface",
                "question": "When did the first moon landing occur and who were the astronauts?",
                "answer": "July 20, 1969, Neil Armstrong and Buzz Aldrin"
            },
            {
                "topic": "geography",
                "fact": "Mount Everest, the highest mountain in the world, stands at 8,848.86 meters above sea level in the Himalayas",
                "question": "What is the height of Mount Everest?",
                "answer": "8,848.86 meters"
            },
            {
                "topic": "biology",
                "fact": "DNA contains the genetic instructions for the development and functioning of all known living organisms",
                "question": "What does DNA contain?",
                "answer": "genetic instructions for development and functioning"
            },
            {
                "topic": "physics",
                "fact": "The Planck constant is 6.62607015 × 10^-34 joule-seconds, fundamental to quantum mechanics",
                "question": "What is the value of the Planck constant?",
                "answer": "6.62607015 × 10^-34 joule-seconds"
            },
            {
                "topic": "chemistry",
                "fact": "The periodic table contains 118 chemical elements, with hydrogen having atomic number 1",
                "question": "How many chemical elements are in the periodic table?",
                "answer": "118"
            },
            {
                "topic": "mathematics",
                "fact": "Pi (π) is an irrational number approximately equal to 3.14159265359, representing the ratio of circumference to diameter",
                "question": "What is the approximate value of pi?",
                "answer": "3.14159265359"
            },
            {
                "topic": "astronomy",
                "fact": "The Milky Way galaxy contains between 100 and 400 billion stars, including our Sun",
                "question": "How many stars does the Milky Way contain?",
                "answer": "100 to 400 billion"
            },
            {
                "topic": "medicine",
                "fact": "The human heart beats approximately 100,000 times per day, pumping about 2,000 gallons of blood",
                "question": "How many times does the human heart beat per day?",
                "answer": "100,000"
            },
            {
                "topic": "technology",
                "fact": "The first programmable computer was the Z3, created by Konrad Zuse in 1941 in Germany",
                "question": "What was the first programmable computer?",
                "answer": "Z3 by Konrad Zuse"
            }
        ]
        return facts

    def _generate_distractors(self, topic: str, num_distractors: int = 10) -> List[str]:
        """Generate topically relevant distractors"""
        distractors = []

        # Topic-specific distractor templates
        templates = {
            "science": [
                "The speed of sound in air is approximately 343 meters per second at room temperature.",
                "Light travels faster in water than in air, but slower than in vacuum.",
                "The Earth's core temperature is estimated to be around 5,000 to 6,000 degrees Celsius.",
                "A light year is the distance light travels in one year, about 9.46 trillion kilometers.",
                "The human body contains about 206 bones in adulthood.",
                "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
                "The Earth's atmosphere is composed mainly of nitrogen (78%) and oxygen (21%).",
                "Gravity on Earth accelerates objects at 9.8 meters per second squared.",
                "The human brain weighs approximately 1.4 kilograms on average.",
                "Sound waves travel faster in solids than in liquids or gases."
            ],
            "history": [
                "The Roman Empire lasted for over 1,000 years, from 27 BC to 476 AD.",
                "World War II ended in 1945 with the defeat of Nazi Germany and Japan.",
                "The Industrial Revolution began in Britain in the late 18th century.",
                "The American Civil War lasted from 1861 to 1865.",
                "The Berlin Wall fell in 1989, leading to German reunification.",
                "Christopher Columbus reached the Americas in 1492.",
                "The French Revolution began in 1789 with the storming of the Bastille.",
                "The Magna Carta was signed in 1215, limiting the power of English kings.",
                "The Cold War lasted from 1947 to 1991 between the US and Soviet Union.",
                "The Renaissance period in Europe lasted from the 14th to 17th centuries."
            ],
            "geography": [
                "The Pacific Ocean is the largest ocean, covering about 46% of Earth's water surface.",
                "Australia is both a country and a continent, located in Oceania.",
                "The Sahara Desert is the largest hot desert in the world, covering 9 million square kilometers.",
                "The Amazon River is the longest river in the world by discharge volume.",
                "Antarctica is the coldest continent, with temperatures as low as -89°C.",
                "The Himalayas contain 9 of the world's 10 highest peaks.",
                "Greenland is the world's largest island by area.",
                "The Dead Sea is the lowest point on Earth, 430 meters below sea level.",
                "The Great Barrier Reef is the world's largest coral reef system.",
                "Mount Kilimanjaro is the highest mountain in Africa at 5,895 meters."
            ],
            "biology": [
                "Photosynthesis converts light energy into chemical energy in plants.",
                "The human body has 23 pairs of chromosomes, totaling 46 chromosomes.",
                "Mitochondria are known as the powerhouse of the cell.",
                "The largest organ in the human body is the skin.",
                "Blood is composed of red blood cells, white blood cells, platelets, and plasma.",
                "The human eye can distinguish about 10 million colors.",
                "Plants produce oxygen as a byproduct of photosynthesis.",
                "The human brain contains approximately 86 billion neurons.",
                "Enzymes are biological catalysts that speed up chemical reactions.",
                "The process of cell division is called mitosis in somatic cells."
            ],
            "physics": [
                "Newton's first law states that an object at rest stays at rest unless acted upon by a force.",
                "Energy cannot be created or destroyed, only transformed (first law of thermodynamics).",
                "The strong nuclear force holds protons and neutrons together in atomic nuclei.",
                "Electromagnetic waves include radio waves, microwaves, infrared, visible light, ultraviolet, X-rays, and gamma rays.",
                "The uncertainty principle states that position and momentum cannot both be precisely measured simultaneously.",
                "Black holes have such strong gravity that not even light can escape.",
                "The Doppler effect explains why sound pitch changes when the source moves.",
                "Superconductors can conduct electricity with zero resistance at very low temperatures.",
                "The photoelectric effect shows that light behaves as both waves and particles.",
                "Quantum entanglement allows particles to be connected regardless of distance."
            ],
            "chemistry": [
                "Carbon has atomic number 6 and can form four covalent bonds.",
                "Water is a polar molecule due to its bent shape and electronegativity difference.",
                "The pH scale measures acidity from 0 (acidic) to 14 (basic), with 7 being neutral.",
                "Catalysts speed up chemical reactions without being consumed.",
                "Isotopes are atoms of the same element with different numbers of neutrons.",
                "The noble gases are helium, neon, argon, krypton, xenon, and radon.",
                "Chemical bonds include ionic bonds, covalent bonds, and metallic bonds.",
                "Avogadro's number is 6.022 × 10^23 particles per mole.",
                "Organic chemistry studies compounds containing carbon-hydrogen bonds.",
                "The halogens are a group of reactive nonmetals: fluorine, chlorine, bromine, iodine, and astatine."
            ],
            "mathematics": [
                "A prime number has exactly two distinct positive divisors: 1 and itself.",
                "The Pythagorean theorem states that a² + b² = c² for right triangles.",
                "Zero is neither positive nor negative and is the additive identity.",
                "The Fibonacci sequence starts with 0, 1, 1, 2, 3, 5, 8, 13...",
                "A square number is the product of an integer multiplied by itself.",
                "The golden ratio (φ) is approximately 1.618 and appears in nature and art.",
                "Probability ranges from 0 (impossible) to 1 (certain).",
                "The area of a circle is πr², where r is the radius.",
                "Statistics uses mean, median, and mode to describe data sets.",
                "Geometry studies shapes, sizes, and properties of space."
            ],
            "astronomy": [
                "The Sun is a G-type main-sequence star at the center of our solar system.",
                "Jupiter is the largest planet with a diameter of about 143,000 kilometers.",
                "A light year is the distance light travels in one year, about 9.46 trillion kilometers.",
                "The Big Bang theory describes the origin of the universe 13.8 billion years ago.",
                "Black holes form when massive stars collapse under their own gravity.",
                "The Moon is tidally locked to Earth, always showing the same face.",
                "Comets are icy bodies that develop tails when approaching the Sun.",
                "The Kuiper Belt contains many icy objects beyond Neptune's orbit.",
                "Galaxies come in spiral, elliptical, and irregular shapes.",
                "The observable universe has a diameter of about 93 billion light years."
            ],
            "medicine": [
                "The human body has four main blood types: A, B, AB, and O.",
                "Vaccines work by stimulating the immune system to recognize and fight pathogens.",
                "Antibiotics are effective against bacterial infections but not viral infections.",
                "The normal human body temperature is approximately 37°C (98.6°F).",
                "Diabetes is a condition where the body cannot properly regulate blood sugar.",
                "The immune system protects the body from infections and diseases.",
                "Placebo effect shows that belief can influence health outcomes.",
                "Cancer occurs when cells divide uncontrollably and spread to other tissues.",
                "Mental health includes emotional, psychological, and social well-being.",
                "Sleep is essential for physical and mental health, with adults needing 7-9 hours."
            ],
            "technology": [
                "The internet was originally developed as ARPANET in the 1960s.",
                "Artificial intelligence uses algorithms to perform tasks that typically require human intelligence.",
                "Blockchain technology underlies cryptocurrencies like Bitcoin.",
                "Quantum computing uses quantum bits (qubits) instead of classical bits.",
                "Renewable energy sources include solar, wind, hydro, and geothermal power.",
                "3D printing creates objects layer by layer from digital designs.",
                "Virtual reality immerses users in simulated environments.",
                "Machine learning algorithms improve performance as they process more data.",
                "The smartphone was first introduced by IBM in 1994.",
                "Cloud computing delivers computing services over the internet."
            ]
        }

        # Get distractors for the topic
        topic_distractors = templates.get(topic, templates["science"])
        selected_distractors = random.sample(topic_distractors, min(num_distractors, len(topic_distractors)))

        return selected_distractors

    def _create_needle_context(self, fact: Dict, size: str) -> str:
        """Create needle context of specified size"""
        base_fact = fact["fact"]
        tokens = base_fact.split()

        if size == 'tiny':
            # Minimal context - just the key fact
            needle = f"The key information is: {base_fact[:50]}..."
        elif size == 'small':
            # Small context - partial fact
            needle = f"Important fact: {base_fact[:100]}..."
        else:  # medium
            # Full context
            needle = f"Complete information: {base_fact}"

        return needle

    def generate_sample(self) -> Dict:
        """Generate a single NIAH sample with diverse configurations"""
        # Randomly select fact and size
        fact = random.choice(self.facts)
        size = random.choice(list(self.needle_sizes.keys()))

        # For diverse configs, randomly select context length and distractor count
        if self.diverse_configs:
            context_len = random.choice(self.context_lengths)
            distractor_count = random.choice(self.distractor_counts)
        else:
            context_len = self.context_length
            distractor_count = self.distractor_counts[0]

        position = random.choice(self.positions)

        # Create needle
        needle = self._create_needle_context(fact, size)

        # Generate distractors (use the selected count)
        distractors = self._generate_distractors(fact["topic"], num_distractors=distractor_count)

        # Calculate insertion position
        total_distractors = len(distractors)
        insert_idx = int(position * total_distractors)

        # Build context
        context_parts = []
        for i, distractor in enumerate(distractors):
            if i == insert_idx:
                context_parts.append(f"Document {i+1}: {needle}")
            context_parts.append(f"Document {i+1}: {distractor}")

        context = " ".join(context_parts)

        # Ensure context fits within length limit
        if len(context.split()) > context_len:
            context = " ".join(context.split()[:context_len])

        return {
            "question": fact["question"],
            "answer": fact["answer"],
            "context": context,
            "needle_size": size,
            "needle_position": position,
            "topic": fact["topic"],
            "needle_text": needle,
            "context_length": context_len,
            "distractor_count": distractor_count
        }

    def generate_dataset(self) -> List[Dict]:
        """Generate the full dataset"""
        dataset = []
        for _ in range(self.num_samples):
            sample = self.generate_sample()
            dataset.append(sample)
        return dataset

    def save_dataset(self, filepath: str):
        """Save dataset to JSON file"""
        dataset = self.generate_dataset()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"Dataset saved to {filepath}")

if __name__ == "__main__":
    # Create optimized NIAH dataset
    dataset_generator = OptimizedNIAHDataset(num_samples=1000, context_length=4096)
    dataset_generator.save_dataset("optimal_niah_dataset.json")