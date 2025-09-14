"""
Model Evaluator for Griffin Model Experiments
Provides comprehensive evaluation metrics and analysis
"""

import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
import json
import os
from dataclasses import dataclass
from torch.utils.data import DataLoader


@dataclass
class EvaluationMetrics:
    """Data class for storing evaluation metrics."""
    model_name: str
    dataset_name: str
    loss: float
    perplexity: float
    accuracy: Optional[float] = None
    recall_accuracy: Optional[float] = None
    inference_time: float = 0.0
    memory_usage: float = 0.0
    throughput: float = 0.0
    parameters: int = 0


class ModelEvaluator:
    """
    Comprehensive evaluator for Griffin, Hawk, and Local Attention models.
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.results = []
    
    def evaluate_model(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        model_name: str,
        dataset_name: str,
        num_samples: int = 1000
    ) -> EvaluationMetrics:
        """
        Comprehensive evaluation of a model.
        
        Args:
            model: Model to evaluate
            dataloader: DataLoader for evaluation data
            model_name: Name of the model (e.g., "Griffin", "Hawk", "LocalAttention")
            dataset_name: Name of the dataset (e.g., "MQAR", "Parentheses")
            num_samples: Number of samples to use for evaluation
            
        Returns:
            EvaluationMetrics object with all computed metrics
        """
        model.eval()
        
        # Basic metrics
        loss, perplexity = self._compute_language_modeling_metrics(model, dataloader)
        
        # Performance metrics
        inference_time, throughput = self._measure_inference_speed(model, dataloader, num_samples)
        memory_usage = self._measure_memory_usage(model)
        
        # Model parameters
        parameters = sum(p.numel() for p in model.parameters())
        
        # Task-specific accuracy (if applicable)
        accuracy = self._compute_classification_accuracy(model, dataloader)
        recall_accuracy = self._compute_recall_accuracy(model, dataloader, dataset_name)
        
        metrics = EvaluationMetrics(
            model_name=model_name,
            dataset_name=dataset_name,
            loss=loss,
            perplexity=perplexity,
            accuracy=accuracy,
            recall_accuracy=recall_accuracy,
            inference_time=inference_time,
            memory_usage=memory_usage,
            throughput=throughput,
            parameters=parameters
        )
        
        self.results.append(metrics)
        return metrics
    
    def _compute_language_modeling_metrics(
        self,
        model: nn.Module,
        dataloader: DataLoader
    ) -> Tuple[float, float]:
        """Compute language modeling loss and perplexity."""
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # Forward pass
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                
                # Compute loss
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.shape[-1]),
                    labels.view(-1),
                    reduction='none'
                )
                
                # Apply mask if available
                if attention_mask is not None:
                    mask = attention_mask.view(-1).float()
                    loss = (loss * mask).sum() / mask.sum()
                else:
                    loss = loss.mean()
                
                batch_size = input_ids.shape[0]
                seq_len = input_ids.shape[1]
                total_loss += loss.item() * batch_size
                total_tokens += batch_size
        
        avg_loss = total_loss / total_tokens
        perplexity = min(np.exp(avg_loss), 1e6)  # Cap perplexity to avoid overflow
        
        return avg_loss, perplexity
    
    def _measure_inference_speed(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        num_samples: int
    ) -> Tuple[float, float]:
        """Measure inference speed and throughput."""
        model.eval()
        
        times = []
        total_tokens = 0
        samples_processed = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if samples_processed >= num_samples:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # Warm up
                if samples_processed == 0:
                    for _ in range(5):
                        _ = model(input_ids, attention_mask=attention_mask)
                
                # Measure time
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                _ = model(input_ids, attention_mask=attention_mask)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                
                batch_time = end_time - start_time
                batch_size = input_ids.shape[0]
                seq_len = input_ids.shape[1]
                
                times.append(batch_time)
                total_tokens += batch_size * seq_len
                samples_processed += batch_size
        
        avg_inference_time = np.mean(times)
        throughput = total_tokens / sum(times)  # tokens per second
        
        return avg_inference_time, throughput
    
    def _measure_memory_usage(self, model: nn.Module) -> float:
        """Measure model memory usage in MB."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            # Measure before
            memory_before = torch.cuda.memory_allocated() / 1024**2
            
            # Create dummy input
            dummy_input = torch.randint(0, 1000, (1, 256)).to(self.device)
            
            with torch.no_grad():
                _ = model(dummy_input)
            
            torch.cuda.synchronize()
            memory_after = torch.cuda.memory_allocated() / 1024**2
            
            return memory_after - memory_before
        else:
            # Estimate based on parameters for CPU
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            return param_size / 1024**2
    
    def _compute_classification_accuracy(
        self,
        model: nn.Module,
        dataloader: DataLoader
    ) -> Optional[float]:
        """Compute classification accuracy if applicable."""
        try:
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    attention_mask = batch.get('attention_mask')
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)
                    
                    outputs = model(input_ids, attention_mask=attention_mask)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    
                    # For classification, predict at the last position
                    predictions = torch.argmax(logits[:, -1, :], dim=-1)
                    
                    # Check if labels are classification labels (single value per sequence)
                    if labels.dim() == 1 or labels.shape[1] == 1:
                        if labels.dim() == 2:
                            labels = labels.squeeze(1)
                        correct += (predictions == labels).sum().item()
                        total += labels.shape[0]
            
            return correct / total if total > 0 else None
        except:
            return None
    
    def _compute_recall_accuracy(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        dataset_name: str
    ) -> Optional[float]:
        """Compute recall accuracy for MQAR dataset."""
        if 'MQAR' not in dataset_name:
            return None
        
        # This would require access to the MQAR dataset's evaluation method
        # For now, return None and implement when needed
        return None
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        dataloaders: Dict[str, DataLoader],
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple models across multiple datasets.
        
        Args:
            models: Dictionary of {model_name: model}
            dataloaders: Dictionary of {dataset_name: dataloader}
            save_path: Path to save results
            
        Returns:
            Dictionary with comparison results
        """
        comparison_results = {
            'models': list(models.keys()),
            'datasets': list(dataloaders.keys()),
            'metrics': {}
        }
        
        for model_name, model in models.items():
            for dataset_name, dataloader in dataloaders.items():
                print(f"Evaluating {model_name} on {dataset_name}...")
                
                metrics = self.evaluate_model(
                    model, dataloader, model_name, dataset_name
                )
                
                key = f"{model_name}_{dataset_name}"
                comparison_results['metrics'][key] = {
                    'loss': metrics.loss,
                    'perplexity': metrics.perplexity,
                    'accuracy': metrics.accuracy,
                    'recall_accuracy': metrics.recall_accuracy,
                    'inference_time': metrics.inference_time,
                    'memory_usage': metrics.memory_usage,
                    'throughput': metrics.throughput,
                    'parameters': metrics.parameters
                }
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(comparison_results, f, indent=2)
        
        return comparison_results
    
    def create_comparison_plots(
        self,
        save_dir: str = './results/plots'
    ):
        """Create comparison plots from evaluation results."""
        os.makedirs(save_dir, exist_ok=True)
        
        if not self.results:
            print("No results to plot")
            return
        
        # Convert results to DataFrames for easy plotting
        import pandas as pd
        
        df = pd.DataFrame([
            {
                'Model': r.model_name,
                'Dataset': r.dataset_name,
                'Loss': r.loss,
                'Perplexity': r.perplexity,
                'Accuracy': r.accuracy,
                'Inference Time': r.inference_time,
                'Memory Usage': r.memory_usage,
                'Throughput': r.throughput,
                'Parameters': r.parameters
            }
            for r in self.results
        ])
        
        # Plot 1: Loss comparison
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='Dataset', y='Loss', hue='Model')
        plt.title('Model Comparison: Loss by Dataset')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'loss_comparison.png'))
        plt.close()
        
        # Plot 2: Perplexity comparison
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='Dataset', y='Perplexity', hue='Model')
        plt.title('Model Comparison: Perplexity by Dataset')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'perplexity_comparison.png'))
        plt.close()
        
        # Plot 3: Inference time comparison
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='Model', y='Inference Time')
        plt.title('Model Comparison: Inference Time')
        plt.ylabel('Time (seconds)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'inference_time_comparison.png'))
        plt.close()
        
        # Plot 4: Memory usage vs parameters
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='Parameters', y='Memory Usage', hue='Model', s=100)
        plt.title('Model Comparison: Memory Usage vs Parameters')
        plt.xlabel('Number of Parameters')
        plt.ylabel('Memory Usage (MB)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'memory_vs_parameters.png'))
        plt.close()
        
        # Plot 5: Throughput comparison
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='Model', y='Throughput')
        plt.title('Model Comparison: Throughput')
        plt.ylabel('Tokens per Second')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'throughput_comparison.png'))
        plt.close()
        
        print(f"Plots saved to {save_dir}")
    
    def generate_report(self, save_path: str = './results/evaluation_report.txt'):
        """Generate a text report of evaluation results."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            f.write("Griffin Model Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Group results by model and dataset
            models = set(r.model_name for r in self.results)
            datasets = set(r.dataset_name for r in self.results)
            
            for model in models:
                f.write(f"Model: {model}\n")
                f.write("-" * 30 + "\n")
                
                model_results = [r for r in self.results if r.model_name == model]
                
                for result in model_results:
                    f.write(f"  Dataset: {result.dataset_name}\n")
                    f.write(f"    Loss: {result.loss:.4f}\n")
                    f.write(f"    Perplexity: {result.perplexity:.4f}\n")
                    if result.accuracy is not None:
                        f.write(f"    Accuracy: {result.accuracy:.4f}\n")
                    f.write(f"    Inference Time: {result.inference_time:.4f}s\n")
                    f.write(f"    Memory Usage: {result.memory_usage:.2f}MB\n")
                    f.write(f"    Throughput: {result.throughput:.2f} tokens/s\n")
                    f.write(f"    Parameters: {result.parameters:,}\n")
                    f.write("\n")
                
                f.write("\n")
            
            # Summary comparison
            f.write("Summary Comparison\n")
            f.write("=" * 30 + "\n")
            
            for dataset in datasets:
                f.write(f"\nDataset: {dataset}\n")
                dataset_results = [r for r in self.results if r.dataset_name == dataset]
                
                # Best loss
                best_loss = min(dataset_results, key=lambda x: x.loss)
                f.write(f"  Best Loss: {best_loss.model_name} ({best_loss.loss:.4f})\n")
                
                # Best perplexity
                best_ppl = min(dataset_results, key=lambda x: x.perplexity)
                f.write(f"  Best Perplexity: {best_ppl.model_name} ({best_ppl.perplexity:.4f})\n")
                
                # Fastest inference
                fastest = min(dataset_results, key=lambda x: x.inference_time)
                f.write(f"  Fastest Inference: {fastest.model_name} ({fastest.inference_time:.4f}s)\n")
                
                # Highest throughput
                highest_throughput = max(dataset_results, key=lambda x: x.throughput)
                f.write(f"  Highest Throughput: {highest_throughput.model_name} ({highest_throughput.throughput:.2f} tokens/s)\n")
        
        print(f"Report saved to {save_path}")


if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator()
    
    # This would be used with actual trained models
    print("ModelEvaluator created successfully")
    print("Use evaluator.evaluate_model() to evaluate trained models")
    print("Use evaluator.compare_models() to compare multiple models")
    print("Use evaluator.create_comparison_plots() to generate visualizations")
