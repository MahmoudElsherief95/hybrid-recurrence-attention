"""
Base Trainer Class for Griffin Model Experiments
Provides common training functionality for all model types
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import os
import time
import math
from typing import Dict, Any, Optional, Tuple, List
from tqdm import tqdm
import logging


class BaseTrainer:
    """
    Base trainer class for Griffin, Hawk, and Local Attention models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: Dict[str, Any],
        device: str = 'cuda',
        checkpoint_dir: str = './checkpoints',
        log_dir: str = './logs'
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        # Setup logging
        self._setup_logging()
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        
        # Setup tensorboard
        if config['logging'].get('tensorboard', True):
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Mixed precision training
        self.use_mixed_precision = config['hardware'].get('mixed_precision', False)
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        training_config = self.config['training']
        
        # Optimizer
        if training_config['optimizer'].lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=training_config['weight_decay']
            )
        elif training_config['optimizer'].lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=training_config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {training_config['optimizer']}")
        
        # Scheduler
        if training_config.get('scheduler') == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config['max_epochs']
            )
        elif training_config.get('scheduler') == 'linear':
            total_steps = len(self.train_dataloader) * training_config['max_epochs']
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=training_config.get('warmup_steps', 1000)
            )
        else:
            self.scheduler = None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.current_epoch + 1}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            self.optimizer.zero_grad()
            
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            # Forward pass
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    loss = self._compute_loss(logits, labels, attention_mask)
            else:
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                loss = self._compute_loss(logits, labels, attention_mask)
            
            # Backward pass
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config['training'].get('gradient_clip_norm'):
                    self.model.train()
                    total_loss = 0.0
                    total_tokens = 0
                    start_time = time.time()
                    cpu_mem_start = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                    gpu_mem_start = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                    progress_bar = tqdm(
                        self.train_dataloader,
                        desc=f"Epoch {self.current_epoch + 1}",
                        leave=False
                    )
                    for batch_idx, batch in enumerate(progress_bar):
                        self.optimizer.zero_grad()
                        # Move batch to device
                        input_ids = batch['input_ids'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        attention_mask = batch.get('attention_mask')
                        if attention_mask is not None:
                            attention_mask = attention_mask.to(self.device)
                        # Forward pass
                        if self.use_mixed_precision:
                            with torch.cuda.amp.autocast():
                                outputs = self.model(input_ids, attention_mask=attention_mask)
                                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                                loss = self._compute_loss(logits, labels, attention_mask)
                        else:
                            outputs = self.model(input_ids, attention_mask=attention_mask)
                            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                            loss = self._compute_loss(logits, labels, attention_mask)
                        # Backward pass
                        if self.use_mixed_precision:
                            self.scaler.scale(loss).backward()
                            # Gradient clipping
                            if self.config['training'].get('gradient_clip_norm'):
                                self.scaler.unscale_(self.optimizer)
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(),
                                    self.config['training']['gradient_clip_norm']
                                )
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            loss.backward()
                            # Gradient clipping
                            if self.config['training'].get('gradient_clip_norm'):
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(),
                                    self.config['training']['gradient_clip_norm']
                                )
                            self.optimizer.step()
                        # Update statistics
                        batch_size = input_ids.shape[0]
                        seq_len = input_ids.shape[1]
                        total_loss += loss.item() * batch_size
                        total_tokens += batch_size * seq_len
                        # Update progress bar
                        progress_bar.set_postfix({
                            'loss': f"{loss.item():.4f}",
                            'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                        })
                        # Log to tensorboard
                        if self.writer and self.global_step % self.config['logging']['log_interval'] == 0:
                            self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                            self.writer.add_scalar('Train/LearningRate', 
                                                 self.optimizer.param_groups[0]['lr'], self.global_step)
                            self.writer.add_scalar('Train/Perplexity', 
                                                 math.exp(min(loss.item(), 10)), self.global_step)
                        self.global_step += 1
                        # Validation
                        if (self.global_step % self.config['logging']['eval_interval'] == 0):
                            val_metrics = self.validate()
                            self._log_metrics(val_metrics, 'Val')
                            self.model.train()  # Return to training mode
                    avg_loss = total_loss / len(self.train_dataloader.dataset)
                    end_time = time.time()
                    cpu_mem_peak = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                    gpu_mem_peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
                    latency_sec_per_step = (end_time - start_time) / len(self.train_dataloader)
                    throughput_samples_per_sec = len(self.train_dataloader.dataset) / (end_time - start_time)
                    return {
                        'loss': avg_loss,
                        'perplexity': math.exp(min(avg_loss, 10)),
                        'latency_sec_per_step': latency_sec_per_step,
                        'throughput_samples_per_sec': throughput_samples_per_sec,
                        'cpu_mem_peak_mb': cpu_mem_peak / (1024 * 1024),
                        'gpu_mem_peak_mb': gpu_mem_peak / (1024 * 1024)
                    }
        avg_loss = total_loss / len(self.val_dataloader.dataset)
        return {'loss': avg_loss, 'perplexity': math.exp(min(avg_loss, 10))}
    
    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute the loss function."""
        # Reshape for cross entropy
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.view(-1, vocab_size)
        labels = labels.view(-1)
        
        # Compute loss
        loss = nn.functional.cross_entropy(logits, labels, reduction='none')
        
        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.view(-1).float()
            loss = loss * mask
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.mean()
        
        return loss
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str):
        """Log metrics to tensorboard and logger."""
        for key, value in metrics.items():
            if self.writer:
                self.writer.add_scalar(f'{prefix}/{key.capitalize()}', value, self.global_step)
        
        metrics_str = ', '.join([f"{key}: {value:.4f}" for key, value in metrics.items()])
        self.logger.info(f"{prefix} - Step {self.global_step} - {metrics_str}")
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        self.logger.info(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info(f"Training samples: {len(self.train_dataloader.dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_dataloader.dataset)}")
        
        max_epochs = self.config['training']['max_epochs']
        
        for epoch in range(self.current_epoch, max_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            self._log_metrics(train_metrics, 'Train')
            
            # Validate
            val_metrics = self.validate()
            self._log_metrics(val_metrics, 'Val')
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % self.config['logging']['save_interval'] == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint('best_model.pt', is_best=True)
            
            self.logger.info(f"Epoch {epoch + 1}/{max_epochs} completed")
        
        self.logger.info("Training completed!")
        
        if self.writer:
            self.writer.close()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_trainer(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    config_path: str,
    device: str = 'cuda'
) -> BaseTrainer:
    """Create trainer instance."""
    config = load_config(config_path)
    
    return BaseTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
        device=device
    )
