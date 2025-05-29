"""
Trainer module for MAMBA-TrackingBERT
"""
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Optional, Callable, List, Tuple
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import logging

from ..utils.metrics import MetricTracker


logger = logging.getLogger(__name__)


class BaseTrainer:
    """Base trainer class with common functionality"""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        config: Dict,
        scheduler: Optional[_LRScheduler] = None,
        scaler: Optional[GradScaler] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.scheduler = scheduler
        self.scaler = scaler
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        
        # Checkpointing
        self.checkpoint_dir = Path(config['training']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics
        self.train_metrics = MetricTracker()
        self.val_metrics = MetricTracker()
        
    def save_checkpoint(
        self,
        metrics: Dict[str, float],
        is_best: bool = False,
        filename: Optional[str] = None,
    ):
        """Save model checkpoint"""
        if filename is None:
            filename = f'checkpoint_epoch_{self.epoch}.pth'
            
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        logger.info(f"Loaded checkpoint from epoch {self.epoch}")
        return checkpoint['metrics']
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints keeping only the last N"""
        keep_last_n = self.config['training'].get('keep_last_n_checkpoints', 3)
        
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        if len(checkpoints) > keep_last_n:
            for ckpt in checkpoints[:-keep_last_n]:
                ckpt.unlink()
                logger.info(f"Removed old checkpoint: {ckpt}")


class MAMBATrainer(BaseTrainer):
    """Trainer specifically for MAMBA-TrackingBERT"""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        mdm_criterion: nn.Module,
        ntp_criterion: nn.Module,
        device: torch.device,
        config: Dict,
        scheduler: Optional[_LRScheduler] = None,
        scaler: Optional[GradScaler] = None,
        tokenizer=None,
    ):
        # Use MDM criterion as the main criterion
        super().__init__(model, optimizer, mdm_criterion, device, config, scheduler, scaler)
        
        self.ntp_criterion = ntp_criterion
        self.tokenizer = tokenizer
        
        # Task weights
        self.mdm_weight = config['training']['mdm_weight']
        self.ntp_weight = config['training']['ntp_weight']
        
        # Masking parameters
        self.current_mask_rate = config['training']['initial_mask_rate']
        self.mask_schedule = config['training']['mask_schedule_epochs']
        
        # Gradient accumulation
        self.accumulation_steps = config['training'].get('accumulation_steps', 1)
        
    def update_mask_rate(self):
        """Update masking rate based on current epoch"""
        config = self.config['training']
        
        if self.epoch < self.mask_schedule[0]:
            self.current_mask_rate = config['initial_mask_rate']
        elif self.epoch < self.mask_schedule[1]:
            self.current_mask_rate = config['intermediate_mask_rate']
        else:
            self.current_mask_rate = config['final_mask_rate']
            
        logger.info(f"Updated mask rate to {self.current_mask_rate}")
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        accumulation_counter = 0
        
        for batch_idx, batch in enumerate(pbar):
            # Alternate between tasks
            if batch_idx % 2 == 0:
                loss, metrics = self._train_mdm_step(batch)
                task = "mdm"
            else:
                loss, metrics = self._train_ntp_step(batch)
                task = "ntp"
            
            # Scale loss for gradient accumulation
            loss = loss / self.accumulation_steps
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            accumulation_counter += 1
            if accumulation_counter % self.accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    self._clip_gradients()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self._clip_gradients()
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Update metrics
            self.train_metrics.update(metrics, batch['input_ids'].size(0))
            
            # Update progress bar
            avg_metrics = self.train_metrics.average()
            pbar.set_postfix({
                'loss': avg_metrics.get('loss', 0),
                'mdm_acc': avg_metrics.get('mdm_accuracy', 0),
                'ntp_acc': avg_metrics.get('ntp_accuracy', 0),
            })
        
        return self.train_metrics.average()
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        self.val_metrics.reset()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
                if batch_idx % 2 == 0:
                    loss, metrics = self._validate_mdm_step(batch)
                else:
                    loss, metrics = self._validate_ntp_step(batch)
                
                self.val_metrics.update(metrics, batch['input_ids'].size(0))
        
        return self.val_metrics.average()
    
    def _train_mdm_step(self, batch):
        """Single training step for MDM task"""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # Create masked input
        masked_input_ids, masked_positions, labels = self._create_masked_batch(
            input_ids, self.current_mask_rate
        )
        
        # Skip if no valid masks
        if masked_positions.numel() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True), {'loss': 0.0, 'mdm_loss': 0.0, 'mdm_accuracy': 0.0}
        
        # Forward pass
        if self.scaler is not None:
            with autocast():
                outputs = self.model(
                    masked_input_ids,
                    attention_mask=attention_mask,
                    masked_positions=masked_positions,
                    task="mdm"
                )
                loss = self.criterion(outputs['mdm_logits'], labels)
        else:
            outputs = self.model(
                masked_input_ids,
                attention_mask=attention_mask,
                masked_positions=masked_positions,
                task="mdm"
            )
            loss = self.criterion(outputs['mdm_logits'], labels)
        
        # Calculate metrics
        with torch.no_grad():
            predictions = outputs['mdm_logits'].argmax(dim=-1)
            mask = labels != 0
            if mask.any():
                accuracy = (predictions == labels)[mask].float().mean().item()
            else:
                accuracy = 0.0
        
        metrics = {
            'loss': loss.item(),
            'mdm_loss': loss.item(),
            'mdm_accuracy': accuracy,
        }
        
        return loss * self.mdm_weight, metrics
    
    def _train_ntp_step(self, batch):
        """Single training step for NTP task"""
        input_ids = batch['input_ids'].to(self.device)
        momentum = batch['momentum'].to(self.device)  # Scalar momenta
        
        # Create track pairs
        batch_size = input_ids.size(0)
        indices = torch.randperm(batch_size)
        
        track1_ids = input_ids
        track2_ids = input_ids[indices]
        
        momentum1 = momentum
        momentum2 = momentum[indices]
        
        # Create labels (1 if track1 has higher momentum, 0 otherwise)
        labels = (momentum1 > momentum2).long()
        
        # Forward pass
        if self.scaler is not None:
            with autocast():
                outputs = self.model(
                    input_ids=None,
                    track_pair=(track1_ids, track2_ids),
                    task="ntp"
                )
                loss = self.ntp_criterion(outputs['ntp_logits'], labels)
        else:
            outputs = self.model(
                input_ids=None,
                track_pair=(track1_ids, track2_ids),
                task="ntp"
            )
            loss = self.ntp_criterion(outputs['ntp_logits'], labels)
        
        # Calculate metrics
        with torch.no_grad():
            predictions = outputs['ntp_logits'].argmax(dim=-1)
            accuracy = (predictions == labels).float().mean().item()
        
        metrics = {
            'loss': loss.item(),
            'ntp_loss': loss.item(),
            'ntp_accuracy': accuracy,
        }
        
        return loss * self.ntp_weight, metrics
    
    def _validate_mdm_step(self, batch):
        """Validation step for MDM task"""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # Use fixed mask rate for validation
        masked_input_ids, masked_positions, labels = self._create_masked_batch(
            input_ids, mask_rate=0.5
        )
        
        # Forward pass
        outputs = self.model(
            masked_input_ids,
            attention_mask=attention_mask,
            masked_positions=masked_positions,
            task="mdm"
        )
        loss = self.criterion(outputs['mdm_logits'], labels)
        
        # Calculate metrics
        predictions = outputs['mdm_logits'].argmax(dim=-1)
        mask = labels != 0
        accuracy = (predictions == labels)[mask].float().mean().item()
        
        metrics = {
            'val_loss': loss.item(),
            'val_mdm_loss': loss.item(),
            'val_mdm_accuracy': accuracy,
        }
        
        return loss, metrics
    
    def _validate_ntp_step(self, batch):
        """Validation step for NTP task"""
        # Similar to _train_ntp_step but without gradient computation
        return self._train_ntp_step(batch)
    
    def _create_masked_batch(self, input_ids, mask_rate):
        """Create masked input batch"""
        batch_size = input_ids.size(0)
        
        masked_input_ids = []
        masked_positions = []
        labels = []
        
        for i in range(batch_size):
            seq = input_ids[i].cpu().numpy()
            masked_seq, positions = self.tokenizer.create_masked_input(
                seq.tolist(),
                mask_rate=mask_rate
            )
            
            masked_input_ids.append(masked_seq)
            masked_positions.append(positions)
            
            # Get labels for masked positions
            label = [seq[pos] for pos in positions]
            labels.append(label)
        
        # Convert to tensors and pad
        masked_input_ids = torch.tensor(masked_input_ids, device=self.device)
        
        max_masked = max(len(pos) for pos in masked_positions)
        padded_positions = torch.zeros(batch_size, max_masked, dtype=torch.long, device=self.device)
        padded_labels = torch.zeros(batch_size, max_masked, dtype=torch.long, device=self.device)
        
        for i, (pos, lab) in enumerate(zip(masked_positions, labels)):
            padded_positions[i, :len(pos)] = torch.tensor(pos)
            padded_labels[i, :len(lab)] = torch.tensor(lab)
        
        return masked_input_ids, padded_positions, padded_labels
    
    def _clip_gradients(self):
        """Clip gradients if configured"""
        max_norm = self.config['training'].get('gradient_clip_val', 0)
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
    
    def train(self, train_loader, val_loader, num_epochs):
        """Main training loop"""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            self.update_mask_rate()
            
            # Training epoch
            train_start = time.time()
            train_metrics = self.train_epoch(train_loader)
            train_time = time.time() - train_start
            
            # Validation
            val_start = time.time()
            val_metrics = self.validate(val_loader)
            val_time = time.time() - val_start
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Logging
            logger.info(f"Epoch {epoch} - Train time: {train_time:.2f}s, Val time: {val_time:.2f}s")
            logger.info(f"Train metrics: {train_metrics}")
            logger.info(f"Val metrics: {val_metrics}")
            
            # Checkpointing
            is_best = val_metrics.get('val_loss', float('inf')) < self.best_metric
            if is_best:
                self.best_metric = val_metrics.get('val_loss', float('inf'))
            
            if epoch % self.config['training']['save_every_n_epochs'] == 0 or is_best:
                all_metrics = {**train_metrics, **val_metrics}
                self.save_checkpoint(all_metrics, is_best)
        
        logger.info("Training completed!")
        return self.best_metric
