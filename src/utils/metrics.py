"""
Metrics for evaluating MAMBA-TrackingBERT
"""
import torch
import numpy as np
from typing import Tuple, Dict


def calculate_accuracy(predictions: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor = None) -> float:
    """
    Calculate accuracy for predictions
    
    Args:
        predictions: Predicted values
        labels: True labels
        mask: Optional mask for valid positions
    
    Returns:
        Accuracy as float
    """
    if mask is not None:
        predictions = predictions[mask]
        labels = labels[mask]
    
    if len(labels) == 0:
        return 0.0
    
    correct = (predictions == labels).float().sum()
    total = len(labels)
    
    return (correct / total).item()


def calculate_distance_accuracy(
    predicted_modules: np.ndarray,
    true_modules: np.ndarray,
    module_positions: Dict[Tuple[int, int, int], np.ndarray],
    distance_threshold: float = 20.0,  # mm
) -> float:
    """
    Calculate accuracy based on physical distance between predicted and true modules
    
    Args:
        predicted_modules: Predicted module IDs
        true_modules: True module IDs
        module_positions: Dictionary mapping module tuples to 3D positions
        distance_threshold: Maximum distance to consider prediction correct
    
    Returns:
        Distance-based accuracy
    """
    if len(predicted_modules) == 0:
        return 0.0
    
    correct = 0
    total = 0
    
    for pred, true in zip(predicted_modules, true_modules):
        if pred is None or true is None:
            continue
            
        pred_pos = module_positions.get(pred)
        true_pos = module_positions.get(true)
        
        if pred_pos is None or true_pos is None:
            continue
        
        distance = np.linalg.norm(pred_pos - true_pos)
        
        if distance <= distance_threshold:
            correct += 1
        
        total += 1
    
    return correct / total if total > 0 else 0.0


def calculate_mdm_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tokenizer,
    module_positions: Dict = None,
) -> Dict[str, float]:
    """
    Calculate all metrics for MDM task
    
    Args:
        logits: Model predictions (batch, seq, vocab)
        labels: True labels (batch, seq)
        tokenizer: Tokenizer for decoding
        module_positions: Optional position information for distance metric
    
    Returns:
        Dictionary of metrics
    """
    predictions = logits.argmax(dim=-1)
    
    # Mask out padding
    mask = labels != 0
    
    # Token accuracy
    token_accuracy = calculate_accuracy(predictions, labels, mask)
    
    # Per-sequence accuracy (all tokens correct)
    batch_correct = []
    for pred_seq, true_seq, seq_mask in zip(predictions, labels, mask):
        seq_pred = pred_seq[seq_mask]
        seq_true = true_seq[seq_mask]
        all_correct = (seq_pred == seq_true).all().item()
        batch_correct.append(all_correct)
    
    sequence_accuracy = np.mean(batch_correct)
    
    metrics = {
        'token_accuracy': token_accuracy,
        'sequence_accuracy': sequence_accuracy,
    }
    
    # Distance accuracy if module positions provided
    if module_positions is not None:
        distance_accuracies = []
        
        for pred_seq, true_seq, seq_mask in zip(predictions, labels, mask):
            pred_modules = tokenizer.decode_modules(pred_seq[seq_mask].cpu().numpy())
            true_modules = tokenizer.decode_modules(true_seq[seq_mask].cpu().numpy())
            
            dist_acc = calculate_distance_accuracy(
                pred_modules,
                true_modules,
                module_positions
            )
            distance_accuracies.append(dist_acc)
        
        metrics['distance_accuracy'] = np.mean(distance_accuracies)
    
    return metrics


def calculate_ntp_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    """
    Calculate metrics for NTP task
    
    Args:
        logits: Model predictions (batch, 2)
        labels: True labels (batch,)
    
    Returns:
        Dictionary of metrics
    """
    predictions = logits.argmax(dim=-1)
    
    accuracy = calculate_accuracy(predictions, labels)
    
    # Calculate precision, recall, F1
    tp = ((predictions == 1) & (labels == 1)).sum().item()
    fp = ((predictions == 1) & (labels == 0)).sum().item()
    fn = ((predictions == 0) & (labels == 1)).sum().item()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


class MetricTracker:
    """Track and average metrics over multiple batches"""
    
    def __init__(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, metrics: Dict[str, float], batch_size: int = 1):
        """Update metrics with new batch"""
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = 0.0
                self.counts[name] = 0
            
            self.metrics[name] += value * batch_size
            self.counts[name] += batch_size
    
    def average(self) -> Dict[str, float]:
        """Get averaged metrics"""
        averaged = {}
        for name in self.metrics:
            if self.counts[name] > 0:
                averaged[name] = self.metrics[name] / self.counts[name]
            else:
                averaged[name] = 0.0
        return averaged
    
    def reset(self):
        """Reset all metrics"""
        self.metrics = {}
        self.counts = {}
