"""
Loss functions for MAMBA-TrackingBERT
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MDMLoss(nn.Module):
    """Loss function for Masked Detector Module task"""
    
    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.label_smoothing = label_smoothing
        
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, seq_len, vocab_size)
            labels: (batch, seq_len)
        Returns:
            loss: scalar tensor
        """
        # Flatten the logits and labels
        logits = logits.reshape(-1, logits.size(-1))
        labels = labels.reshape(-1)
        
        # Ignore padding tokens (label == 0)
        mask = labels != 0
        logits = logits[mask]
        labels = labels[mask]
        
        if len(labels) == 0:
            return torch.tensor(0.0, device=logits.device)
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            vocab_size = logits.size(-1)
            smoothed_labels = torch.zeros_like(logits)
            smoothed_labels.fill_(self.label_smoothing / (vocab_size - 1))
            smoothed_labels.scatter_(1, labels.unsqueeze(1), 1.0 - self.label_smoothing)
            
            # Use KL divergence loss
            log_probs = F.log_softmax(logits, dim=-1)
            loss = F.kl_div(log_probs, smoothed_labels, reduction='batchmean')
        else:
            # Standard cross entropy
            loss = F.cross_entropy(logits, labels)
        
        return loss


class NTPLoss(nn.Module):
    """Loss function for Next Track Prediction task"""
    
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, 2) - binary classification
            labels: (batch,) - 0 or 1
        Returns:
            loss: scalar tensor
        """
        return self.criterion(logits, labels)


class CombinedLoss(nn.Module):
    """Combined loss for both tasks"""
    
    def __init__(
        self,
        mdm_weight: float = 1.0,
        ntp_weight: float = 0.5,
        label_smoothing: float = 0.1
    ):
        super().__init__()
        self.mdm_weight = mdm_weight
        self.ntp_weight = ntp_weight
        self.mdm_loss = MDMLoss(label_smoothing=label_smoothing)
        self.ntp_loss = NTPLoss()
        
    def forward(
        self,
        mdm_logits: torch.Tensor = None,
        mdm_labels: torch.Tensor = None,
        ntp_logits: torch.Tensor = None,
        ntp_labels: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Combined loss calculation
        
        Returns:
            Total weighted loss
        """
        total_loss = 0
        losses = {}
        
        if mdm_logits is not None and mdm_labels is not None:
            mdm_loss = self.mdm_loss(mdm_logits, mdm_labels)
            total_loss += self.mdm_weight * mdm_loss
            losses['mdm_loss'] = mdm_loss
            
        if ntp_logits is not None and ntp_labels is not None:
            ntp_loss = self.ntp_loss(ntp_logits, ntp_labels)
            total_loss += self.ntp_weight * ntp_loss
            losses['ntp_loss'] = ntp_loss
            
        losses['total_loss'] = total_loss
        
        return total_loss, losses

