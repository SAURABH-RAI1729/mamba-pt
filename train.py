#!/usr/bin/env python
"""
Training script for MAMBA-PT

Usage:
    python train.py --config configs/base_config.yaml
    python train.py --data_dir data/processed --batch_size 128 --learning_rate 0.001
"""
import argparse
import logging
import os
import random
import yaml
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler
import wandb
from torch.utils.tensorboard import SummaryWriter

from src.models.tracking_mamba import create_model
from src.models.tokenizer import DetectorTokenizer
from src.data.trackml_dataset import TrackMLDataModule
from src.training.trainer import MAMBATrainer
from src.training.losses import MDMLoss, NTPLoss


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(config: dict):
    """Setup logging and experiment tracking"""
    # TensorBoard
    writer = SummaryWriter(config['logging']['tensorboard_dir'])
    
    # Weights & Biases
    if config['logging']['wandb_project']:
        wandb.init(
            project=config['logging']['wandb_project'],
            entity=config['logging']['wandb_entity'],
            config=config,
            name=f"mamba-trackingbert-{config['seed']}"
        )
    
    return writer


def main():
    parser = argparse.ArgumentParser(description="Train MAMBA-TrackingBERT")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                        help="Path to configuration file")
    
    # Override config parameters
    parser.add_argument("--data_dir", type=str, help="Data directory")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs")
    parser.add_argument("--d_model", type=int, help="Model dimension")
    parser.add_argument("--n_layers", type=int, help="Number of MAMBA layers")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs
    if args.d_model:
        config['model']['d_model'] = args.d_model
    if args.n_layers:
        config['model']['n_layers'] = args.n_layers
    
    # Set random seed
    set_seed(config['seed'])
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Setup logging
    writer = setup_logging(config)
    
    # Initialize tokenizer
    tokenizer_path = Path(config['data']['data_dir']) / 'tokenizer'
    if tokenizer_path.exists():
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = DetectorTokenizer.load(tokenizer_path)
    else:
        logger.info("Creating new tokenizer")
        tokenizer = DetectorTokenizer(vocab_size=config['model']['vocab_size'])
    
    # Initialize data module
    data_module = TrackMLDataModule(
        config['data']['data_dir'],
        tokenizer,
        config,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
    )
    
    logger.info("Setting up datasets...")
    data_module.setup()
    
    # Save tokenizer if newly created
    if not tokenizer_path.exists():
        logger.info(f"Saving tokenizer to {tokenizer_path}")
        tokenizer.save(tokenizer_path)
    
    # Initialize model
    logger.info("Creating model...")
    model = create_model(config, tokenizer).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=(config['training']['adam_beta1'], config['training']['adam_beta2']),
        eps=config['training']['adam_epsilon'],
    )
    
    # Initialize scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=config['training']['min_lr'],
    )
    
    # Initialize loss functions
    mdm_criterion = MDMLoss(label_smoothing=config['training']['label_smoothing'])
    ntp_criterion = NTPLoss()
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler() if config['training']['amp_enabled'] else None
    
    # Initialize trainer
    trainer = MAMBATrainer(
        model=model,
        optimizer=optimizer,
        mdm_criterion=mdm_criterion,
        ntp_criterion=ntp_criterion,
        device=device,
        config=config,
        scheduler=scheduler,
        scaler=scaler,
        tokenizer=tokenizer,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(Path(args.resume))
    
    # Get data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Train model
    logger.info("Starting training...")
    try:
        best_metric = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['training']['num_epochs']
        )
        
        logger.info(f"Training completed! Best validation metric: {best_metric:.4f}")
        
        # Save final model
        final_checkpoint_path = trainer.checkpoint_dir / 'final_model.pth'
        trainer.save_checkpoint(
            metrics={'final_best_metric': best_metric},
            filename='final_model.pth'
        )
        logger.info(f"Saved final model to {final_checkpoint_path}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise
    finally:
        # Cleanup
        writer.close()
        if config['logging']['wandb_project']:
            wandb.finish()


if __name__ == "__main__":
    main()
