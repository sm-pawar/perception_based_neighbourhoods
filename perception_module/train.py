#!/usr/bin/env python
"""
Main training script for perception models
"""

import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler
from datetime import datetime

from config import (
    IMAGE_DIR, CLEAN_CSV_DIR, BASE_SAVE_PATH,
    BATCH_SIZE, NUM_WORKERS, LEARNING_RATE, WEIGHT_DECAY,
    PATIENCE, DATE, BACKBONES, DEFAULT_PERCEPTION_DIM
)
from data import pp_process_input, create_dataloaders
from models import (
    PerceptionModel, WeightedMSELoss, 
    load_compatible_weights, 
    get_optimizer_scheduler
)
from trainers import train_step, evaluate, save_checkpoint
from utils import setup_logging, setup_device, set_seed, get_save_paths


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train perception models with different backbones')
    
    parser.add_argument('--perception', type=str, required=True, 
                        help='Perception type to train for (e.g., "beautiful", "lively", etc.)')
    
    parser.add_argument('--backbone', type=str, default='vit_large_patch14_dinov2', 
                        choices=list(BACKBONES.keys()),
                        help='Backbone architecture to use')
    
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of training epochs')
    
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, 
                        help=f'Batch size (default: {BATCH_SIZE})')
    
    parser.add_argument('--workers', type=int, default=NUM_WORKERS, 
                        help=f'Number of data loading workers (default: {NUM_WORKERS})')
    
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, 
                        help=f'Learning rate (default: {LEARNING_RATE})')
    
    parser.add_argument('--weight-decay', type=float, default=WEIGHT_DECAY, 
                        help=f'Weight decay (default: {WEIGHT_DECAY})')
    
    parser.add_argument('--patience', type=int, default=PATIENCE, 
                        help=f'Patience for early stopping (default: {PATIENCE})')
    
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed (default: 42)')
    
    parser.add_argument('--pretrained-weights', type=str, default=None, 
                        help='Path to pretrained weights file (optional)')
    
    parser.add_argument('--freeze-backbone', action='store_true', 
                        help='Freeze backbone weights and train only the regression head')
    
    parser.add_argument('--unfreeze-after', type=int, default=None, 
                        help='Epoch after which to unfreeze the backbone (if frozen)')
    
    # Distributed training arguments
    parser.add_argument('--local-rank', '--local_rank', type=int, default=None, 
                        help='Local rank for distributed training (set automatically)')
    
    # Date for output directory naming
    parser.add_argument('--date', type=str, default=DATE,
                        help=f'Date string for output directory names (default: {DATE})')
    
    return parser.parse_args()


def train(args):

    """Main training function"""
    # Setup device and distributed training if applicable
    #device, is_distributed = setup_device(int(os.environ["LOCAL_RANK"]))
    device, is_distributed = setup_device()
    
    # Verbose flag (only print from rank 0 in distributed mode)
    verbose = not is_distributed or dist.get_rank() == 0
    
    print(f"Using device: {device}, Distributed: {is_distributed}, Verbose: {verbose}")


    # Set random seed
    set_seed(args.seed)
    
    # Setup logging
    if verbose:
        logger = setup_logging(BASE_SAVE_PATH, args.perception, args.date)
        logger.info(f"Starting training for perception: {args.perception}")
        logger.info(f"Using backbone: {args.backbone}")
        logger.info(f"Device: {device}, Distributed: {is_distributed}")
    
    # Get model save paths
    save_paths = get_save_paths(BASE_SAVE_PATH, args.perception, args.backbone, args.date)
    
    # Process input data
    if verbose:
        logging.info("Loading and processing data...")
    
    df_train, df_val, df_test = pp_process_input(CLEAN_CSV_DIR, args.perception)
    
    # Create data loaders
    model_config = BACKBONES[args.backbone]
    train_loader, val_loader, test_loader, train_sampler = create_dataloaders(
        df_train, df_val, df_test,
        IMAGE_DIR, 
        model_config,
        args.batch_size,
        args.workers,
        is_distributed
    )
    
    if verbose:
        logging.info(f"Created data loaders: {len(train_loader)} training batches, "
                    f"{len(val_loader)} validation batches, {len(test_loader)} test batches")
    
    # Create model
    model = PerceptionModel(
        backbone_name=args.backbone,
        pretrained=True,
        perception_dim=DEFAULT_PERCEPTION_DIM
    ).to(device)
    
    # Load pretrained weights if specified
    if args.pretrained_weights is not None and os.path.exists(args.pretrained_weights):
        if verbose:
            logging.info(f"Loading pretrained weights from {args.pretrained_weights}")
        
        try:
            pretrained_state = torch.load(args.pretrained_weights, map_location=device)
            model = load_compatible_weights(model, pretrained_state)
        except Exception as e:
            if verbose:
                logging.error(f"Error loading pretrained weights: {e}")
    
    # Wrap model with DDP for distributed training
    if is_distributed:
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True) # find_unused_parameters=True is important for models with frozen layers
    
    # Create loss function, optimizer, and scheduler
    loss_fn = WeightedMSELoss()
    
    # Freeze backbone if requested
    if args.freeze_backbone:
        if verbose:
            logging.info("Freezing backbone, training only regression head")
        
        for param in model.module.backbone.parameters() if is_distributed else model.backbone.parameters():
            param.requires_grad = False
    
    # Create optimizer and scheduler
    optimizer, scheduler = get_optimizer_scheduler(
        model.module if is_distributed else model,
        args.lr,
        args.weight_decay
    )
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler('cuda')
    
    # Training loop variables
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    patience_counter = 0
    
    # Start timing
    start_time = datetime.now()
    if verbose:
        logging.info(f"Training started at {start_time}")
    
    for epoch in range(args.epochs):
        # Set epoch for distributed sampler
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Unfreeze backbone if requested
        if args.freeze_backbone and args.unfreeze_after is not None and epoch == args.unfreeze_after:
            if verbose:
                logging.info(f"Unfreezing backbone at epoch {epoch}")
            
            # Enable gradients for backbone
            for param in model.module.backbone.parameters() if is_distributed else model.backbone.parameters():
                param.requires_grad = True
            
            # Create new optimizer with different learning rates
            optimizer = torch.optim.AdamW([
                {'params': (model.module if is_distributed else model).regression_head.parameters(), 'lr': args.lr},
                {'params': (model.module if is_distributed else model).backbone.parameters(), 'lr': args.lr * 0.1}
            ], weight_decay=args.weight_decay)
            
            # Create new scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3)
        
        # Train one epoch
        train_loss = train_step(model, train_loader, loss_fn, optimizer, scaler)
        
        # Evaluate
        val_loss, person_r = evaluate(model, val_loader, loss_fn)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Log progress
        if verbose:
            logging.info(f"Epoch: {epoch+1}/{args.epochs}, "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}, "
                        f"Val Pearson R: {person_r:.4f}")
            
            # Check for best validation model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best validation model
                if isinstance(model, DDP):
                    torch.save(model.module.state_dict(), save_paths['best_val_model'])
                else:
                    torch.save(model.state_dict(), save_paths['best_val_model'])
                
                logging.info(f"New best validation model saved with loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
            
            # Check for best training model
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                
                # Save best training model
                if isinstance(model, DDP):
                    torch.save(model.module.state_dict(), save_paths['best_train_model'])
                else:
                    torch.save(model.state_dict(), save_paths['best_train_model'])
                
                logging.info(f"New best training model saved with loss: {best_train_loss:.4f}")
            
            # Save latest checkpoint
            save_checkpoint(
                model, 
                optimizer, 
                epoch, 
                val_loss, 
                save_paths['latest_checkpoint']
            )
            
            # Check for early stopping
            if patience_counter >= args.patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Training completed
    end_time = datetime.now()
    if verbose:
        logging.info(f"Training completed in: {end_time - start_time}")
        
        # Final evaluation on test set
        logging.info("Evaluating on test set...")
        
        # Test with best validation model
        try:
            if isinstance(model, DDP):
                model.module.load_state_dict(torch.load(save_paths['best_val_model']))
            else:
                model.load_state_dict(torch.load(save_paths['best_val_model']))
            
            test_loss, test_r = evaluate(model, test_loader, loss_fn)
            logging.info(f"Test results with best validation model: "
                        f"Loss: {test_loss:.4f}, Pearson R: {test_r:.4f}")
            
            # Test with best training model
            if isinstance(model, DDP):
                model.module.load_state_dict(torch.load(save_paths['best_train_model']))
            else:
                model.load_state_dict(torch.load(save_paths['best_train_model']))
            
            test_loss, test_r = evaluate(model, test_loader, loss_fn)
            logging.info(f"Test results with best training model: "
                        f"Loss: {test_loss:.4f}, Pearson R: {test_r:.4f}")
        except Exception as e:
            logging.error(f"Error during final evaluation: {e}")
    
    # Clean up (for distributed training)
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    train(args)
