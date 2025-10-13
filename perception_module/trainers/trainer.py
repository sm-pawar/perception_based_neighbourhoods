"""
Training and evaluation utilities for perception models
"""

import os
import logging
import torch
import numpy as np
from datetime import datetime
import torch.distributed as dist

def train_step(model, train_dataloader, loss_fn, optimizer, scaler=None):
    """
    Run one epoch of training
    
    Args:
        model (nn.Module): The model
        train_dataloader (DataLoader): Training data loader
        loss_fn (nn.Module): Loss function
        optimizer (Optimizer): Optimizer
        scaler (GradScaler, optional): Gradient scaler for mixed precision
        
    Returns:
        float: Average training loss
    """
    model.train()
    running_loss = 0
    
    for data, target, sigma in train_dataloader:
        # Synchronize before moving data to GPU if using DDP
        if dist.is_initialized():
            torch.cuda.synchronize()
            
        # Move to GPU
        train_x = data.cuda(non_blocking=True)
        y = target.cuda(non_blocking=True)
        sigmas = sigma.cuda(non_blocking=True)
        
        # Forward pass with mixed precision if scaler is provided
        if scaler is not None:
            with torch.autocast(device_type="cuda"):
                output = model(train_x)
                loss = loss_fn(output, y.unsqueeze(1), sigmas.unsqueeze(1))
                
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular forward and backward pass
            output = model(train_x)
            loss = loss_fn(output, y.unsqueeze(1), sigmas.unsqueeze(1))
            
            loss.backward()
            optimizer.step()
            
        optimizer.zero_grad()
        running_loss += loss.item()
        
        # Synchronize after update if using DDP
        if dist.is_initialized():
            torch.cuda.synchronize()
    
    return running_loss / len(train_dataloader)


def evaluate(model, dataloader, loss_fn):
    """
    Evaluate the model on a dataset
    
    Args:
        model (nn.Module): The model
        dataloader (DataLoader): Data loader for evaluation
        loss_fn (nn.Module): Loss function
        
    Returns:
        tuple: (loss, pearson_correlation)
    """
    model.eval()
    running_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, y, sigma in dataloader:
            # Move to GPU
            test_x = data.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            sigmas = sigma.cuda(non_blocking=True)

            # Forward pass with mixed precision
            with torch.autocast(device_type="cuda", enabled=True):
                output = model(test_x)
                loss = loss_fn(output, y.unsqueeze(1), sigmas.unsqueeze(1))
            
            running_loss += loss.item()
            
            # Collect predictions and targets for correlation
            all_predictions.extend(output.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    # Calculate Pearson correlation
    pearson_corr = np.corrcoef(
        np.array(all_predictions).flatten(), 
        np.array(all_targets).flatten()
    )[0, 1]

    return running_loss / len(dataloader), pearson_corr


def save_checkpoint(model, optimizer, epoch, val_loss, save_path, is_best=False):
    """
    Save model checkpoint
    
    Args:
        model (nn.Module): The model
        optimizer (Optimizer): Optimizer
        epoch (int): Current epoch
        val_loss (float): Validation loss
        save_path (str): Path to save checkpoint
        is_best (bool): Whether this is the best model so far
    """
    # Extract the model state (handle DDP)
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
        
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        # Save a copy as the best model
        best_path = os.path.join(os.path.dirname(save_path), 'best_model.pt')
        torch.save(model_state, best_path)
        logging.info(f"Saved new best model with val loss: {val_loss:.4f}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load model from checkpoint
    
    Args:
        model (nn.Module): The model
        optimizer (Optimizer): Optimizer
        checkpoint_path (str): Path to checkpoint
        
    Returns:
        tuple: (model, optimizer, epoch, val_loss)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    
    # Handle DDP model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
    
    return model, optimizer, epoch, val_loss
