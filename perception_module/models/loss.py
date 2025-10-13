"""
Custom loss functions for perception models
"""

import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    """
    Weighted MSE Loss that uses sigma values (confidence) as weights
    Higher confidence (lower sigma) = higher weight
    """
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
        
    def forward(self, predictions, targets, sigmas):
        # Convert to float just in case
        predictions = predictions.float()
        targets = targets.float()
        sigmas = sigmas.float()
        
        # Higher confidence (lower sigma) = higher weight
        weights = 1.0 / (sigmas + 1e-6)  # Add small epsilon to avoid division by zero
        weights = weights / weights.sum()  # Normalize weights
        
        # Weighted MSE
        squared_errors = (predictions - targets) ** 2
        weighted_errors = weights * squared_errors
        
        return weighted_errors.mean()
