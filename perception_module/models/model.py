"""
PerceptionModel architecture and related utilities
"""

import torch
import torch.nn as nn
import logging

from .backbones import get_backbone

class PerceptionModel(nn.Module):
    """
    Model for predicting perception scores from images
    """
    def __init__(self, backbone_name='vit_large_patch14_dinov2', pretrained=True, perception_dim=1):
        """
        Initialize the perception model
        
        Args:
            backbone_name (str): Name of the backbone model
            pretrained (bool): Whether to use pretrained weights
            perception_dim (int): Dimension of perception output
        """
        super(PerceptionModel, self).__init__()
        
        # Load backbone
        self.backbone, self.feature_dim = get_backbone(backbone_name, pretrained)
        
        # Custom regression head
        self.regression_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, perception_dim)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 3, H, W]
            
        Returns:
            torch.Tensor: Perception score
        """
        # Extract features using backbone
        features = self.backbone(x)
        
        # Handle different feature shapes returned by different backbones
        if len(features.shape) > 2:
            # For architectures that return feature maps, flatten them
            features = torch.mean(features, dim=[2, 3])
        
        # Apply regression head
        perception_score = self.regression_head(features)
        return perception_score

    def get_embeddings_and_predictions(self, x):
        features = self.backbone(x)
        
        # Handle different feature shapes returned by different backbones
        if len(features.shape) > 2:
            # For architectures that return feature maps, flatten them
            features = torch.mean(features, dim=[2, 3])

        # Process through regression head layers to get embeddings
        x = self.regression_head[0](features)  # Linear
        x = self.regression_head[1](x)  # BatchNorm
        x = self.regression_head[2](x)  # ReLU
        x = self.regression_head[3](x)  # Dropout
        x = self.regression_head[4](x)  # Linear
        x = self.regression_head[5](x)  # BatchNorm
        embeddings = self.regression_head[6](x)  # ReLU
        
        # Continue through remaining layers for final prediction
        x = self.regression_head[7](embeddings)  # Dropout
        predictions = self.regression_head[8](x)  # Final Linear layer
        
        return embeddings, predictions

def load_compatible_weights(new_model, old_state_dict):
    """
    Load weights from a previous model when architectures might differ
    
    Args:
        new_model (nn.Module): The new model to load weights into
        old_state_dict (dict): State dict from old model
        
    Returns:
        nn.Module: The model with compatible weights loaded
    """
    new_state_dict = new_model.state_dict()
    
    # Count compatible parameters
    compatible_params = 0
    total_params = len(new_state_dict)
    
    for name, param in old_state_dict.items():
        if name in new_state_dict and new_state_dict[name].shape == param.shape:
            new_state_dict[name] = param
            compatible_params += 1
    
    logging.info(f"Loaded {compatible_params}/{total_params} compatible parameters")
    
    # Load the updated state dict
    new_model.load_state_dict(new_state_dict)
    return new_model


def get_optimizer_scheduler(model, learning_rate=1e-4, weight_decay=1e-5):
    """
    Create optimizer and learning rate scheduler
    
    Args:
        model (nn.Module): The model
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay
        
    Returns:
        tuple: (optimizer, scheduler)
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)
    
    return optimizer, scheduler


def freeze_backbone(model, learning_rate=1e-4):
    """
    Freeze the backbone and train only the regression head
    
    Args:
        model (nn.Module): The model
        learning_rate (float): Learning rate
        
    Returns:
        tuple: (optimizer, scheduler)
    """
    # Freeze backbone parameters
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # Optimize only regression head
    optimizer = torch.optim.Adam(model.regression_head.parameters(), lr=learning_rate)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    return optimizer, scheduler


def unfreeze_backbone(model, learning_rate=1e-4):
    """
    Unfreeze the backbone with a lower learning rate
    
    Args:
        model (nn.Module): The model
        learning_rate (float): Learning rate
        
    Returns:
        tuple: (optimizer, scheduler)
    """
    # Unfreeze backbone parameters
    for param in model.backbone.parameters():
        param.requires_grad = True
    
    # Use different learning rates for backbone and head
    optimizer = torch.optim.Adam([
        {'params': model.regression_head.parameters(), 'lr': learning_rate},
        {'params': model.backbone.parameters(), 'lr': learning_rate * 0.1}
    ])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    return optimizer, scheduler
