"""
Model backbone implementations for different architectures
"""

import torch
import timm
import torch.nn as nn
import torchvision.models as models
import logging

def get_backbone(backbone_name, pretrained=True):
    """
    Create a backbone model based on the specified name
    
    Args:
        backbone_name (str): Name of the backbone model
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        tuple: (backbone_model, feature_dim)
    """
    from perception_module.config import BACKBONES
    
    if backbone_name not in BACKBONES:
        raise ValueError(f"Unknown backbone: {backbone_name}. Available backbones: {list(BACKBONES.keys())}")
    
    config = BACKBONES[backbone_name]
    model_name = config['name']
    feature_dim = config['feature_dim']
    
    logging.info(f"Creating backbone: {backbone_name} ({model_name})")
    
    # Load the model from timm
    try:
        backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove classifier to get features
        )
        return backbone, feature_dim
    except Exception as e:
        logging.error(f"Error creating model with timm: {e}")
        
        # Fallback to torchvision if timm fails
        if "resnet" in backbone_name:
            if backbone_name == "resnet50":
                backbone = models.resnet50(pretrained=pretrained)
            elif backbone_name == "resnet101":
                backbone = models.resnet101(pretrained=pretrained)
            else:
                raise ValueError(f"Unsupported ResNet variant: {backbone_name}")
                
            # Remove the final fully connected layer
            backbone = nn.Sequential(*list(backbone.children())[:-1])
            
        elif "efficientnet" in backbone_name:
            if backbone_name == "efficientnet_b0":
                backbone = models.efficientnet_b0(pretrained=pretrained)
            elif backbone_name == "efficientnet_b4":
                backbone = models.efficientnet_b4(pretrained=pretrained)
            else:
                raise ValueError(f"Unsupported EfficientNet variant: {backbone_name}")
                
            # Get the features without classifier
            backbone = nn.Sequential(*list(backbone.children())[:-1])
            
        else:
            raise ValueError(f"Failed to create model {backbone_name} with both timm and torchvision")
            
        return backbone, feature_dim
