# Configuration settings for perception model training

import os
from datetime import datetime

# Path configurations
# These paths can be overridden via environment variables or command-line arguments
IMAGE_DIR = os.getenv('PP_IMAGE_DIR', './place-pulse/final_photo_dataset')
CLEAN_CSV_DIR = os.getenv('PP_CSV_DIR', './place-pulse/trueskill_perception_score_EuropeEurope_img_check.csv')
BASE_SAVE_PATH = os.getenv('PP_SAVE_PATH', './perception_module/output/')

# Training parameters
BATCH_SIZE = 16
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 100  # For early stopping
DATE = datetime.now().strftime('%Y%m%d%H%M')

# Model parameters
DEFAULT_PERCEPTION_DIM = 1

# Available backbone models
BACKBONES = {
    # Vision Transformers
    'vit_large_patch14_dinov2': {
        'name': 'vit_large_patch14_dinov2.lvd142m',
        'feature_dim': 1024,
        'input_size': (3, 518, 518),
        'interpolation': 'bicubic'
    },
    'vit_base_patch16': {
        'name': 'vit_base_patch16_224',
        'feature_dim': 768,
        'input_size': (3, 224, 224),
        'interpolation': 'bicubic'
    },
    
    # ResNets
    'resnet101': {
        'name': 'resnet101',
        'feature_dim': 2048,
        'input_size': (3, 224, 224),
        'interpolation': 'bilinear'
    },
    'resnet50': {
        'name': 'resnet50',
        'feature_dim': 2048,
        'input_size': (3, 224, 224),
        'interpolation': 'bilinear'
    },
    
    # EfficientNets
    'efficientnet_b0': {
        'name': 'efficientnet_b0',
        'feature_dim': 1280,
        'input_size': (3, 224, 224),
        'interpolation': 'bicubic'
    },
    'efficientnet_b4': {
        'name': 'efficientnet_b4',
        'feature_dim': 1792,
        'input_size': (3, 380, 380),
        'interpolation': 'bicubic'
    }
}

# Image normalization parameters (ImageNet stats)
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)
