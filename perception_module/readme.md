# Perception Model

This is a modular implementation of a deep learning system for predicting urban perception scores from street view images. The system supports multiple backbone architectures including Vision Transformers, ResNets, and EfficientNets.

## Project Structure

```
perception_module/
├── config.py               # Configuration settings
├── data/
│   ├── __init__.py
│   ├── dataset.py          # Dataset and preprocessing
│   └── utils.py            # Data processing utilities
├── models/
│   ├── __init__.py
│   ├── backbones.py        # Model backbone implementations
│   ├── model.py            # PerceptionModel architecture
│   └── loss.py             # Custom loss functions
├── trainers/
│   ├── __init__.py
│   └── trainer.py          # Training/validation loops
├── utils/
│   ├── __init__.py
│   └── misc.py             # Miscellaneous utility functions
└── train.py                # Main training script
```

## Features

- Support for multiple backbone architectures:
  - Vision Transformers (ViT)
  - ResNet models (ResNet50, ResNet101)
  - EfficientNet models (B0, B4)
- Distributed training with PyTorch DDP
- Mixed precision training for faster training
- Early stopping and model checkpointing
- Weighted loss function based on confidence scores
- Modular design for easy extension with new architectures

## Usage

### Basic Training

Train a model with default settings for a specific perception type:

```bash
python -m perception_module.train --perception beautiful --backbone vit_large_patch14_dinov2
```

### Training with Different Backbones

Train with a ResNet101 backbone:

```bash
python -m perception_module.train --perception beautiful --backbone resnet101
```

Train with an EfficientNet backbone:

```bash
python -m perception_module.train --perception beautiful --backbone efficientnet_b4
```

### Distributed Training

Train using multiple GPUs with distributed data parallel:

```bash
python -m torch.distributed.launch --nproc_per_node=4 -m perception_module.train \
    --perception beautiful --backbone resnet101
```

### Additional Options

- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 1e-4)
- `--weight-decay`: Weight decay for regularization (default: 1e-5)
- `--freeze-backbone`: Freeze backbone weights and train only the regression head
- `--unfreeze-after`: Epoch after which to unfreeze the backbone
- `--pretrained-weights`: Path to pretrained weights
- `--patience`: Number of epochs to wait before early stopping (default: 100)
- `--seed`: Random seed for reproducibility (default: 42)

Example with multiple options:

```bash
python -m perception_module.train \
    --perception beautiful \
    --backbone efficientnet_b4 \
    --epochs 150 \
    --batch-size 32 \
    --lr 5e-5 \
    --freeze-backbone \
    --unfreeze-after 50 \
    --pretrained-weights /path/to/pretrained/model.pt
```

## Adding New Backbones

To add a new backbone architecture:

1. Add the backbone configuration to `BACKBONES` dictionary in `config.py`
2. If necessary, add implementation details in `models/backbones.py`

## Dependencies

- PyTorch (>=1.8.0)
- torchvision
- timm
- numpy
- pandas
- Pillow
- scikit-learn

## Acknowledgments

This project is based on research in using deep learning for urban perception prediction.
