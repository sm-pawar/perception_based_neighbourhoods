#!/usr/bin/env python
"""
Prediction and embedding extraction script for perception models
Supports both regular images and panorama images with configurable view extraction
"""

import argparse
import os
import glob
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from scipy.ndimage import map_coordinates

from perception_module.config import BACKBONES, BATCH_SIZE, NUM_WORKERS
from perception_module.models import PerceptionModel
from perception_module.data.utils import get_preprocessing_transforms


# ============================================================================
# Panorama Processing Functions
# ============================================================================

def map_to_sphere(x, y, z, yaw_radian, pitch_radian):
    """
    Map planar coordinates to spherical coordinates with rotation
    
    Args:
        x, y, z: Planar coordinates
        yaw_radian: Yaw angle in radians
        pitch_radian: Pitch angle in radians
        
    Returns:
        tuple: (theta, phi) spherical coordinates
    """
    theta = np.arccos(z / np.sqrt(x ** 2 + y ** 2 + z ** 2))
    phi = np.arctan2(y, x)

    # Apply rotation transformations
    theta_prime = np.arccos(np.sin(theta) * np.sin(phi) * np.sin(pitch_radian) +
                            np.cos(theta) * np.cos(pitch_radian))

    phi_prime = np.arctan2(np.sin(theta) * np.sin(phi) * np.cos(pitch_radian) -
                            np.cos(theta) * np.sin(pitch_radian),
                            np.sin(theta) * np.cos(phi))
    phi_prime += yaw_radian
    phi_prime = phi_prime % (2 * np.pi)

    return theta_prime.flatten(), phi_prime.flatten()


def interpolate_color(coords, img, method='bilinear'):
    """
    Interpolate colors from panorama image
    
    Args:
        coords: Coordinate array
        img: Image array
        method: Interpolation method
        
    Returns:
        np.ndarray: Interpolated RGB values
    """
    order = {'nearest': 0, 'bilinear': 1, 'bicubic': 3}.get(method, 1)
    red = map_coordinates(img[:, :, 0], coords, order=order, mode='reflect')
    green = map_coordinates(img[:, :, 1], coords, order=order, mode='reflect')
    blue = map_coordinates(img[:, :, 2], coords, order=order, mode='reflect')
    return np.stack((red, green, blue), axis=-1)


def panorama_to_plane(panorama_path, FOV, output_size, yaw, pitch):
    """
    Extract a planar view from a panoramic image
    
    Args:
        panorama_path: Path to panorama image
        FOV: Field of view in degrees
        output_size: Tuple of (width, height) for output
        yaw: Yaw angle in degrees
        pitch: Pitch angle in degrees
        
    Returns:
        PIL.Image: Extracted planar view
    """
    panorama = Image.open(panorama_path).convert('RGB')
    pano_width, pano_height = panorama.size
    pano_array = np.array(panorama)
    yaw_radian = np.radians(yaw)
    pitch_radian = np.radians(pitch)

    W, H = output_size
    f = (0.5 * W) / np.tan(np.radians(FOV) / 2)

    u, v = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')

    x = u - W / 2
    y = H / 2 - v
    z = f

    theta, phi = map_to_sphere(x, y, z, yaw_radian, pitch_radian)

    U = phi * pano_width / (2 * np.pi)
    V = theta * pano_height / np.pi

    U, V = U.flatten(), V.flatten()
    coords = np.vstack((V, U))

    colors = interpolate_color(coords, pano_array)
    output_image = Image.fromarray(colors.reshape((H, W, 3)).astype('uint8'), 'RGB')

    return output_image


# ============================================================================
# Dataset Classes
# ============================================================================

class PredictionDataset(Dataset):
    """
    Dataset for making predictions on regular images
    """
    def __init__(self, image_paths, model_config):
        """
        Initialize prediction dataset
        
        Args:
            image_paths: List of image file paths
            model_config: Model configuration dict from BACKBONES
        """
        self.image_paths = image_paths
        self.model_config = model_config
        self.transform = get_preprocessing_transforms(model_config, is_training=False)
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
            # Return blank image on error
            target_size = self.model_config['input_size'][1]
            image = Image.new('RGB', (target_size, target_size))
        
        img_id = Path(img_path).stem
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_id


class PanoramaDataset(Dataset):
    """
    Dataset for making predictions on panorama images
    """
    def __init__(self, image_paths, model_config, fov=90, yaw=270, pitch=80):
        """
        Initialize panorama dataset
        
        Args:
            image_paths: List of panorama image file paths
            model_config: Model configuration dict from BACKBONES
            fov: Field of view in degrees
            yaw: Yaw angle in degrees
            pitch: Pitch angle in degrees
        """
        self.image_paths = image_paths
        self.model_config = model_config
        self.fov = fov
        self.yaw = yaw
        self.pitch = pitch
        
        # Use input size from model config
        target_size = model_config['input_size'][1]
        self.output_size = (target_size, target_size)
        
        self.transform = get_preprocessing_transforms(model_config, is_training=False)
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            image = panorama_to_plane(
                img_path, 
                self.fov, 
                self.output_size, 
                self.yaw, 
                self.pitch
            ).convert('RGB')
        except Exception as e:
            logging.error(f"Error processing panorama {img_path}: {e}")
            # Return blank image on error
            image = Image.new('RGB', self.output_size)
        
        img_id = Path(img_path).stem
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_id


# ============================================================================
# Prediction Functions
# ============================================================================

def get_embeddings_layer(model, layer_name='pre_final'):
    """
    Get the specified layer from the model for embedding extraction
    
    Args:
        model: The perception model
        layer_name: Which layer to extract embeddings from
            - 'pre_final': Second to last layer (128-dim)
            - 'backbone': Backbone features
            - 'mid': Middle of regression head
            
    Returns:
        int: Index of the layer to extract from
    """
    layer_indices = {
        'pre_final': 6,  # After second ReLU (128-dim)
        'mid': 2,        # After first ReLU (512-dim)
        'backbone': 0    # Backbone features
    }
    return layer_indices.get(layer_name, 6)


def extract_predictions(model, dataloader, device):
    """
    Extract predictions only (no embeddings)
    
    Args:
        model: The perception model
        dataloader: DataLoader with images
        device: torch device
        
    Returns:
        tuple: (predictions, img_ids)
    """
    model.eval()
    all_predictions = []
    all_img_ids = []
    
    with torch.no_grad():
        for batch_images, batch_img_ids in dataloader:
            batch_images = batch_images.to(device)
            predictions = model(batch_images)
            
            all_predictions.append(predictions.cpu().numpy())
            all_img_ids.extend(batch_img_ids)
    
    return np.vstack(all_predictions).squeeze(), all_img_ids


def extract_embeddings_and_predictions(model, dataloader, device, embedding_layer='pre_final'):
    """
    Extract both embeddings and predictions from the model
    
    Args:
        model: The perception model
        dataloader: DataLoader with images
        device: torch device
        embedding_layer: Which layer to extract embeddings from
        
    Returns:
        tuple: (embeddings, predictions, img_ids)
    """
    model.eval()
    all_embeddings = []
    all_predictions = []
    all_img_ids = []
    
    # Get the embedding layer index
    embed_idx = get_embeddings_layer(model, embedding_layer)
    
    with torch.no_grad():
        for batch_images, batch_img_ids in dataloader:
            batch_images = batch_images.to(device)
            
            # Extract features from backbone
            features = model.backbone(batch_images)
            
            # Handle different feature shapes
            if len(features.shape) > 2:
                features = torch.mean(features, dim=[2, 3])
            
            # Process through regression head up to embedding layer
            x = features
            for i, layer in enumerate(model.regression_head[:embed_idx+1]):
                x = layer(x)
            embeddings = x
            
            # Continue through remaining layers for predictions
            for layer in model.regression_head[embed_idx+1:]:
                x = layer(x)
            predictions = x
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_img_ids.extend(batch_img_ids)
    
    return (np.vstack(all_embeddings), 
            np.vstack(all_predictions).squeeze(), 
            all_img_ids)


def extract_backbone_features(model, dataloader, device):
    """
    Extract raw backbone features (useful for transfer learning)
    
    Args:
        model: The perception model
        dataloader: DataLoader with images
        device: torch device
        
    Returns:
        tuple: (features, img_ids)
    """
    model.eval()
    all_features = []
    all_img_ids = []
    
    with torch.no_grad():
        for batch_images, batch_img_ids in dataloader:
            batch_images = batch_images.to(device)
            
            # Extract features from backbone
            features = model.backbone(batch_images)
            
            # Handle different feature shapes
            if len(features.shape) > 2:
                features = torch.mean(features, dim=[2, 3])
            
            all_features.append(features.cpu().numpy())
            all_img_ids.extend(batch_img_ids)
    
    return np.vstack(all_features), all_img_ids


# ============================================================================
# Output Functions
# ============================================================================

def save_to_netcdf(embeddings, predictions, img_ids, output_file, 
                   perception_type, model_weights, metadata=None):
    """
    Save embeddings and predictions to netCDF format
    
    Args:
        embeddings: Embedding array
        predictions: Prediction array
        img_ids: List of image IDs
        output_file: Output file path
        perception_type: Type of perception
        model_weights: Path to model weights
        metadata: Additional metadata dict
    """
    embed_dim = embeddings.shape[1]
    
    ds = xr.Dataset(
        data_vars={
            'embeddings': (['image', 'features'], embeddings),
            f'{perception_type}_score': (['image'], predictions),
        },
        coords={
            'image': img_ids,
            'features': np.arange(embed_dim),
        }
    )
    
    # Add attributes
    ds.attrs['description'] = f'Street view image embeddings and {perception_type} scores'
    ds.attrs['model_weights'] = model_weights
    ds.attrs['perception_type'] = perception_type
    ds.attrs['extraction_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    ds.attrs['embedding_dimension'] = embed_dim
    
    if metadata:
        ds.attrs.update(metadata)
    
    # Add variable attributes
    ds['embeddings'].attrs['description'] = f'{embed_dim}-dimensional embeddings from model'
    ds[f'{perception_type}_score'].attrs['description'] = f'Predicted {perception_type} scores'
    
    # Save to netCDF file
    ds.to_netcdf(output_file)
    logging.info(f"Data saved to {output_file}")


def save_to_csv(predictions, img_ids, output_file, perception_type):
    """
    Save predictions to CSV format
    
    Args:
        predictions: Prediction array
        img_ids: List of image IDs
        output_file: Output file path
        perception_type: Type of perception
    """
    df = pd.DataFrame({
        'img_id': img_ids,
        f'{perception_type}_score': predictions
    })
    
    df.to_csv(output_file, index=False)
    logging.info(f"Predictions saved to {output_file}")


def save_embeddings_to_npy(embeddings, img_ids, output_dir, perception_type):
    """
    Save embeddings to numpy format
    
    Args:
        embeddings: Embedding array
        img_ids: List of image IDs
        output_dir: Output directory
        perception_type: Type of perception
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save embeddings
    embed_file = os.path.join(output_dir, f'{perception_type}_embeddings.npy')
    np.save(embed_file, embeddings)
    
    # Save image IDs
    ids_file = os.path.join(output_dir, f'{perception_type}_img_ids.npy')
    np.save(ids_file, np.array(img_ids))
    
    logging.info(f"Embeddings saved to {output_dir}")


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract predictions and embeddings from perception models'
    )
    
    # Required arguments
    parser.add_argument('--model-weights', type=str, required=True,
                        help='Path to trained model weights (.pt file)')
    parser.add_argument('--backbone', type=str, required=True,
                        choices=list(BACKBONES.keys()),
                        help='Backbone architecture used in the model')
    parser.add_argument('--image-dir', type=str, required=True,
                        help='Directory containing images to process')
    parser.add_argument('--perception-type', type=str, required=True,
                        help='Type of perception score (e.g., beautiful, safe, lively)')
    
    # Output arguments
    parser.add_argument('--output-file', type=str, required=True,
                        help='Path to save output file')
    parser.add_argument('--output-format', type=str, default='netcdf',
                        choices=['netcdf', 'csv', 'npy'],
                        help='Output format (default: netcdf)')
    
    # Processing options
    parser.add_argument('--panorama', action='store_true',
                        help='Process images as panoramas')
    parser.add_argument('--fov', type=float, default=90,
                        help='Field of view for panorama extraction (default: 90)')
    parser.add_argument('--yaw', type=float, default=270,
                        help='Yaw angle for panorama extraction (default: 270)')
    parser.add_argument('--pitch', type=float, default=80,
                        help='Pitch angle for panorama extraction (default: 80)')
    
    # Extraction options
    parser.add_argument('--extract-embeddings', action='store_true',
                        help='Extract embeddings in addition to predictions')
    parser.add_argument('--embedding-layer', type=str, default='pre_final',
                        choices=['pre_final', 'mid', 'backbone'],
                        help='Which layer to extract embeddings from (default: pre_final)')
    parser.add_argument('--backbone-features-only', action='store_true',
                        help='Extract only backbone features (no predictions)')
    
    # Data loading options
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--workers', type=int, default=NUM_WORKERS,
                        help=f'Number of data loading workers (default: {NUM_WORKERS})')
    parser.add_argument('--image-pattern', type=str, default='*.jpg',
                        help='Glob pattern for image files (default: *.jpg)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load image paths
    image_pattern = os.path.join(args.image_dir, args.image_pattern)
    image_paths = glob.glob(image_pattern)
    
    if len(image_paths) == 0:
        logging.error(f"No images found matching pattern: {image_pattern}")
        return
    
    logging.info(f"Found {len(image_paths)} images in {args.image_dir}")
    
    # Get model configuration
    model_config = BACKBONES[args.backbone]
    logging.info(f"Using backbone: {args.backbone}")
    
    # Create dataset
    if args.panorama:
        logging.info(f"Processing as panoramas (FOV={args.fov}, yaw={args.yaw}, pitch={args.pitch})")
        dataset = PanoramaDataset(
            image_paths, 
            model_config,
            fov=args.fov,
            yaw=args.yaw,
            pitch=args.pitch
        )
    else:
        logging.info("Processing as regular images")
        dataset = PredictionDataset(image_paths, model_config)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=False
    )
    
    # Initialize and load model
    logging.info("Loading model...")
    model = PerceptionModel(
        backbone_name=args.backbone,
        pretrained=False,  # We're loading trained weights
        perception_dim=1
    )
    
    # Load weights
    try:
        state_dict = torch.load(args.model_weights, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f"Loaded weights from {args.model_weights}")
    except Exception as e:
        logging.error(f"Error loading model weights: {e}")
        return
    
    model = model.to(device)
    
    # Extract features/predictions based on options
    if args.backbone_features_only:
        logging.info("Extracting backbone features only...")
        features, img_ids = extract_backbone_features(model, dataloader, device)
        
        # Save backbone features
        if args.output_format == 'npy':
            output_dir = os.path.dirname(args.output_file) or '.'
            save_embeddings_to_npy(features, img_ids, output_dir, f'{args.perception_type}_backbone')
        else:
            # Save as netCDF
            save_to_netcdf(
                features, 
                np.zeros(len(img_ids)),  # Dummy predictions
                img_ids,
                args.output_file,
                args.perception_type,
                args.model_weights,
                metadata={'feature_type': 'backbone_only', 'backbone': args.backbone}
            )
    
    elif args.extract_embeddings:
        logging.info(f"Extracting embeddings and predictions (layer: {args.embedding_layer})...")
        embeddings, predictions, img_ids = extract_embeddings_and_predictions(
            model, dataloader, device, args.embedding_layer
        )
        
        logging.info(f"Extracted embeddings shape: {embeddings.shape}")
        logging.info(f"Extracted predictions shape: {predictions.shape}")
        
        # Save based on format
        if args.output_format == 'netcdf':
            save_to_netcdf(
                embeddings, 
                predictions, 
                img_ids,
                args.output_file,
                args.perception_type,
                args.model_weights,
                metadata={
                    'backbone': args.backbone,
                    'embedding_layer': args.embedding_layer,
                    'panorama': args.panorama
                }
            )
        elif args.output_format == 'npy':
            output_dir = os.path.dirname(args.output_file) or '.'
            save_embeddings_to_npy(embeddings, img_ids, output_dir, args.perception_type)
            # Also save predictions
            pred_file = os.path.join(output_dir, f'{args.perception_type}_predictions.npy')
            np.save(pred_file, predictions)
        else:  # csv
            save_to_csv(predictions, img_ids, args.output_file, args.perception_type)
            logging.warning("CSV format selected but embeddings will not be saved. Use netcdf or npy format to save embeddings.")
    
    else:
        logging.info("Extracting predictions only...")
        predictions, img_ids = extract_predictions(model, dataloader, device)
        
        logging.info(f"Extracted predictions shape: {predictions.shape}")
        
        # Save predictions
        if args.output_format == 'csv':
            save_to_csv(predictions, img_ids, args.output_file, args.perception_type)
        else:
            # Save as netCDF without embeddings
            save_to_netcdf(
                np.zeros((len(img_ids), 1)),  # Dummy embeddings
                predictions,
                img_ids,
                args.output_file,
                args.perception_type,
                args.model_weights,
                metadata={'backbone': args.backbone, 'predictions_only': True}
            )
    
    logging.info("Processing complete!")


if __name__ == '__main__':
    main()
