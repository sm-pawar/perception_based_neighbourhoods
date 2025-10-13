"""
Data utility functions for processing and preparing perception datasets
"""

import pandas as pd
import os
import logging
from torchvision import transforms
import timm

def pp_process_input(csv_path, perception):
    """
    Load and process the input CSV file for a specific perception type
    
    Args:
        csv_path (str): Path to the CSV file
        perception (str): Perception type to filter
        
    Returns:
        tuple: (train_df, val_df, test_df) DataFrames
    """
    # Load training csv and basic cleaning 
    df = pd.read_csv(csv_path)
    df = df[df['study_id'] == perception] #.replace('_', ' ')] # for euro training commented out replace part. 
    df = df[df['img_check'] == 'pass']
    df = df.dropna(subset=['trueskill.score'], ignore_index=True)
    
    # Split into train/val/test
    df_train = df.sample(frac=0.6, random_state=42)
    remaining_data = df.drop(df_train.index)
    df_val = remaining_data.sample(frac=0.5, random_state=42)
    df_test = remaining_data.drop(df_val.index)
    
    logging.info(f"Dataset split: Train {len(df_train)}, Val {len(df_val)}, Test {len(df_test)}")
    
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)

def get_preprocessing_transforms(model_config, is_training=True):
    """
    Get the appropriate image preprocessing transforms for a model
    
    Args:
        model_config (dict): Model configuration with input size and interpolation method
        is_training (bool): Whether to apply training augmentations
        
    Returns:
        transforms.Compose: The preprocessing transforms
    """
    data_config = {
        'input_size': model_config['input_size'],
        'interpolation': model_config['interpolation'],
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
        'crop_pct': 1.0,
        'crop_mode': 'center'
    }
    
    # Base transforms from timm
    base_transforms = timm.data.create_transform(**data_config, is_training=False)
    
    if is_training:
        # Get the target size for cropping
        target_size = model_config['input_size'][1]
        
        # Add augmentations for training
        return transforms.Compose([
            transforms.RandomResizedCrop(target_size, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
            base_transforms
        ])
    else:
        return base_transforms


# Prediction Images cropping scripts
## Function to extracting streetview from panos
def map_to_sphere(x, y, z, yaw_radian, pitch_radian):
    theta = np.arccos(z / np.sqrt(x ** 2 + y ** 2 + z ** 2))
    phi = np.arctan2(y, x)

    # Apply rotation transformations here
    theta_prime = np.arccos(np.sin(theta) * np.sin(phi) * np.sin(pitch_radian) +
                            np.cos(theta) * np.cos(pitch_radian))

    phi_prime = np.arctan2(np.sin(theta) * np.sin(phi) * np.cos(pitch_radian) -
                            np.cos(theta) * np.sin(pitch_radian),
                            np.sin(theta) * np.cos(phi))
    phi_prime += yaw_radian
    phi_prime = phi_prime % (2 * np.pi)

    return theta_prime.flatten(), phi_prime.flatten()


def interpolate_color(coords, img, method='bilinear'):
    order = {'nearest': 0, 'bilinear': 1, 'bicubic': 3}.get(method, 1)
    red = map_coordinates(img[:, :, 0], coords, order=order, mode='reflect')
    green = map_coordinates(img[:, :, 1], coords, order=order, mode='reflect')
    blue = map_coordinates(img[:, :, 2], coords, order=order, mode='reflect')
    return np.stack((red, green, blue), axis=-1)


def panorama_to_plane(panorama_path, FOV, output_size, yaw, pitch):
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