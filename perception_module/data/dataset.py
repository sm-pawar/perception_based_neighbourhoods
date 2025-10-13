"""
Dataset classes for the perception model
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image

from data.utils import get_preprocessing_transforms, panorama_to_plane

class CustomImageDataset(Dataset):
    """
    Custom dataset for loading and preprocessing street view images with perception scores
    """
    def __init__(self, df, image_dir, model_config, is_training=True):
        """
        Initialize the dataset
        
        Args:
            df (pandas.DataFrame): DataFrame containing image paths and scores
            image_dir (str): Directory containing the images
            model_config (dict): Configuration for the model architecture
            is_training (bool): Whether to apply training augmentations
        """
        self.df = df
        self.image_dir = image_dir
        self.model_config = model_config
        self.is_training = is_training
        self.transform = get_preprocessing_transforms(model_config, is_training)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.df.iloc[idx]["img_id"] + '.jpg')
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if there's an error
            image = Image.new('RGB', (self.model_config['input_size'][1], self.model_config['input_size'][2]))
            
        label = self.df.iloc[idx]["trueskill.score"]
        sigma = self.df.iloc[idx]["trueskill.sigma"]

        if self.transform:
            image = self.transform(image)
            
        return image, label, sigma


def create_dataloaders(train_df, val_df, test_df, image_dir, model_config, batch_size, num_workers, distributed=False):
    """
    Create data loaders for training, validation, and testing
    
    Args:
        train_df (pandas.DataFrame): Training data
        val_df (pandas.DataFrame): Validation data
        test_df (pandas.DataFrame): Test data
        image_dir (str): Directory containing images
        model_config (dict): Model configuration
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        distributed (bool): Whether to use distributed sampler
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = CustomImageDataset(train_df, image_dir, model_config, is_training=True)
    val_dataset = CustomImageDataset(val_df, image_dir, model_config, is_training=False)
    test_dataset = CustomImageDataset(test_df, image_dir, model_config, is_training=False)
    
    # Create samplers if using distributed training
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return train_loader, val_loader, test_loader, train_sampler

class EmbeddingImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.model_config = model_config
        self.is_training = False
        self.image_size = (model_config['input_size'][1], model_config['input_size'][2])
        self.transform = get_preprocessing_transforms(model_config, is_training)
        
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = panorama_to_plane(img_path, 90, self.image_size, 270, 80).convert('RGB')
        img_id = Path(img_path).stem
        
        if self.transform:
            image = self.transform(image)
        return image, img_id