"""
Miscellaneous utility functions
"""

import os
import logging
import torch
import random
import numpy as np
from datetime import datetime


def setup_logging(base_save_path, perception, date=None):
    """
    Set up logging to file and console
    
    Args:
        base_save_path (str): Base directory for saving logs
        perception (str): Perception type
        date (str, optional): Date string for log name
        
    Returns:
        logging.Logger: Configured logger
    """
    if date is None:
        date = datetime.now().strftime('%Y%m%d%H%M')
    
    # Create the save directory if it doesn't exist
    os.makedirs(base_save_path, exist_ok=True)
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure logging
    log_file = os.path.join(base_save_path, f"{date}_{perception}_training.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True  # This forces reconfiguration in Python 3.8+
    )
    
    logger = logging.getLogger()
    print(f"Logging to {log_file}")
    logger.info(f"Logging initialized. Writing to {log_file}")
    
    return logger

# def setup_logging(base_save_path, perception, date=None):
#     """
#     Set up logging to file and console
    
#     Args:
#         base_save_path (str): Base directory for saving logs
#         perception (str): Perception type
#         date (str, optional): Date string for log name
        
#     Returns:
#         logging.Logger: Configured logger
#     """
#     if date is None:
#         date = datetime.now().strftime('%Y%m%d%H%M')
    
#     # Create the save directory if it doesn't exist
#     os.makedirs(base_save_path, exist_ok=True)
    
#     # Configure logging
#     log_file = os.path.join(base_save_path, f"{date}_{perception}_training.log")
    
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(log_file),
#             logging.StreamHandler()
#         ]
#     )
    
#     logger = logging.getLogger()
#     print(f"Logging to {log_file}")
#     logger.info(f"Logging initialized. Writing to {log_file}")
    
#     return logger


def setup_device():
    """
    Set up device (CPU/GPU) and distributed training if needed
    
    Args:
        local_rank (int, optional): Local rank for distributed training
        
    Returns:
        tuple: (device, is_distributed)
    """
    # Check if CUDA is available
    #cuda_available = torch.cuda.is_available()
    
    # if not cuda_available:
    #     logging.warning("CUDA is not available. Using CPU for training.")
    #     return torch.device("cpu"), False
    
    # Initialize distributed training if local_rank is provided
    # if local_rank is not None:
    #print(f"Initializing distributed training with local rank {local_rank}")
    #if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend='nccl')

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
        
    logging.info(f"Distributed training initialized. Local rank: {local_rank}")
    return torch.device(f"cuda:{local_rank}"), True
    
    # # Regular single-GPU training
    # return torch.device("cuda:0"), False



# def setup_device(local_rank=None):
#     """
#     Set up device (CPU/GPU) and distributed training if needed
    
#     Args:
#         local_rank (int, optional): Local rank for distributed training
        
#     Returns:
#         tuple: (device, is_distributed)
#     """
#     # Check if CUDA is available
#     cuda_available = torch.cuda.is_available()
    
#     if not cuda_available:
#         logging.warning("CUDA is not available. Using CPU for training.")
#         return torch.device("cpu"), False
    
#     # Initialize distributed training if local_rank is provided
#     if local_rank is not None:
#         print(f"Initializing distributed training with local rank {local_rank}")
#         if not torch.distributed.is_initialized():
#             torch.distributed.init_process_group(backend='nccl')

#         local_rank = int(local_rank)
#         torch.cuda.set_device(local_rank)
        
#         logging.info(f"Distributed training initialized. Local rank: {local_rank}")
#         return torch.device(f"cuda:{local_rank}"), True
    
#     # Regular single-GPU training
#     return torch.device("cuda:0"), False


def set_seed(seed=42):
    """
    Set random seed for reproducibility
    
    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"Random seed set to {seed}")


def get_save_paths(base_path, perception, backbone, date=None):
    """
    Generate save paths for models and logs
    
    Args:
        base_path (str): Base directory for saving
        perception (str): Perception type
        backbone (str): Backbone architecture name
        date (str, optional): Date string
        
    Returns:
        dict: Dictionary with paths for saving models and logs
    """
    if date is None:
        date = datetime.now().strftime('%Y%m%d')
    
    # Create output directory
    output_dir = os.path.join(base_path, f"{date}_{perception}_{backbone}")
    os.makedirs(output_dir, exist_ok=True)
    
    paths = {
        'output_dir': output_dir,
        'best_val_model': os.path.join(output_dir, f"best_model_val.pt"),
        'best_train_model': os.path.join(output_dir, f"best_model_train.pt"),
        'latest_checkpoint': os.path.join(output_dir, f"latest_checkpoint.pt"),
        'log_file': os.path.join(output_dir, f"training.log")
    }
    
    return paths