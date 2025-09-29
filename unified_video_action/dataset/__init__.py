"""
Dataset module for unified video action
"""

from .base_dataset import BaseDataset, BaseLowdimDataset, BaseImageDataset
from .base_lazy_dataset import BaseLazyDataset
from .umi_multi_dataset import UmiMultiDataset
from .umi_lazy_dataset import UmiLazyDataset
from .pusht_image_dataset import PushTImageDataset
from .libero_replay_image_dataset import LiberoReplayImageDataset
from .robomimic_replay_image_dataset import RobomimicReplayImageDataset

import os
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Union, Any, Optional


def get_dataset(dataset_path: str, **kwargs) -> Union[BaseDataset, BaseImageDataset, UmiMultiDataset]:
    """
    Load a dataset from the given path.
    
    Args:
        dataset_path: Path to the dataset configuration or data
        **kwargs: Additional arguments for dataset instantiation
        
    Returns:
        Dataset instance
    """
    # If dataset_path is a config file, load it with Hydra
    if dataset_path.endswith('.yaml') or dataset_path.endswith('.yml'):
        if os.path.exists(dataset_path):
            cfg = OmegaConf.load(dataset_path)
            return hydra.utils.instantiate(cfg, **kwargs)
        else:
            raise FileNotFoundError(f"Dataset config file not found: {dataset_path}")
    
    # If dataset_path is a directory, try to infer the dataset type
    if os.path.isdir(dataset_path):
        # Check for common dataset patterns
        if 'pusht' in dataset_path.lower():
            # For pusht, look for the zarr file inside the directory
            zarr_path = os.path.join(dataset_path, 'pusht_cchi_v7_replay.zarr')
            if os.path.exists(zarr_path):
                return PushTImageDataset(zarr_path, **kwargs)
            else:
                # If no zarr file found, try the directory itself
                return PushTImageDataset(dataset_path, **kwargs)
        elif 'libero' in dataset_path.lower():
            # Default shape_meta for libero datasets
            default_shape_meta = {
                'action': {'shape': [7]},
                'obs': {
                    'agentview_rgb': {'shape': [3, 128, 128], 'type': 'rgb'},
                    'eye_in_hand_rgb': {'shape': [3, 128, 128], 'type': 'rgb'},
                    'ee_pos': {'shape': [3], 'type': 'low_dim'},
                    'ee_ori': {'shape': [3], 'type': 'low_dim'},
                    'gripper_states': {'shape': [2], 'type': 'low_dim'},
                    'joint_states': {'shape': [7], 'type': 'low_dim'},
                }
            }
            # Remove split parameter if present (not supported by LiberoReplayImageDataset)
            kwargs_clean = {k: v for k, v in kwargs.items() if k != 'split'}
            return LiberoReplayImageDataset(shape_meta=default_shape_meta, dataset_path=dataset_path, **kwargs_clean)
        elif 'umi' in dataset_path.lower():
            return UmiLazyDataset(dataset_path, **kwargs)
        else:
            # Default to UmiLazyDataset for unknown datasets
            return UmiLazyDataset(dataset_path, **kwargs)
    
    # If it's a file path, try to load as zarr
    if os.path.isfile(dataset_path):
        if dataset_path.endswith('.zarr'):
            return UmiLazyDataset(dataset_path, **kwargs)
        else:
            raise ValueError(f"Unsupported dataset file format: {dataset_path}")
    
    raise ValueError(f"Cannot determine dataset type for path: {dataset_path}")


__all__ = [
    'BaseDataset',
    'BaseLowdimDataset', 
    'BaseImageDataset',
    'BaseLazyDataset',
    'UmiMultiDataset',
    'UmiLazyDataset',
    'PushTImageDataset',
    'LiberoReplayImageDataset',
    'RobomimicReplayImageDataset',
    'get_dataset'
]
