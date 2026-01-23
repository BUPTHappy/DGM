from typing import Dict
import torch
import numpy as np
import copy
from unified_video_action.common.pytorch_util import dict_apply
from unified_video_action.common.replay_buffer import ReplayBuffer
from unified_video_action.common.sampler import (
    SequenceSampler,
    get_val_mask,
    downsample_mask,
)
from unified_video_action.model.common.normalizer import LinearNormalizer
from unified_video_action.dataset.base_dataset import BaseImageDataset
from unified_video_action.common.normalize_util import get_image_range_normalizer
import torchvision.transforms as transforms
import torchvision


class UmiCupArrangementDataset(BaseImageDataset):
    """
    Dataset class for UMI cup arrangement tasks.
    Supports both cup_in_the_wild and cup_in_the_lab datasets.
    """
    def __init__(
        self,
        dataset_path,
        horizon=32,
        pad_before=1,
        pad_after=7,
        seed=42,
        val_ratio=0.02,
        max_train_episodes=None,
        language_emb_model=None,
        data_aug=False,
        normalizer_type=None,
        dataset_type="singletask",
    ):

        super().__init__()

        # First, check what keys are available in the zarr dataset
        import zarr
        zarr_store = zarr.open(dataset_path, mode="r")
        
        # ReplayBuffer expects keys in "data" group
        if "data" not in zarr_store:
            raise ValueError(f"Dataset does not have 'data' group. Available groups: {list(zarr_store.keys())}")
        
        data_group = zarr_store["data"]
        available_keys = list(data_group.keys())
        print(f"Available keys in dataset['data']: {available_keys}")
        
        # Try to find image key (common names: img, image, rgb)
        image_key = None
        for key in ["img", "image", "rgb"]:
            if key in available_keys:
                image_key = key
                break
        
        if image_key is None:
            # Check if there's an "obs" group with image inside
            if "obs" in available_keys:
                obs_item = data_group["obs"]
                if isinstance(obs_item, zarr.Group):
                    obs_keys = list(obs_item.keys())
                    print(f"Keys inside 'data/obs': {obs_keys}")
                    for key in ["image", "rgb", "img"]:
                        if key in obs_keys:
                            # For ReplayBuffer, we'll need to handle this differently
                            # Let's try using "obs" as the key and handle it in _sample_to_data
                            image_key = "obs"
                            self.obs_image_key = key
                            break
        
        if image_key is None:
            raise ValueError(
                f"Could not find image key in dataset. Available keys: {available_keys}. "
                "Please check the dataset structure. Expected keys: img, image, rgb, or obs/image"
            )
        
        print(f"Using image key: {image_key}")
        
        # Build keys list for ReplayBuffer (only top-level keys in data group)
        keys_to_load = [image_key, "action"]
        if "state" in available_keys:
            keys_to_load.append("state")
        
        print(f"Loading keys: {keys_to_load}")
        
        # Load zarr dataset using ReplayBuffer
        self.replay_buffer = ReplayBuffer.copy_from_path(
            dataset_path, keys=keys_to_load
        )
        
        # Store the image key for later use
        self.image_key = image_key
        # Check if we need nested key access
        if hasattr(self, 'obs_image_key'):
            self.use_nested_image = True
        else:
            self.use_nested_image = False
        
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, max_n=max_train_episodes, seed=seed
        )

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.data_aug = data_aug
        self.dataset_type = dataset_type

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
        }
        
        # Add state if available (adjust based on actual data structure)
        if "state" in self.replay_buffer:
            # UMI datasets may have different state structure
            # Adjust based on actual state keys
            state_data = self.replay_buffer["state"]
            if state_data.shape[-1] >= 2:
                data["agent_pos"] = state_data[..., :2]
        
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer["image"] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # Extract image using the detected key
        # Handle both direct key and nested key (e.g., obs/image inside obs group)
        if self.use_nested_image:
            # obs is a group, and image is inside it (e.g., obs/image)
            image_data = sample["obs"][self.obs_image_key]
        elif "/" in self.image_key:
            # Nested key like "obs/image" (string path)
            parts = self.image_key.split("/")
            image_data = sample
            for part in parts:
                image_data = image_data[part]
        else:
            # Direct key like "img", "image", "rgb"
            image_data = sample[self.image_key]
        
        # Convert to (T, C, H, W) format
        # Handle different possible shapes: (T, H, W, C) or (T, C, H, W)
        if len(image_data.shape) == 4:
            if image_data.shape[-1] == 3 or image_data.shape[-1] == 1:
                # Shape is (T, H, W, C), need to move axis
                image = np.moveaxis(image_data, -1, 1) / 255.0
            else:
                # Shape might already be (T, C, H, W)
                image = image_data.astype(np.float32) / 255.0
        else:
            raise ValueError(f"Unexpected image shape: {image_data.shape}")
        
        # Extract action
        action = sample["action"].astype(np.float32)
        
        # Extract state/agent_pos if available
        obs_dict = {"image": image}
        if "state" in sample:
            state = sample["state"]
            if state.shape[-1] >= 2:
                agent_pos = state[..., :2].astype(np.float32)
                obs_dict["agent_pos"] = agent_pos

        if self.data_aug:
            image_tensor = torch.tensor(image, dtype=torch.float32)
            video_seed = torch.randint(0, 10000, (1,)).item()

            def consistent_augmentations(frame):
                torch.manual_seed(video_seed)
                frame_size = image.shape[-1]  # Use actual image size
                augmentation = transforms.Compose(
                    [
                        torchvision.transforms.RandomApply(
                            [
                                torchvision.transforms.RandomCrop(
                                    size=int(frame_size * 0.95)
                                )
                            ],
                            p=0.5,
                        ),
                        torchvision.transforms.Resize(
                            size=frame_size, antialias=True
                        ),
                        torchvision.transforms.RandomApply(
                            [
                                torchvision.transforms.GaussianBlur(
                                    kernel_size=(5, 5), sigma=(0.1, 2.0)
                                )
                            ],
                            p=0.5,
                        ),
                    ]
                )
                return augmentation(frame)

            augmented_images = torch.stack(
                [consistent_augmentations(frame) for frame in image_tensor]
            )
            image = augmented_images.numpy()
            obs_dict["image"] = image

        data = {
            "obs": obs_dict,
            "action": action,
        }

        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
