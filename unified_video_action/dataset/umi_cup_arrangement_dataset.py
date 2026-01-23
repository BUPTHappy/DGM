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
from unified_video_action.codecs.imagecodecs_numcodecs import register_codecs
import torchvision.transforms as transforms
import torchvision

# Register imagecodecs codecs for zarr (needed for JPEGXL compression)
register_codecs()


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
        
        # Try to find image key (common names: img, image, rgb, camera0_rgb, etc.)
        image_key = None
        # Check for common image key names
        for key in ["img", "image", "rgb", "camera0_rgb", "camera_rgb"]:
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
                    for key in ["image", "rgb", "img", "camera0_rgb", "camera_rgb"]:
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
        
        # Check if action exists, if not we'll compute it from state deltas
        has_action = "action" in available_keys
        if not has_action:
            print("No 'action' key found. Will compute action from state deltas.")
            # Check if we have the required state keys for computing action
            required_state_keys = ["robot0_eef_pos", "robot0_eef_rot_axis_angle", "robot0_gripper_width"]
            has_all_state = all(key in available_keys for key in required_state_keys)
            if not has_all_state:
                raise ValueError(
                    f"Cannot compute action: missing required state keys. "
                    f"Available: {available_keys}, Required: {required_state_keys}"
                )
            self.compute_action_from_state = True
        else:
            self.compute_action_from_state = False
        
        # Build keys list for ReplayBuffer
        keys_to_load = [image_key]
        if has_action:
            keys_to_load.append("action")
        else:
            # Load state keys to compute action
            keys_to_load.extend(["robot0_eef_pos", "robot0_eef_rot_axis_angle", "robot0_gripper_width"])
        
        print(f"Loading keys: {keys_to_load}")
        
        # Load zarr dataset using ReplayBuffer
        # Note: imagecodecs_jpegxl compression may cause issues, but ReplayBuffer should handle it
        try:
            self.replay_buffer = ReplayBuffer.copy_from_path(
                dataset_path, keys=keys_to_load
            )
        except Exception as e:
            if "imagecodecs_jpegxl" in str(e) or "codec" in str(e).lower():
                raise RuntimeError(
                    f"Failed to load dataset due to image compression codec issue: {e}\n"
                    "Please install imagecodecs: conda install -c conda-forge imagecodecs"
                ) from e
            raise
        
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
        data = {}
        
        # Get action data (either from buffer or compute from state)
        if self.compute_action_from_state:
            # Compute action from state deltas for normalization
            eef_pos = self.replay_buffer["robot0_eef_pos"]
            eef_rot = self.replay_buffer["robot0_eef_rot_axis_angle"]
            gripper = self.replay_buffer["robot0_gripper_width"]
            
            # Compute deltas (next - current)
            action_pos = np.diff(eef_pos, axis=0, prepend=eef_pos[0:1])
            action_rot = np.diff(eef_rot, axis=0, prepend=eef_rot[0:1])
            action_gripper = np.diff(gripper, axis=0, prepend=gripper[0:1])
            
            # Concatenate to form action: [pos(3), rot(3), gripper(1)] = 7D
            action = np.concatenate([action_pos, action_rot, action_gripper], axis=-1)
            data["action"] = action
        else:
            data["action"] = self.replay_buffer["action"]
        
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
            # Direct key like "camera0_rgb", "img", "image", "rgb"
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
        
        # Extract or compute action
        if self.compute_action_from_state:
            # Compute action as state delta: action[t] = state[t+1] - state[t]
            # For sequence, we compute deltas within the sequence
            eef_pos = sample["robot0_eef_pos"].astype(np.float32)  # (T, 3)
            eef_rot = sample["robot0_eef_rot_axis_angle"].astype(np.float32)  # (T, 3)
            gripper = sample["robot0_gripper_width"].astype(np.float32)  # (T, 1)
            
            # Compute deltas: next - current
            # For the last timestep, use zero delta (no change)
            T = eef_pos.shape[0]
            action_pos = np.diff(eef_pos, axis=0, prepend=eef_pos[0:1])
            action_rot = np.diff(eef_rot, axis=0, prepend=eef_rot[0:1])
            action_gripper = np.diff(gripper, axis=0, prepend=gripper[0:1])
            
            # Concatenate: [pos(3), rot(3), gripper(1)] = 7D for single arm
            # For dual arm it would be 14D (7 per arm)
            action = np.concatenate([action_pos, action_rot, action_gripper], axis=-1).astype(np.float32)
        else:
            action = sample["action"].astype(np.float32)
        
        # Build obs dict with image and state information
        obs_dict = {"image": image}
        
        # Add state information to obs (needed for UMI datasets)
        if self.compute_action_from_state:
            obs_dict["robot0_eef_pos"] = eef_pos
            obs_dict["robot0_eef_rot_axis_angle"] = eef_rot
            obs_dict["robot0_gripper_width"] = gripper
        elif "robot0_eef_pos" in sample:
            # State keys might be available even if action exists
            obs_dict["robot0_eef_pos"] = sample["robot0_eef_pos"].astype(np.float32)
            if "robot0_eef_rot_axis_angle" in sample:
                obs_dict["robot0_eef_rot_axis_angle"] = sample["robot0_eef_rot_axis_angle"].astype(np.float32)
            if "robot0_gripper_width" in sample:
                obs_dict["robot0_gripper_width"] = sample["robot0_gripper_width"].astype(np.float32)

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
