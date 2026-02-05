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


class UmiBimanualDataset(BaseImageDataset):
    """
    Dataset class for UMI bimanual (dual-arm) tasks.
    Supports dish_washing, cloth_folding, dynamic_tossing datasets.
    
    Action dimension: 14D (7D per arm)
    - robot0: pos(3) + rot(3) + gripper(1) = 7D
    - robot1: pos(3) + rot(3) + gripper(1) = 7D
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
                            image_key = "obs"
                            self.obs_image_key = key
                            break
        
        if image_key is None:
            raise ValueError(
                f"Could not find image key in dataset. Available keys: {available_keys}. "
                "Please check the dataset structure. Expected keys: img, image, rgb, or obs/image"
            )
        
        print(f"Using image key: {image_key}")
        
        # Check for bimanual state keys (both robot0 and robot1)
        robot0_state_keys = ["robot0_eef_pos", "robot0_eef_rot_axis_angle", "robot0_gripper_width"]
        robot1_state_keys = ["robot1_eef_pos", "robot1_eef_rot_axis_angle", "robot1_gripper_width"]
        
        has_robot0 = all(key in available_keys for key in robot0_state_keys)
        has_robot1 = all(key in available_keys for key in robot1_state_keys)
        
        if not has_robot0:
            raise ValueError(f"Missing robot0 state keys. Available: {available_keys}, Required: {robot0_state_keys}")
        if not has_robot1:
            raise ValueError(f"Missing robot1 state keys. Available: {available_keys}, Required: {robot1_state_keys}")
        
        print("✓ Found bimanual state keys (robot0 and robot1)")
        
        # Check if action exists
        has_action = "action" in available_keys
        if not has_action:
            print("No 'action' key found. Will compute action from state deltas.")
            self.compute_action_from_state = True
        else:
            # Check action dimension
            action_array = data_group["action"]
            action_dim = action_array.shape[-1] if len(action_array.shape) > 1 else 1
            print(f"Found 'action' key with dimension: {action_dim}")
            self.compute_action_from_state = False
        
        # Build keys list for ReplayBuffer
        keys_to_load = [image_key]
        if has_action and not self.compute_action_from_state:
            keys_to_load.append("action")
        
        # Always load state keys for bimanual
        keys_to_load.extend(robot0_state_keys)
        keys_to_load.extend(robot1_state_keys)
        
        # Remove duplicates while preserving order
        keys_to_load = list(dict.fromkeys(keys_to_load))
        print(f"Loading keys: {keys_to_load}")
        
        # Load zarr dataset using ReplayBuffer
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
            # Compute bimanual action from state deltas for normalization
            # Robot 0
            eef_pos_0 = self.replay_buffer["robot0_eef_pos"]
            eef_rot_0 = self.replay_buffer["robot0_eef_rot_axis_angle"]
            gripper_0 = self.replay_buffer["robot0_gripper_width"]
            
            # Robot 1
            eef_pos_1 = self.replay_buffer["robot1_eef_pos"]
            eef_rot_1 = self.replay_buffer["robot1_eef_rot_axis_angle"]
            gripper_1 = self.replay_buffer["robot1_gripper_width"]
            
            # Compute deltas (next - current)
            action_pos_0 = np.diff(eef_pos_0, axis=0, prepend=eef_pos_0[0:1])
            action_rot_0 = np.diff(eef_rot_0, axis=0, prepend=eef_rot_0[0:1])
            action_gripper_0 = np.diff(gripper_0, axis=0, prepend=gripper_0[0:1])
            
            action_pos_1 = np.diff(eef_pos_1, axis=0, prepend=eef_pos_1[0:1])
            action_rot_1 = np.diff(eef_rot_1, axis=0, prepend=eef_rot_1[0:1])
            action_gripper_1 = np.diff(gripper_1, axis=0, prepend=gripper_1[0:1])
            
            # Concatenate to form bimanual action:
            # [pos0(3), rot0(3), gripper0(1), pos1(3), rot1(3), gripper1(1)] = 14D
            action = np.concatenate([
                action_pos_0, action_rot_0, action_gripper_0,
                action_pos_1, action_rot_1, action_gripper_1
            ], axis=-1)
            data["action"] = action
        else:
            data["action"] = self.replay_buffer["action"]
        
        # Fit normalizers for obs fields (needed for normalization during training)
        # Robot 0 state
        data["robot0_eef_pos"] = self.replay_buffer["robot0_eef_pos"]
        data["robot0_eef_rot_axis_angle"] = self.replay_buffer["robot0_eef_rot_axis_angle"]
        data["robot0_gripper_width"] = self.replay_buffer["robot0_gripper_width"]
        
        # Robot 1 state
        data["robot1_eef_pos"] = self.replay_buffer["robot1_eef_pos"]
        data["robot1_eef_rot_axis_angle"] = self.replay_buffer["robot1_eef_rot_axis_angle"]
        data["robot1_gripper_width"] = self.replay_buffer["robot1_gripper_width"]
        
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer["image"] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # Extract image using the detected key
        if self.use_nested_image:
            image_data = sample["obs"][self.obs_image_key]
        elif "/" in self.image_key:
            parts = self.image_key.split("/")
            image_data = sample
            for part in parts:
                image_data = image_data[part]
        else:
            image_data = sample[self.image_key]
        
        # Convert to (T, C, H, W) format
        if len(image_data.shape) == 4:
            if image_data.shape[-1] == 3 or image_data.shape[-1] == 1:
                # Shape is (T, H, W, C), need to move axis
                image = np.moveaxis(image_data, -1, 1) / 255.0
            else:
                # Shape might already be (T, C, H, W)
                image = image_data.astype(np.float32) / 255.0
        else:
            raise ValueError(f"Unexpected image shape: {image_data.shape}")
        
        # Extract state for both robots
        eef_pos_0 = sample["robot0_eef_pos"].astype(np.float32)  # (T, 3)
        eef_rot_0 = sample["robot0_eef_rot_axis_angle"].astype(np.float32)  # (T, 3)
        gripper_0 = sample["robot0_gripper_width"].astype(np.float32)  # (T, 1)
        
        eef_pos_1 = sample["robot1_eef_pos"].astype(np.float32)  # (T, 3)
        eef_rot_1 = sample["robot1_eef_rot_axis_angle"].astype(np.float32)  # (T, 3)
        gripper_1 = sample["robot1_gripper_width"].astype(np.float32)  # (T, 1)
        
        # Extract or compute action
        if self.compute_action_from_state:
            # Compute bimanual action as state delta
            action_pos_0 = np.diff(eef_pos_0, axis=0, prepend=eef_pos_0[0:1])
            action_rot_0 = np.diff(eef_rot_0, axis=0, prepend=eef_rot_0[0:1])
            action_gripper_0 = np.diff(gripper_0, axis=0, prepend=gripper_0[0:1])
            
            action_pos_1 = np.diff(eef_pos_1, axis=0, prepend=eef_pos_1[0:1])
            action_rot_1 = np.diff(eef_rot_1, axis=0, prepend=eef_rot_1[0:1])
            action_gripper_1 = np.diff(gripper_1, axis=0, prepend=gripper_1[0:1])
            
            # Concatenate: [robot0(7), robot1(7)] = 14D
            action = np.concatenate([
                action_pos_0, action_rot_0, action_gripper_0,
                action_pos_1, action_rot_1, action_gripper_1
            ], axis=-1).astype(np.float32)
        else:
            action = sample["action"].astype(np.float32)
        
        # Build obs dict with image and bimanual state information
        obs_dict = {
            "image": image,
            # Robot 0 state
            "robot0_eef_pos": eef_pos_0,
            "robot0_eef_rot_axis_angle": eef_rot_0,
            "robot0_gripper_width": gripper_0,
            # Robot 1 state
            "robot1_eef_pos": eef_pos_1,
            "robot1_eef_rot_axis_angle": eef_rot_1,
            "robot1_gripper_width": gripper_1,
        }

        if self.data_aug:
            image_tensor = torch.tensor(image, dtype=torch.float32)
            video_seed = torch.randint(0, 10000, (1,)).item()

            def consistent_augmentations(frame):
                torch.manual_seed(video_seed)
                frame_size = image.shape[-1]
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
