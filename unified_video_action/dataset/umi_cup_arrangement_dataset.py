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

        # Load zarr dataset using ReplayBuffer
        # UMI datasets typically have keys: img, state, action
        # Adjust keys based on actual zarr file structure
        self.replay_buffer = ReplayBuffer.copy_from_path(
            dataset_path, keys=["img", "state", "action"]
        )
        
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
        # Extract image and convert to (T, C, H, W) format
        image = np.moveaxis(sample["img"], -1, 1) / 255.0
        
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
