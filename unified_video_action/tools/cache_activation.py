
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pickle
from pathlib import Path

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import your existing modules
try:
    from unified_video_action.dataset import get_dataset
    from unified_video_action.model.autoregressive.mar_con_unified import MAR
    from unified_video_action.model.autoregressive.diffusion_action_loss import DiffActLoss
    from unified_video_action.model.autoregressive.diffusion_loss import SimpleMLPAdaLN
    from unified_video_action.model.autoregressive.cross_attention_diffusion import CrossAttentionAdaLN
    from unified_video_action.utils.data_utils import resize_image
except ImportError as e:
    print(f"Import error: {e}")
    raise


class TeacherActivationCache:
    
    def __init__(self, model, device='cuda', dataset_path=None, vae_model=None):
        self.model = model
        self.device = device
        self.dataset_path = dataset_path
        self.vae_model = vae_model
        self.model.eval()
        
        # Hook to capture activations
        self.activations = {}
        self.hooks = []
        
    def register_hooks(self):
        """注册钩子函数来捕获激活"""
        
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach().cpu()
            return hook
        
        print("Registering hooks...")
        
        # Debug: Check model structure
        print(f"Model has diffactloss: {hasattr(self.model, 'diffactloss')}")
        if hasattr(self.model, 'diffactloss'):
            print(f"diffactloss has net: {hasattr(self.model.diffactloss, 'net')}")
            if hasattr(self.model.diffactloss, 'net'):
                net = self.model.diffactloss.net
                print(f"Net type: {type(net)}")
                print(f"Is SimpleMLPAdaLN: {isinstance(net, SimpleMLPAdaLN)}")
                print(f"Is CrossAttentionAdaLN: {isinstance(net, CrossAttentionAdaLN)}")
        
        # Hook the diffusion action head components
        if hasattr(self.model, 'diffactloss') and hasattr(self.model.diffactloss, 'net'):
            net = self.model.diffactloss.net
            
            # Hook SimpleMLPAdaLN components
            if isinstance(net, SimpleMLPAdaLN):
                print("Hooking SimpleMLPAdaLN components...")
                # Hook input projection
                self.hooks.append(net.input_proj.register_forward_hook(get_activation('input_proj')))
                
                # Hook each residual block
                for i, block in enumerate(net.res_blocks):
                    self.hooks.append(block.register_forward_hook(get_activation(f'res_block_{i}')))
                
                # Hook final layer
                self.hooks.append(net.final_layer.register_forward_hook(get_activation('final_layer')))
                
            elif isinstance(net, CrossAttentionAdaLN):
                print("Hooking CrossAttentionAdaLN components...")
                # Hook input projection
                self.hooks.append(net.input_proj.register_forward_hook(get_activation('input_proj')))
                
                # Hook video projection
                self.hooks.append(net.video_proj.register_forward_hook(get_activation('video_proj')))
                
                # Hook each cross attention block
                for i, block in enumerate(net.cross_attn_blocks):
                    self.hooks.append(block.register_forward_hook(get_activation(f'cross_attn_block_{i}')))
                
                # Hook final layer
                self.hooks.append(net.final_layer.register_forward_hook(get_activation('final_layer')))
            else:
                print(f"Unknown net type: {type(net)}")
        else:
            print("No diffactloss.net found, trying alternative hook locations...")
            # Try hooking other components
            if hasattr(self.model, 'diffactloss'):
                print("Model has diffactloss but no net attribute")
            else:
                print("Model has no diffactloss attribute")
        
        print(f"Registered {len(self.hooks)} hooks")
    
    def remove_hooks(self):
        """移除钩子函数"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def cache_activations(self, dataloader, num_samples=8000):
        """缓存激活数据"""
        cached_data = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Caching activations")):
                if len(cached_data) >= num_samples:
                    break
                
                # Clear activations before each batch
                self.activations.clear()
                
                # Move batch to device - handle nested structures
                def move_to_device(obj, device):
                    if isinstance(obj, torch.Tensor):
                        return obj.to(device)
                    elif isinstance(obj, dict):
                        return {k: move_to_device(v, device) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return type(obj)(move_to_device(item, device) for item in obj)
                    else:
                        return obj
                
                batch = move_to_device(batch, self.device)
                
                # Forward pass
                try:
                    # Try different forward call patterns
                    if hasattr(self.model, 'forward'):
                        # Handle different data formats
                        if 'image' in batch:
                            # Direct image format
                            imgs = batch['image']
                        elif 'obs' in batch and 'image' in batch['obs']:
                            # Nested obs format
                            imgs = batch['obs']['image']
                        elif 'obs' in batch and 'agentview_rgb' in batch['obs']:
                            # Libero dataset uses agentview_rgb
                            imgs = batch['obs']['agentview_rgb']
                        else:
                            raise KeyError("Could not find image data in batch")
                        
                        # Use the original library's resize_image function for consistency
                        print(f"Original batch keys: {batch.keys()}")
                        if 'obs' in batch:
                            print(f"Obs keys: {batch['obs'].keys()}")
                        
                        # Create a minimal config object for resize_image function
                        class SimpleConfig:
                            class Task:
                                def __init__(self, name):
                                    self.name = name
                            def __init__(self, task_name):
                                self.task = self.Task(task_name)
                        
                        # Determine task name from dataset path
                        if 'libero' in str(self.dataset_path).lower():
                            cfg = SimpleConfig('libero')
                        elif 'pusht' in str(self.dataset_path).lower():
                            cfg = SimpleConfig('pusht')
                        elif 'umi' in str(self.dataset_path).lower():
                            cfg = SimpleConfig('umi')
                        else:
                            cfg = SimpleConfig('other')
                        
                        # Apply the original library's resize function
                        print(f"Using task config: {cfg.task.name}")
                        batch = resize_image(cfg, batch)
                        
                        # Now get the processed image
                        imgs = batch['obs']['image']
                        print(f"After resize_image - imgs.shape: {imgs.shape}")
                        
                        # Ensure we have 4 frames for the model
                        if len(imgs.shape) == 5:  # [B, T, C, H, W] format
                            batch_size, time_dim = imgs.shape[0], imgs.shape[1]
                            
                            # Ensure we have 4 frames
                            if time_dim != 4:
                                if time_dim == 1:
                                    # Repeat single frame 4 times
                                    imgs = imgs.repeat(1, 4, 1, 1, 1)
                                    print(f"Repeated single frame to 4 frames - imgs.shape: {imgs.shape}")
                                else:
                                    # Take first 4 frames or pad with last frame
                                    if time_dim > 4:
                                        imgs = imgs[:, :4, :, :, :]
                                    else:
                                        # Pad with last frame
                                        last_frame = imgs[:, -1:, :, :, :]
                                        padding_frames = last_frame.repeat(1, 4 - time_dim, 1, 1, 1)
                                        imgs = torch.cat([imgs, padding_frames], dim=1)
                                    print(f"Adjusted to 4 frames - imgs.shape: {imgs.shape}")
                                    
                            batch['obs']['image'] = imgs
                        
                        elif len(imgs.shape) == 4:  # [B, C, H, W] format
                            # Add time dimension by repeating the image
                            imgs = imgs.unsqueeze(1).repeat(1, 4, 1, 1, 1)  # Repeat 4 times for 4 frames
                            print(f"Added time dimension - imgs.shape: {imgs.shape}")
                            batch['obs']['image'] = imgs
                        
                        else:
                            raise ValueError(f"Unexpected image shape: {imgs.shape}")
                    
                        print("Performing VAE encoding first...")
                        
                        # Prepare inputs for VAE encoding
                        imgs = batch['obs']['image']  # [B, T, C, H, W]
                        cond = imgs  # Use same images for condition
                        
                        # Handle action data
                        if 'action' in batch:
                            actions = batch['action']
                            print(f"Found action data: {actions.shape}")
                            
                            # Handle different action data formats
                            if len(actions.shape) == 3:  # [B, T, A] format
                                # Take the last action or average across time
                                if actions.shape[1] > 1:
                                    actions = actions[:, -1, :]  # Take last action
                                else:
                                    actions = actions.squeeze(1)  # Remove time dimension
                            elif len(actions.shape) == 2:  # [B, A] format
                                pass  # Already correct
                            else:
                                raise ValueError(f"Unexpected action shape: {actions.shape}")
                            
                            print(f"Reshaped action data: {actions.shape}")
                        else:
                            # Create dummy actions if not available
                            actions = torch.zeros(imgs.shape[0], 2, device=imgs.device)
                            print("Created dummy actions")
                        
                        # Perform VAE encoding first
                        try:
                            with torch.no_grad():
                                # Import the VAE encoding function
                                from unified_video_action.utils.data_utils import extract_latent_autoregressive
                                
                                # VAE encode the images
                                # Convert from [B, T, C, H, W] to [B, C, T, H, W] for VAE
                                imgs_for_vae = imgs.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
                                cond_for_vae = cond.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
                                
                                # Use the VAE model passed to the cache
                                if self.vae_model is None:
                                    print("VAE model not available")
                                    raise NotImplementedError("VAE model not available")
                                
                                # Encode images to latents
                                z, latent_size = extract_latent_autoregressive(self.vae_model, imgs_for_vae)
                                c, _ = extract_latent_autoregressive(self.vae_model, cond_for_vae)
                                
                                print(f"VAE encoded - z.shape: {z.shape}, c.shape: {c.shape}")
                                
                                # Now call model with VAE-encoded latents
                                # extract_latent_autoregressive already returns [B, T, C, H, W] format
                                # No permutation needed - the model expects this exact format
                                
                                print(f"Final model inputs - z.shape: {z.shape}, c.shape: {c.shape}")
                                
                                # Debug: Check what happens in patchify
                                print(f"Before patchify - z: {z.shape}, c: {c.shape}")
                                output = self.model(z, c, nactions=actions, task_mode='policy_model')
                                print("Model forward successful!")
                                
                        except Exception as e:
                            print(f"VAE encoding or model forward failed: {e}")
                            continue
                        
                        # Model forward was successful, now capture activations
                        print("Model forward completed successfully!")
                        
                        # Handle different return formats
                        print(f"Model output type: {type(output)}, length: {len(output) if isinstance(output, tuple) else 'N/A'}")
                        if isinstance(output, tuple):
                            if len(output) == 3:
                                loss, video_loss, act_loss = output
                                print(f"Got 3 outputs: loss={loss}, video_loss={video_loss}, act_loss={act_loss}")
                            elif len(output) == 2:
                                # For policy_model, the model returns (loss, act_loss) where loss = act_loss
                                loss, act_loss = output
                                video_loss = torch.tensor(0.0)
                                print(f"Got 2 outputs: loss={loss}, act_loss={act_loss}")
                            else:
                                loss = output[0]
                                video_loss = torch.tensor(0.0)
                                act_loss = torch.tensor(0.0)
                                print(f"Got {len(output)} outputs, using first as loss")
                        else:
                            loss = output
                            video_loss = torch.tensor(0.0)
                            act_loss = torch.tensor(0.0)
                            print(f"Got single output: loss={loss}")
                    else:
                        raise AttributeError("Model has no forward method")
                    
                    # Only store if we have activations
                    if self.activations:
                        batch_data = {
                            'batch_idx': batch_idx,
                            'activations': self.activations.copy(),
                            'input_shape': imgs.shape,
                            'cond_shape': batch.get('cond', torch.zeros_like(imgs[:, :1])).shape,
                            'action_shape': actions.shape,
                        }
                        
                        cached_data.append(batch_data)
                    else:
                        print(f"Warning: No activations captured for batch {batch_idx}")
                    
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    # Clear activations on error to prevent accumulation
                    self.activations.clear()
                    continue
        
        return cached_data


def main():
    parser = argparse.ArgumentParser(description='Cache teacher activations for distillation')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to teacher model checkpoint')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to dataset')
    parser.add_argument('--out', type=str, required=True,
                       help='Output directory for cached activations')
    parser.add_argument('--num_samples', type=int, default=8000,
                       help='Number of samples to cache (default: 8000)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for caching (default: 32)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')
    parser.add_argument('--num_gpus', type=int, default=1,
                       help='Number of GPUs to use for parallel processing (default: 1)')
    parser.add_argument('--gpu_ids', type=str, default=None,
                       help='Specific GPU IDs to use (e.g., "0,1,2,3"). If not specified, uses first num_gpus GPUs')
    
    args = parser.parse_args()
    
    # Setup GPU configuration
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        args.num_gpus = len(gpu_ids)
    else:
        gpu_ids = list(range(args.num_gpus))
    
    print(f"Using {args.num_gpus} GPUs: {gpu_ids}")
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    
    # Load teacher model
    print(f"Loading teacher model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Extract model config from checkpoint - try multiple possible keys
    model_config = None
    for key in ['model_config', 'config', 'hyper_parameters', 'hparams']:
        if key in checkpoint:
            model_config = checkpoint[key]
            print(f"Found model config in checkpoint key: {key}")
            break
    
    print("FORCING use of fixed model config for Libero dataset compatibility")
    model_config = None  # Force use of our fixed config below
    
    if model_config is None:
        print("Warning: No model config found in checkpoint, using defaults")
        model_config = {
            'task_name': 'libero',
            'different_history_freq': False,
            'use_history_action': False,
            'action_mask_ratio': 0.5,
            'use_proprioception': False,
            'predict_wrist_img': False,
            'predict_proprioception': False,
            'img_size': 256,  # VAE model requires 256x256 input
            'vae_stride': 16,
            'patch_size': 1,
            'encoder_embed_dim': 1024,
            'decoder_embed_dim': 1024,
            'predict_action': True,
            'language_emb_model': None,
            'shape_meta': {
                'action': {
                    'shape': [7]  # Libero action dimension
                },
                'obs': {
                    'agentview_rgb': {
                        'shape': [3, 128, 128],  # Original Libero image size
                        'type': 'rgb'
                    },
                    'ee_pos': {
                        'shape': [3],
                        'type': 'low_dim'
                    }
                }
            },
            'action_model_params': {
                'predict_action': True,
                'act_model_type': 'conv_fc',  # Use original MLP head
                'num_attention_heads': 8
            }
        }
    
    # Create model with safe parameter handling
    try:
        print(f"Creating model with config: img_size={model_config['img_size']}")
        model = MAR(**model_config)
    except Exception as e:
        print(f"Error creating model with config: {e}")
        print("Falling back to default config...")
        model_config = {
            'task_name': 'libero',
            'different_history_freq': False,
            'use_history_action': False,
            'action_mask_ratio': 0.5,
            'use_proprioception': False,
            'predict_wrist_img': False,
            'predict_proprioception': False,
            'img_size': 256,  # VAE model requires 256x256 input
            'vae_stride': 16,
            'patch_size': 1,
            'encoder_embed_dim': 1024,
            'decoder_embed_dim': 1024,
            'predict_action': True,
            'language_emb_model': None,
            'shape_meta': {
                'action': {
                    'shape': [7]  # Libero action dimension
                },
                'obs': {
                    'agentview_rgb': {
                        'shape': [3, 128, 128],  # Original Libero image size
                        'type': 'rgb'
                    },
                    'ee_pos': {
                        'shape': [3],
                        'type': 'low_dim'
                    }
                }
            },
            'action_model_params': {
                'predict_action': True,
                'act_model_type': 'conv_fc',
                'num_attention_heads': 8
            }
        }
        model = MAR(**model_config)
    
    # Safe state dict loading
    model_state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', {}))
    if not model_state_dict:
        print("Warning: No model state dict found in checkpoint")
        model_state_dict = {}
    
    # Load with strict=False and handle missing keys
    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
    
    if missing_keys:
        print(f"Missing keys in checkpoint: {len(missing_keys)} keys")
        if len(missing_keys) <= 10:  # Only print if not too many
            print(f"Missing keys: {missing_keys}")
    
    if unexpected_keys:
        print(f"Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
        if len(unexpected_keys) <= 10:  # Only print if not too many
            print(f"Unexpected keys: {unexpected_keys}")
    
    # Setup model for multi-GPU if needed
    if args.num_gpus > 1:
        print(f"Setting up model for {args.num_gpus} GPUs")
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        primary_device = f'cuda:{gpu_ids[0]}'
    else:
        primary_device = args.device
    
    model = model.to(primary_device)
    model.eval()
    
    # Load VAE model
    print("Loading VAE model...")
    from unified_video_action.vae.vaekl import AutoencoderKL
    
    # VAE model parameters (matching the pretrained checkpoint)
    vae_model_params = {
        'autoencoder_path': 'pretrained_models/vae/kl16.ckpt',
        'ddconfig': type('obj', (object,), {
            'vae_embed_dim': 16,  # Must match the pretrained checkpoint
            'ch_mult': (1, 1, 2, 2, 4),
            'num_res_blocks': 2,
            'attn_resolutions': (16,),
            'dropout': 0.0,
            'resamp_with_conv': True,
            'in_channels': 3,
            'resolution': 256,
            'z_channels': 16,  # Must match the pretrained checkpoint
            'double_z': True,
        })(),
        'use_variational': True,
    }
    
    with torch.no_grad():
        vae_model = AutoencoderKL(**vae_model_params)
    vae_model.eval()
    for param in vae_model.parameters():
        param.requires_grad = False
    vae_model = vae_model.to(primary_device)
    print("VAE model loaded successfully")
    
    # Create activation cache
    cache = TeacherActivationCache(model, primary_device, args.dataset, vae_model)
    cache.register_hooks()
    
    # Load dataset with proper error handling
    print(f"Loading dataset from {args.dataset}")
    try:
        # Try different dataset loading approaches
        if hasattr(get_dataset, '__call__'):
            # If get_dataset is a function, try calling it
            try:
                dataset = get_dataset(args.dataset)
            except Exception as e:
                print(f"Error calling get_dataset({args.dataset}): {e}")
                # Fallback: try with additional parameters
                try:
                    dataset = get_dataset(args.dataset, split='train')
                except Exception as e2:
                    print(f"Error with split parameter: {e2}")
                    raise e
        else:
            # If get_dataset is a class, instantiate it
            dataset = get_dataset(args.dataset)
        
        # Verify dataset is properly loaded
        if not hasattr(dataset, '__len__'):
            raise AttributeError("Dataset does not have __len__ method")
        if not hasattr(dataset, '__getitem__'):
            raise AttributeError("Dataset does not have __getitem__ method")
        
        print(f"Dataset loaded successfully: {len(dataset)} samples")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check your dataset path and ensure get_dataset function is properly imported")
        raise
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Cache activations with proper error handling
    print(f"Caching activations for {args.num_samples} samples...")
    try:
        cached_data = cache.cache_activations(dataloader, args.num_samples)
        
        if not cached_data:
            raise RuntimeError("No activations were cached. Check model hooks and forward pass.")
        
        # Save cached data with memory optimization
        output_file = os.path.join(args.out, 'teacher_activations.pkl')
        
        # Process activations to reduce memory usage
        processed_data = []
        for batch_data in cached_data:
            processed_batch = {
                'batch_idx': batch_data['batch_idx'],
                'input_shape': batch_data['input_shape'],
                'cond_shape': batch_data['cond_shape'],
                'action_shape': batch_data['action_shape'],
            }
            
            # Process activations to reduce memory
            processed_activations = {}
            for key, tensor in batch_data['activations'].items():
                # Convert to float16 to save memory
                if tensor.dtype == torch.float32:
                    tensor = tensor.half()
                # Move to CPU if on GPU
                if tensor.is_cuda:
                    tensor = tensor.cpu()
                processed_activations[key] = tensor
            
            processed_batch['activations'] = processed_activations
            processed_data.append(processed_batch)
        
        # Save with compression
        print(f"Saving cached data to {output_file}...")
        with open(output_file, 'wb') as f:
            pickle.dump(processed_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Cached {len(processed_data)} samples to {output_file}")
        
        # Calculate and print memory usage
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        print(f"File size: {file_size:.2f} MB")
        
        # Save metadata
        metadata = {
            'num_samples': len(cached_data),
            'model_config': model_config,
            'checkpoint_path': args.checkpoint,
            'dataset_path': args.dataset,
            'batch_size': args.batch_size,
        }
        
        metadata_file = os.path.join(args.out, 'metadata.pkl')
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Metadata saved to {metadata_file}")
        
    except Exception as e:
        print(f"Error during caching: {e}")
        raise
    finally:
        try:
            cache.remove_hooks()
            print("Hooks removed successfully")
        except Exception as e:
            print(f"Warning: Error removing hooks: {e}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cleared")


if __name__ == '__main__':
    main()
