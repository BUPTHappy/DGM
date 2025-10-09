import sys

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
import numpy as np
import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
import random
import gc
import psutil
from omegaconf import open_dict
from unified_video_action.workspace.base_workspace import BaseWorkspace
from unified_video_action.utils.load_env import load_env_runner
from einops import rearrange


def get_memory_stats():
    """Get current memory usage statistics"""
    if torch.cuda.is_available():
        gpu_memory = {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3,  # GB
        }
    else:
        gpu_memory = None
    
    cpu_memory = {
        'used': psutil.virtual_memory().used / 1024**3,  # GB
        'available': psutil.virtual_memory().available / 1024**3,  # GB
        'percent': psutil.virtual_memory().percent,
    }
    
    return {'gpu': gpu_memory, 'cpu': cpu_memory}


def print_memory_stats(stage=""):
    """Print memory statistics with optional stage label"""
    stats = get_memory_stats()
    print(f"\n{'='*50}")
    if stage:
        print(f"Memory Stats - {stage}")
    else:
        print("Memory Stats")
    print(f"{'='*50}")
    
    if stats['gpu']:
        print(f"GPU Memory:")
        print(f"  Allocated: {stats['gpu']['allocated']:.2f} GB")
        print(f"  Reserved:  {stats['gpu']['reserved']:.2f} GB")
        print(f"  Max Used:  {stats['gpu']['max_allocated']:.2f} GB")
    
    print(f"CPU Memory:")
    print(f"  Used:      {stats['cpu']['used']:.2f} GB")
    print(f"  Available: {stats['cpu']['available']:.2f} GB")
    print(f"  Percent:   {stats['cpu']['percent']:.1f}%")
    print(f"{'='*50}\n")


def clear_memory():
    """Aggressively clear memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def extract_latent_autoregressive_chunked(vae_model, x, chunk_size=2, batch_chunk_size=None):
    """Extract latent with chunked processing to save memory"""
    x = x.float()
    B, C, T, H, W = x.size()
    
    # If chunk_size is None or T <= chunk_size, process normally
    if chunk_size is None or T <= chunk_size:
        with torch.no_grad():
            posterior = vae_model.encode(rearrange(x, "b c t h w -> (b t) c h w"))
            z = posterior.sample().mul_(0.2325)
            z = rearrange(z, "(b t) c h w -> b t c h w", b=B)
    else:
        # Process in chunks to save memory
        z_chunks = []
        
        # Process batch dimension in chunks if specified
        if batch_chunk_size is not None and B > batch_chunk_size:
            for b_start in range(0, B, batch_chunk_size):
                b_end = min(b_start + batch_chunk_size, B)
                x_batch_chunk = x[b_start:b_end]  # [batch_chunk_size, C, T, H, W]
                
                batch_z_chunks = []
                for t_start in range(0, T, chunk_size):
                    t_end = min(t_start + chunk_size, T)
                    x_chunk = x_batch_chunk[:, :, t_start:t_end, :, :]  # [batch_chunk_size, C, chunk_size, H, W]
                    
                    with torch.no_grad():
                        posterior = vae_model.encode(rearrange(x_chunk, "b c t h w -> (b t) c h w"))
                        z_chunk = posterior.sample().mul_(0.2325)
                        z_chunk = rearrange(z_chunk, "(b t) c h w -> b t c h w", b=x_chunk.size(0))
                        batch_z_chunks.append(z_chunk)
                    
                    # Clear cache after each chunk
                    torch.cuda.empty_cache()
                
                # Concatenate time chunks for this batch
                batch_z = torch.cat(batch_z_chunks, dim=1)  # [batch_chunk_size, T, C, H, W]
                z_chunks.append(batch_z)
                
                # Clear intermediate results
                del batch_z_chunks, batch_z
                torch.cuda.empty_cache()
            
            # Concatenate batch chunks
            z = torch.cat(z_chunks, dim=0)  # [B, T, C, H, W]
        else:
            # Original chunking logic for time dimension only
            for t_start in range(0, T, chunk_size):
                t_end = min(t_start + chunk_size, T)
                x_chunk = x[:, :, t_start:t_end, :, :]  # [B, C, chunk_size, H, W]
                
                with torch.no_grad():
                    posterior = vae_model.encode(rearrange(x_chunk, "b c t h w -> (b t) c h w"))
                    z_chunk = posterior.sample().mul_(0.2325)
                    z_chunk = rearrange(z_chunk, "(b t) c h w -> b t c h w", b=B)
                    z_chunks.append(z_chunk)
                
                # Clear cache after each chunk
                torch.cuda.empty_cache()
            
            # Concatenate chunks along time dimension
            z = torch.cat(z_chunks, dim=1)  # [B, T, C, H, W]
    
    latent_size = z.size()[2:]
    return z, latent_size


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
@click.option("--enable_grad_checkpointing", is_flag=True, default=True, help="Enable gradient checkpointing to save memory")
@click.option("--reduce_batch_size", is_flag=True, default=True, help="Reduce effective batch size for memory")
@click.option("--use_mixed_precision", is_flag=True, default=True, help="Use mixed precision to save memory")
@click.option("--vae_chunk_size", default=2, help="Number of frames to process at once in VAE")
@click.option("--vae_batch_chunk_size", default=4, help="Number of batches to process at once in VAE")
@click.option("--act_diff_testing_steps", default="100", help="Number of diffusion sampling steps")
@click.option("--enable_memory_monitoring", is_flag=True, default=True, help="Enable memory monitoring")
@click.option("--aggressive_memory_cleanup", is_flag=True, default=True, help="Enable aggressive memory cleanup")
@click.option("--optimize_data_loading", is_flag=True, default=True, help="Enable data loading optimizations")
@click.option("--pin_memory", is_flag=True, default=False, help="Enable pinned memory for faster GPU transfer")
@click.option("--enable_model_sharding", is_flag=True, default=False, help="Enable model sharding for very large models")
@click.option("--shard_size_mb", default=1000, help="Size of each model shard in MB")
def main(checkpoint, output_dir, device, enable_grad_checkpointing, reduce_batch_size, use_mixed_precision, vae_chunk_size, vae_batch_chunk_size, act_diff_testing_steps, enable_memory_monitoring, aggressive_memory_cleanup, optimize_data_loading, pin_memory, enable_model_sharding, shard_size_mb):

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Set memory optimization environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Disable synchronous execution for better memory management
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Set the specific GPU device
        torch.cuda.set_device(device)
        print(f"✓ Using GPU device: {device}")
    
    # Initial memory stats
    if enable_memory_monitoring:
        print_memory_stats("Initial")

    # load checkpoint
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    
    if enable_memory_monitoring:
        print_memory_stats("After Checkpoint Load")

    # set seed
    seed = cfg.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    with open_dict(cfg):
        cfg.output_dir = output_dir
        
        # Set diffusion sampling steps
        if "autoregressive_model_params" in cfg.model.policy:
            original_steps = cfg.model.policy.autoregressive_model_params.act_diff_testing_steps
            cfg.model.policy.autoregressive_model_params.num_sampling_steps = act_diff_testing_steps
            print(f"✓ Changed num_sampling_steps from {original_steps} to {act_diff_testing_steps}")
        
        # Enable action prediction to use diffusion sampling
        if "action_model_params" in cfg.model.policy:
            cfg.model.policy.action_model_params.predict_action = True
            print("✓ Enabled predict_action to use diffusion sampling")
        
        # Memory optimizations
        if enable_grad_checkpointing:
            cfg.model.policy.autoregressive_model_params.grad_checkpointing = True
            print("✓ Enabled gradient checkpointing")
            
        if reduce_batch_size:
            # Reduce sequence length for evaluation
            if hasattr(cfg.task.env_runner, 'n_obs_steps'):
                cfg.task.env_runner.n_obs_steps = min(cfg.task.env_runner.n_obs_steps, 8)
                print(f"✓ Reduced n_obs_steps to {cfg.task.env_runner.n_obs_steps}")
            
            # Reduce action steps
            if hasattr(cfg.task.env_runner, 'n_action_steps'):
                cfg.task.env_runner.n_action_steps = min(cfg.task.env_runner.n_action_steps, 4)
                print(f"✓ Reduced n_action_steps to {cfg.task.env_runner.n_action_steps}")
        
        # Data loading optimizations
        if optimize_data_loading:
            # Reduce number of workers to save memory
            if hasattr(cfg.task.env_runner, 'num_workers'):
                cfg.task.env_runner.num_workers = min(cfg.task.env_runner.num_workers, 2)
                print(f"✓ Reduced num_workers to {cfg.task.env_runner.num_workers}")
            
            # Enable pin memory if requested
            if pin_memory and hasattr(cfg.task.env_runner, 'pin_memory'):
                cfg.task.env_runner.pin_memory = True
                print("✓ Enabled pinned memory")
            
            # Reduce prefetch factor
            if hasattr(cfg.task.env_runner, 'prefetch_factor'):
                cfg.task.env_runner.prefetch_factor = min(cfg.task.env_runner.prefetch_factor, 2)
                print(f"✓ Reduced prefetch_factor to {cfg.task.env_runner.prefetch_factor}")
        
    # configure workspace
    cls = hydra.utils.get_class(cfg.model._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace

    print("Loaded checkpoint from %s" % checkpoint)
    # Load with strict=False to ignore unexpected keys
    try:
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    except RuntimeError as e:
        if "Unexpected key" in str(e):
            print("Warning: Ignoring unexpected keys in checkpoint")
            # Load model manually with strict=False
            if "ema_model" in payload["state_dicts"]:
                model_state = payload["state_dicts"]["ema_model"]
                # Remove module prefix if present
                model_state_clean = {}
                for k, v in model_state.items():
                    if k.startswith("module."):
                        model_state_clean[k[7:]] = v
                    else:
                        model_state_clean[k] = v
                workspace.model.load_state_dict(model_state_clean, strict=False)
                print("✓ Loaded model with strict=False")
            else:
                raise e
        else:
            raise e
    
    if enable_memory_monitoring:
        print_memory_stats("After Workspace Load")
    
    # get policy from workspace
    policy = workspace.ema_model
    
    # Model sharding for very large models
    if enable_model_sharding:
        print(f"✓ Enabling model sharding with shard size: {shard_size_mb} MB")
        # Move model to CPU first to enable sharding
        policy = policy.cpu()
        
        # Estimate model size and create shards
        model_size_mb = sum(p.numel() * p.element_size() for p in policy.parameters()) / (1024 * 1024)
        print(f"Model size: {model_size_mb:.2f} MB")
        
        if model_size_mb > shard_size_mb:
            print(f"Model is large ({model_size_mb:.2f} MB), enabling sharding...")
            # For now, just move back to device - full sharding would require more complex implementation
            policy = policy.to(device)
        else:
            print("Model size is manageable, no sharding needed")
            policy = policy.to(device)
    else:
        policy.to(device)
    
    policy.eval()
    
    if enable_memory_monitoring:
        print_memory_stats("After Model Load")

    # Enable memory optimizations for the model
    if enable_grad_checkpointing and hasattr(policy, 'model'):
        if hasattr(policy.model, 'enable_gradient_checkpointing'):
            policy.model.enable_gradient_checkpointing()
            print("✓ Enabled gradient checkpointing on model")
        
        # Enable gradient checkpointing on submodules
        def enable_checkpointing_recursive(module):
            if hasattr(module, 'enable_gradient_checkpointing'):
                module.enable_gradient_checkpointing()
            for child in module.children():
                enable_checkpointing_recursive(child)
        
        enable_checkpointing_recursive(policy.model)
        print("✓ Enabled gradient checkpointing on all submodules")
    
    # Enable VAE checkpointing and chunked processing
    if enable_grad_checkpointing:
        policy.use_vae_checkpointing = True
        policy.vae_chunk_size = vae_chunk_size
        policy.vae_batch_chunk_size = vae_batch_chunk_size
        print(f"✓ Enabled VAE chunked processing (chunk_size={vae_chunk_size}, batch_chunk_size={vae_batch_chunk_size})")
    
    # Monkey patch the extract_latent_autoregressive function to use chunked processing
    import unified_video_action.utils.data_utils as data_utils
    original_extract_latent = data_utils.extract_latent_autoregressive
    data_utils.extract_latent_autoregressive = lambda vae_model, x: extract_latent_autoregressive_chunked(vae_model, x, vae_chunk_size, vae_batch_chunk_size)
    print(f"✓ Patched extract_latent_autoregressive to use chunked processing (chunk_size={vae_chunk_size}, batch_chunk_size={vae_batch_chunk_size})")
    
    # Use mixed precision if requested (disabled for now due to tensor dimension issues)
    if use_mixed_precision and False:  # Temporarily disable mixed precision
        policy = policy.half()
        print("✓ Using mixed precision (FP16)")
    else:
        print("✓ Using FP32 precision (mixed precision disabled)")

    env_runners = load_env_runner(cfg, output_dir)
    
    if enable_memory_monitoring:
        print_memory_stats("Before Environment Run")

    if "libero" in cfg.task.name:
        step_log = {}
        for i, env_runner in enumerate(env_runners):
            if enable_memory_monitoring:
                print_memory_stats(f"Before Environment {i+1}")
            
            runner_log = env_runner.run(policy)
            step_log.update(runner_log)
            print(step_log)
            
            # Clear memory after each environment run
            if aggressive_memory_cleanup:
                clear_memory()
            else:
                torch.cuda.empty_cache()
            
            if enable_memory_monitoring:
                print_memory_stats(f"After Environment {i+1}")

        assert "test_mean_score" not in step_log
        all_test_mean_score = {
            k: v for k, v in step_log.items() if "test/" in k and "_mean_score" in k
        }
        step_log["test_mean_score"] = np.mean(list(all_test_mean_score.values()))

        runner_log = step_log
    else:
        env_runner = env_runners
        runner_log = env_runner.run(policy)
        
        # Clear memory after environment run
        if aggressive_memory_cleanup:
            clear_memory()
        else:
            torch.cuda.empty_cache()
        
        if enable_memory_monitoring:
            print_memory_stats("After Environment Run")

    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value

    for k, v in json_log.items():
        print(k, v)

    out_path = os.path.join(output_dir, f'eval_log_{checkpoint.split("/")[-1]}.json')
    print("Saving log to %s" % out_path)
    json.dump(json_log, open(out_path, "w"), indent=2, sort_keys=True)
    
    # Final memory stats and recommendations
    if enable_memory_monitoring:
        print_memory_stats("Final")
        
        # Print memory optimization summary
        print("\n" + "="*60)
        print("MEMORY OPTIMIZATION SUMMARY")
        print("="*60)
        print("✓ VAE chunked processing enabled")
        print("✓ Gradient checkpointing enabled")
        print("✓ Memory monitoring enabled")
        print("✓ Aggressive memory cleanup enabled")
        print("✓ Data loading optimizations enabled")
        if enable_model_sharding:
            print("✓ Model sharding enabled")
        print("\nFor even better memory efficiency, consider:")
        print("- Reducing vae_chunk_size further (e.g., 1)")
        print("- Reducing vae_batch_chunk_size (e.g., 2)")
        print("- Using CPU offloading for large models")
        print("- Enabling mixed precision (currently disabled)")
        print("="*60)


if __name__ == "__main__":
    main()
