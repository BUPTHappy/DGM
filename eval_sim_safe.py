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
from omegaconf import open_dict
from unified_video_action.workspace.base_workspace import BaseWorkspace
from unified_video_action.utils.load_env import load_env_runner
from omegaconf import OmegaConf
from types import SimpleNamespace


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("--use_ucgm", is_flag=True, help="Enable UCGM mode.")
@click.option("-d", "--device", default="cuda:0")
@click.option('--pruning_ratios_file', type=str, required=False, help='List of lists input in JSON format')
@click.option(
    "--num_sampling_steps",
    type=int,
    default=None,
    show_default=True,
    help="Number of sampling steps to use."
)
@click.option(
    "--stochasticity_rate",
    type=float,
    default=None,
    show_default=True,
    help="Stochasticity rate for sampling."
)
@click.option(
    "--cfg_strength",
    type=float,
    default=None,
    show_default=True,
    help="Classifier-free guidance strength."
)
@click.option(
    "--temperature",
    type=float,
    default=None,
    show_default=True,
    help="Sampling temperature."
)
@click.option(
    "--rfba_gap_end",
    type=float,
    default=None,
    show_default=True,
    help="RFBA gap end value."
)
@click.option(
    "--extrapol_ratio",
    type=float,
    default=None,
    show_default=True,
    help="Extrapolation ratio."
)
def main(checkpoint, output_dir, device, pruning_ratios_file, use_ucgm, 
         num_sampling_steps, stochasticity_rate, cfg_strength, temperature, rfba_gap_end, extrapol_ratio):

    # Safe memory optimization settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,garbage_collection_threshold:0.5"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Enable for debugging
    
    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print("Starting safe memory-optimized evaluation...")
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print("Loading checkpoint...")
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    
    # Safe memory optimizations (preserve original sequence lengths)
    with open_dict(cfg):
        # Only reduce test episodes, keep sequence lengths intact
        if hasattr(cfg.task.env_runner, 'n_test'):
            cfg.task.env_runner.n_test = min(cfg.task.env_runner.n_test, 50)  # Moderate reduction
            print(f"✓ Reduced n_test to {cfg.task.env_runner.n_test}")
        
        # Keep original sequence lengths to avoid index errors
        print(f"✓ Keeping original sequence lengths: n_obs_steps={cfg.task.env_runner.n_obs_steps}, n_action_steps={cfg.task.env_runner.n_action_steps}")

    # Configure UCGM settings
    with open_dict(cfg.model.policy.autoregressive_model_params):
        if "ucgmts_config" not in cfg.model.policy.autoregressive_model_params:
            cfg.model.policy.autoregressive_model_params.ucgmts_config = OmegaConf.create({})
            cfg.model.policy.autoregressive_model_params.ucgmts_config.transport_type = "Linear"
            cfg.model.policy.autoregressive_model_params.ucgmts_config.scaled_cbs_eps = 0.0
            cfg.model.policy.autoregressive_model_params.ucgmts_config.ema_decay_rate = 0.0

        if stochasticity_rate is not None:
            cfg.model.policy.autoregressive_model_params.ucgmts_config.consistc_ratio = stochasticity_rate
        if num_sampling_steps:
            cfg.model.policy.autoregressive_model_params.num_sampling_steps = num_sampling_steps
            if num_sampling_steps <= 2:
                cfg.model.policy.autoregressive_model_params.ucgmts_config.rfba_gap_steps = [0.001, 0.5] 
            else:
                cfg.model.policy.autoregressive_model_params.ucgmts_config.rfba_gap_steps = [0.001, 0.001]
        
        # Apply Bayesian optimization parameters
        if cfg_strength is not None:
            cfg.model.policy.autoregressive_model_params.cfg = cfg_strength
        if temperature is not None:
            cfg.model.policy.autoregressive_model_params.temperature = temperature
        if rfba_gap_end is not None:
            cfg.model.policy.autoregressive_model_params.ucgmts_config.rfba_gap_steps = [0.001, rfba_gap_end]
        if extrapol_ratio is not None:
            cfg.model.policy.autoregressive_model_params.ucgmts_config.extrapol_ratio = extrapol_ratio
    
    if use_ucgm:
        OmegaConf.set_struct(cfg, False)
        cfg.model.policy.autoregressive_model_params.use_ucgm = True
        print("Using UCGM mode")

    if pruning_ratios_file is not None:
        with open(pruning_ratios_file, 'r') as f:
            OmegaConf.set_struct(cfg, False)
            cfg.model.policy.autoregressive_model_params.pruning_ratios = json.load(f)
        cfg.model.policy.autoregressive_model_params.token_pruning = True
        cfg.model.policy.autoregressive_model_params.restore_after_encoder = True
    else:
        OmegaConf.set_struct(cfg, False)
        cfg.model.policy.autoregressive_model_params.pruning_ratios = None
        cfg.model.policy.autoregressive_model_params.token_pruning = False
        cfg.model.policy.autoregressive_model_params.restore_after_encoder = False

    # Set seed
    seed = cfg.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    with open_dict(cfg):
        cfg.output_dir = output_dir

    # Configure workspace
    print("Creating workspace...")
    cls = hydra.utils.get_class(cfg.model._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace

    print("Loaded checkpoint from %s" % checkpoint)

    # Load model
    print("Loading model...")
    workspace.load_payload_new(payload, exclude_keys=None, include_keys=None)

    # Get policy from workspace
    if cfg.training.use_ema:
        print("Using EMA policy for evaluation.")
        policy = workspace.ema_model
    else:
        policy = workspace.model
        print("Using regular policy for evaluation.")
    
    # Move to device and optimize for evaluation
    policy.to(device)
    policy.eval()
    
    # Safe memory cleanup
    del payload
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    print(f"GPU Memory after model loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    # Patch the extract_latent_autoregressive function to use safe chunking
    from unified_video_action.utils import data_utils
    original_extract_latent = data_utils.extract_latent_autoregressive
    
    def safe_chunked_extract_latent(vae_model, x):
        return original_extract_latent(vae_model, x, chunk_size=8)  # Safe chunk size
    
    data_utils.extract_latent_autoregressive = safe_chunked_extract_latent
    print("✓ Enabled safe chunked VAE processing with chunk_size=8")
    
    # Configure environment runner
    if "libero" in cfg.task.name:
        cfg.task.env_runner.n_test = 10  # Moderate reduction for libero
    else:
        cfg.task.env_runner.n_test = 20  # Moderate reduction
        
    print("Creating environment runner...")
    env_runners = load_env_runner(cfg, output_dir)

    # Run evaluation with safe memory management
    print("Starting evaluation...")
    try:
        if "libero" in cfg.task.name:
            step_log = {}
            for env_runner in env_runners:
                runner_log = env_runner.run(policy)
                step_log.update(runner_log)
                print(step_log)
                
                # Safe cleanup after each runner
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            assert "test_mean_score" not in step_log
            all_test_mean_score = {
                k: v for k, v in step_log.items() if "test/" in k and "_mean_score" in k
            }
            step_log["test_mean_score"] = np.mean(list(all_test_mean_score.values()))

            runner_log = step_log
        else:
            env_runner = env_runners
            runner_log = env_runner.run(policy)
            
    except Exception as e:
        print(f"Error during evaluation: {e}")
        # Try to recover with cleanup
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        raise e

    # Restore original function
    data_utils.extract_latent_autoregressive = original_extract_latent

    # Dump log to json
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
    
    # Final cleanup
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    print(f"Final GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print("Safe memory-optimized evaluation completed!")


if __name__ == "__main__":
    main()
