import sys
import os

# CRITICAL: Set TOKENIZERS_PARALLELISM before any imports that might use tokenizers
# This must be set before forking processes (AsyncVectorEnv) to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
import numpy as np
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
import random
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
@click.option(
    "--dataset_path",
    type=str,
    required=False,
    help="Override dataset path (directory containing LIBERO *.hdf5).",
)
@click.option(
    "--no_ema",
    is_flag=True,
    help="Disable EMA policy and use raw model for evaluation.",
)
@click.option(
    "--n_test",
    type=int,
    default=None,
    show_default=True,
    help="Override number of test rollouts per task in Libero.",
)
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
    "--window_size",
    type=int,
    default=None,
    show_default=True,
    help="Window size for local causal attention."
)
@click.option(
    "--lambda_local",
    type=float,
    default=None,
    show_default=True,
    help="Lambda parameter for local feature fusion."
)
def main(checkpoint, output_dir, device, dataset_path, no_ema, n_test, pruning_ratios_file, use_ucgm, num_sampling_steps, stochasticity_rate, window_size, lambda_local):
    import time
    start_time = time.time()
    print(f"[eval_sim] Starting evaluation at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[eval_sim] Checkpoint: {checkpoint}")
    print(f"[eval_sim] Output dir: {output_dir}")
    print(f"[eval_sim] Device: {device}")
    print(f"[eval_sim] n_test: {n_test}")
    
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load checkpoint
    print(f"[eval_sim] Loading checkpoint...")
    checkpoint_load_start = time.time()
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    print(f"[eval_sim] Checkpoint loaded in {time.time() - checkpoint_load_start:.2f} seconds")
    

    #cfg.model.policy.autoregressive_model_params.num_sampling_steps = str(num_sampling_steps)
    #cfg.model.policy.autoregressive_model_params.act_diff_testing_steps = str(num_sampling_steps)

    with open_dict(cfg.model.policy.autoregressive_model_params):
        if "ucgmts_config" not in cfg.model.policy.autoregressive_model_params:
            # create as a DictConfig; empty dict is fine
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
        
        # 处理local attention参数
        if window_size is not None:
            cfg.model.policy.autoregressive_model_params.window_size = window_size
        if lambda_local is not None:
            cfg.model.policy.autoregressive_model_params.lambda_local = lambda_local
    
        

    
    if use_ucgm:
        OmegaConf.set_struct(cfg, False)  # Allow new keys
        cfg.model.policy.autoregressive_model_params.use_ucgm = True
        print("Using UCGM mode")

    if pruning_ratios_file is not None:
        with open(pruning_ratios_file, 'r') as f:
            OmegaConf.set_struct(cfg, False)  # Allow new keys
            cfg.model.policy.autoregressive_model_params.pruning_ratios = json.load(f)
        cfg.model.policy.autoregressive_model_params.token_pruning = True
        cfg.model.policy.autoregressive_model_params.restore_after_encoder = True
    else:
        OmegaConf.set_struct(cfg, False)  # Allow new keys
        cfg.model.policy.autoregressive_model_params.pruning_ratios = None
        cfg.model.policy.autoregressive_model_params.token_pruning = False
        cfg.model.policy.autoregressive_model_params.restore_after_encoder = False

    # set seed
    seed = cfg.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    with open_dict(cfg):
        cfg.output_dir = output_dir

    # configure workspace
    cls = hydra.utils.get_class(cfg.model._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace

    print("Loaded checkpoint from %s" % checkpoint)

    workspace.load_payload_new(payload, exclude_keys=None, include_keys=None, strict=False)
    
    # Re-copy encoder parameters to local causal encoder blocks after loading checkpoint
    if hasattr(workspace.model, 'model') and hasattr(workspace.model.model, 'copy_encoder_parameters'):
        workspace.model.model.copy_encoder_parameters()
        print("Re-copied encoder parameters to local causal encoder blocks")

    # get policy from workspace
    if no_ema:
        with open_dict(cfg):
            cfg.training.use_ema = False
    if cfg.training.use_ema:
        print("Using EMA policy for evaluation.")
        policy = workspace.ema_model
    else:
        policy = workspace.model
        print("Using regular policy for evaluation.")
    
    # CRITICAL: Do NOT move policy to CUDA before forking!
    # CUDA contexts don't work with fork() - if we have CUDA tensors when forking,
    # the child processes will have invalid CUDA contexts, causing nvidia-smi to hang
    # and the entire system to become unresponsive.
    # Strategy: Keep policy on CPU, create all environments (which fork), THEN move to CUDA
    policy.eval()
    
    # Ensure we're not holding any CUDA context
    if "cuda" in device:
        # Release any existing CUDA context
        torch.cuda.empty_cache()
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        print(f"[eval_sim] Policy kept on CPU to avoid CUDA context issues during fork")
    
    if "libero" in cfg.task.name:
        cfg.task.env_runner.n_test = 10 if n_test is None else int(n_test)
        # allow overriding dataset path for evaluation
        if dataset_path is not None and len(dataset_path) > 0:
            with open_dict(cfg):
                if "dataset" in cfg.task and "dataset_path" in cfg.task.dataset:
                    cfg.task.dataset.dataset_path = dataset_path
                if "env_runner" in cfg.task and "dataset_path" in cfg.task.env_runner:
                    cfg.task.env_runner.dataset_path = dataset_path
            print(f"Using dataset_path override: {cfg.task.env_runner.dataset_path}")
    else:
        cfg.task.env_runner.n_test = 50
        
    print(f"[eval_sim] Loading environment runners...")
    env_runner_start = time.time()
    try:
        # CRITICAL: Create environments BEFORE moving policy to CUDA
        # This avoids CUDA context issues when AsyncVectorEnv forks processes
        env_runners = load_env_runner(cfg, output_dir)
        print(f"[eval_sim] Environment runners loaded in {time.time() - env_runner_start:.2f} seconds")
        print(f"[eval_sim] Number of env runners: {len(env_runners) if isinstance(env_runners, list) else 1}")
    except Exception as e:
        print(f"[eval_sim] ERROR loading environment runners: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # NOW move policy to CUDA after all forks are complete
    print(f"[eval_sim] Moving policy to device {device} (after env creation)...")
    if "cuda" in device:
        # Ensure CUDA is ready
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    policy.to(device)
    if "cuda" in device:
        torch.cuda.synchronize()
        print(f"[eval_sim] Policy moved to CUDA and synchronized")

    if "libero" in cfg.task.name:
        step_log = {}
        print(f"[eval_sim] Starting evaluation for {len(env_runners)} task(s)...")
        for idx, env_runner in enumerate(env_runners):
            print(f"[eval_sim] Running task {idx+1}/{len(env_runners)}...")
            task_start = time.time()
            try:
                runner_log = env_runner.run(policy)
                step_log.update(runner_log)
                task_elapsed = time.time() - task_start
                print(f"[eval_sim] Task {idx+1} completed in {task_elapsed:.2f} seconds ({task_elapsed/60:.2f} minutes)")
                print(f"[eval_sim] Task {idx+1} results: {runner_log}")
            except Exception as e:
                print(f"[eval_sim] ERROR in task {idx+1}: {e}")
                import traceback
                traceback.print_exc()
                # Continue with next task instead of failing completely
                continue

        assert "test_mean_score" not in step_log
        all_test_mean_score = {
            k: v for k, v in step_log.items() if "test/" in k and "_mean_score" in k
        }
        if len(all_test_mean_score) == 0:
            print("Warning: No test mean score keys found. Check dataset_path and tasks.\n"
                  f"dataset_path={getattr(cfg.task.env_runner, 'dataset_path', 'N/A')}")
            step_log["test_mean_score"] = float("nan")
        else:
            step_log["test_mean_score"] = np.mean(list(all_test_mean_score.values()))

        runner_log = step_log
    else:
        env_runner = env_runners
        runner_log = env_runner.run(policy)

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
    print(f"[eval_sim] Saving log to {out_path}")
    json.dump(json_log, open(out_path, "w"), indent=2, sort_keys=True)
    total_time = time.time() - start_time
    print(f"[eval_sim] Evaluation completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")


if __name__ == "__main__":
    main()
