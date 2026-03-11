import sys

sys.path.extend([".", "src"])
import torch
import os
from einops import rearrange
import torch.nn.functional as F
import wandb
import numpy as np
import math

from unified_video_action.fvd.fvd import get_fvd_logits, frechet_distance
from unified_video_action.fvd.download import load_i3d_pretrained
from unified_video_action.common.pytorch_util import dict_apply
from unified_video_action.utils.utils import AverageMeter
from unified_video_action.utils.data_utils import resize_image
from unified_video_action.utils.data_utils import (
    normalize_action,
    normalize_obs,
    unnormalize_future_action,
)
from unified_video_action.utils.data_utils import (
    process_data,
    save_image_grid,
    get_vae_latent,
    get_trajectory,
    decode_from_sample_autoregressive,
)
from unified_video_action.utils.language_model import extract_text_features
import json




def prepare_data_predict_action(
    cfg, x, actions, model, T, device, language_goal=None, eval=False
):
    ## normalize actions and observations
    nactions = normalize_action(
        normalizer=model.normalizer,
        normalizer_type=model.normalizer_type,
        actions=actions,
    )
    x = normalize_obs(
        normalizer=model.normalizer, normalizer_type=model.normalizer_type, batch=x
    )

    ## process data
    x, proprioception_input, _ = process_data(
        x,
        task_name=cfg.task.name,
        eval=eval,
        use_proprioception=cfg.model.policy.use_proprioception,
        different_history_freq=cfg.model.policy.different_history_freq,
    )

    real, _, c, latent_size, proprioception_input = get_vae_latent(
        x, model.vae_model, eval=True, proprioception_input=proprioception_input
    )
    history_trajectory, trajectory = get_trajectory(
        nactions,
        T,
        cfg.model.policy.shift_action,
        use_history_action=cfg.model.policy.use_history_action,
    )

    text_latents = None
    if cfg.task.dataset.language_emb_model is not None:
        if "umi" in cfg.task.name:
            text_latents = language_goal
        elif "libero" in cfg.task.name:
            if cfg.task.dataset.language_emb_model == "clip":
                text_tokens = {
                    "input_ids": language_goal[:, 0].long()[:, 0],
                    "attention_mask": language_goal[:, 0].long()[:, 1],
                }
                text_latents = extract_text_features(
                    model.text_model,
                    text_tokens,
                    language_emb_model=cfg.task.dataset.language_emb_model,
                )
            elif cfg.task.dataset.language_emb_model == "flant5":
                text_tokens = language_goal[:, 0].long()
                text_latents = extract_text_features(
                    model.text_model,
                    text_tokens,
                    language_emb_model=cfg.task.dataset.language_emb_model,
                ).float()
            else:
                raise NotImplementedError
    return (
        x,
        real,
        latent_size,
        c,
        text_latents,
        history_trajectory,
        trajectory,
        proprioception_input,
    )


def test_video_fvd(
    cfg,
    model,
    loader,
    it,
    output_dir,
    device,
    name_label="",
    plot_actions=False,
    max_batches=4,
    save_full_sequences=False,
    max_full_sequences=20,
):
    losses = dict()
    losses["fvd"] = AverageMeter()

    i3d = load_i3d_pretrained(device)
    real_embeddings = []
    pred_embeddings = []

    reals = []
    predictions = []

    # Keep preview compact by default: 4 samples per batch for up to max_batches.
    preview_per_batch = 4
    saved_full_sequences = 0

    with torch.no_grad():
        for n, batch in enumerate(loader):
            if n % 10 == 0:
                print("test_video_fvd", n, len(loader))

            x = batch
            if n >= max_batches:
                break

            x = dict_apply(x, lambda x: x.to(device, non_blocking=True))
            actions = x["action"]

            if cfg.model.policy.use_history_action:
                x = dict_apply(x, lambda x: x[:, 1:])

            x = resize_image(cfg, x)

            B, T, C, H, W = x["obs"]["image"].size()
            k = min(preview_per_batch, B)

            actions = actions[:k]
            x = dict_apply(x, lambda x: x[:k])

            if cfg.task.dataset.language_emb_model is not None:
                if "language" in x["obs"]:
                    language_goal = x["obs"]["language"]
                    del x["obs"]["language"]
                elif "language_latents" in x:
                    language_goal = x["language_latents"]
                    del x["language_latents"]
                else:
                    raise NotImplementedError
            else:
                language_goal = None

            (
                x,
                real,
                _,
                c,
                text_latents,
                history_trajectory,
                trajectory,
                proprioception_input,
            ) = prepare_data_predict_action(
                cfg, x, actions, model, T, device, language_goal=language_goal
            )
            z, act_out = model.model.sample_tokens(
                bsz=k,
                cond=c,
                text_latents=text_latents,
                num_iter=cfg.model.policy.autoregressive_model_params.num_iter,
                cfg=cfg.model.policy.autoregressive_model_params.cfg,
                cfg_schedule=cfg.model.policy.autoregressive_model_params.cfg_schedule,
                temperature=cfg.model.policy.autoregressive_model_params.temperature,
                history_nactions=history_trajectory,
                nactions=trajectory,
                proprioception_input=proprioception_input,
                task_mode="full_dynamic_model",
            )
            pred = decode_from_sample_autoregressive(model.vae_model, z / 0.2325)
            pred = pred.clamp(-1, 1).cpu()

            pred = 1 + rearrange(pred, "(b t) c h w -> b t h w c", b=k)
            real = (1 + rearrange(real, "b c t h w -> b t h w c")).cpu()

            pred = pred * 127.5
            pred = pred.type(torch.uint8)

            real = real * 127.5
            real = real.type(torch.uint8)

            x = (1 + x) * 127.5  # b c t h w
            x = x.type(torch.uint8).cpu()

            real_full = torch.cat(
                [
                    x[:, :, : x.size(2) // 2],
                    rearrange(real, "b t h w c -> b c t h w"),
                ],
                dim=2,
            )
            pred_full = torch.cat(
                [
                    x[:, :, : x.size(2) // 2],
                    rearrange(pred, "b t h w c -> b c t h w"),
                ],
                dim=2,
            )

            reals.append(real_full)
            predictions.append(pred_full)

            # Optional: export full sequence videos per sample for easier inspection.
            if save_full_sequences and saved_full_sequences < max_full_sequences:
                full_dir = os.path.join(output_dir, "vis", "full_sequences")
                os.makedirs(full_dir, exist_ok=True)
                for i in range(k):
                    if saved_full_sequences >= max_full_sequences:
                        break
                    seq_idx = saved_full_sequences
                    save_image_grid(
                        real_full[i : i + 1].cpu().numpy(),
                        os.path.join(full_dir, f"{name_label}real_seq_{seq_idx:04d}.gif"),
                        drange=[0, 255],
                        grid_size=(1, 1),
                    )
                    save_image_grid(
                        pred_full[i : i + 1].cpu().numpy(),
                        os.path.join(full_dir, f"{name_label}pred_seq_{seq_idx:04d}.gif"),
                        drange=[0, 255],
                        grid_size=(1, 1),
                    )
                    saved_full_sequences += 1

            if real.shape[1] < 16:
                pred = pred.repeat_interleave(repeats=4, dim=1)
                real = real.repeat_interleave(repeats=4, dim=1)

            pred_embeddings.append(get_fvd_logits(pred.numpy(), i3d=i3d, device=device))
            real_embeddings.append(get_fvd_logits(real.numpy(), i3d=i3d, device=device))

    log_data = dict()
    reals = torch.cat(reals)
    predictions = torch.cat(predictions)

    real_embeddings = torch.cat(real_embeddings)
    pred_embeddings = torch.cat(pred_embeddings)
    fvd = frechet_distance(
        pred_embeddings.clone().detach(), real_embeddings.clone().detach()
    )
    fvd = fvd.item()

    os.makedirs(output_dir + "/vis", exist_ok=True)
    n_vis = reals.size(0)
    cols = min(4, n_vis)
    rows = int(math.ceil(n_vis / cols))
    total_slots = rows * cols
    if total_slots > n_vis:
        pad_count = total_slots - n_vis
        reals = torch.cat([reals, reals[-1:].repeat(pad_count, 1, 1, 1, 1)], dim=0)
        predictions = torch.cat(
            [predictions, predictions[-1:].repeat(pad_count, 1, 1, 1, 1)], dim=0
        )

    real_vid = save_image_grid(
        reals.cpu().numpy(),
        os.path.join(output_dir, f"vis/{name_label}real_{it}.gif"),
        drange=[0, 255],
        grid_size=(cols, rows),
    )  # [4, 3, 8, 128, 128]
    pred_vid = save_image_grid(
        predictions.cpu().numpy(),
        os.path.join(output_dir, f"vis/{name_label}predicted_{it}.gif"),
        drange=[0, 255],
        grid_size=(cols, rows),
    )  # [4, 3, 8, 128, 128]

    real_video = wandb.Video(os.path.join(output_dir, f"vis/{name_label}real_{it}.gif"))
    pred_video = wandb.Video(
        os.path.join(output_dir, f"vis/{name_label}predicted_{it}.mp4")
    )

    log_data[f"{name_label}video_fvd"] = fvd
    log_data[f"{name_label}real_img"] = real_video
    log_data[f"{name_label}predicted_img"] = pred_video

    return log_data


def test_action_l2(
    cfg,
    model,
    loader,
    it,
    output_dir,
    device,
    text_model=None,
    name_label="",
    plot_actions=False,
):
    action_l2_distances = []

    with torch.no_grad():
        for n, batch in enumerate(loader):
            if n % 10 == 0:
                print("test_action_l2", n, len(loader))

            x = batch
            x = dict_apply(x, lambda x: x.to(device, non_blocking=True))
            actions = x["action"]

            if cfg.model.policy.use_history_action:
                x = dict_apply(x, lambda x: x[:, 1:])

            x = resize_image(cfg, x)

            B, T, C, H, W = x["obs"]["image"].size()

            if cfg.task.dataset.language_emb_model is not None:
                if "language" in x["obs"]:
                    language_goal = x["obs"]["language"]
                    del x["obs"]["language"]
                elif "language_latents" in x:
                    language_goal = x["language_latents"]
                    del x["language_latents"]
                else:
                    raise NotImplementedError
            else:
                language_goal = None

            (
                x,
                real,
                _,
                c,
                text_latents,
                history_trajectory,
                trajectory,
                proprioception_input,
            ) = prepare_data_predict_action(
                cfg, x, actions, model, T, device, language_goal=language_goal
            )

            z, act_out = model.model.sample_tokens(
                bsz=B,
                cond=c,
                text_latents=text_latents,
                num_iter=cfg.model.policy.autoregressive_model_params.num_iter,
                cfg=cfg.model.policy.autoregressive_model_params.cfg,
                cfg_schedule=cfg.model.policy.autoregressive_model_params.cfg_schedule,
                temperature=cfg.model.policy.autoregressive_model_params.temperature,
                history_nactions=history_trajectory,
                nactions=trajectory,
                proprioception_input=proprioception_input,
                task_mode="policy_model",
            )

            if cfg.model.policy.action_model_params.predict_action:
                # Some legacy multitask checkpoints can output a larger action head than
                # the single-task dataset action dimension. Align dims for evaluation.
                gt_dim = trajectory.shape[-1]
                pred_actions = act_out
                if pred_actions.shape[-1] != gt_dim:
                    pred_actions = pred_actions[:, :, :gt_dim]
                    if n == 0:
                        print(
                            f"Warning: action dim mismatch (pred={act_out.shape[-1]}, gt={gt_dim}); "
                            f"using first {gt_dim} dims for metrics."
                        )
                try:
                    pred_actions = unnormalize_future_action(
                        normalizer=model.normalizer,
                        normalizer_type=model.normalizer_type,
                        actions=pred_actions,
                    )
                    trajectory = unnormalize_future_action(
                        normalizer=model.normalizer,
                        normalizer_type=model.normalizer_type,
                        actions=trajectory,
                    )
                except Exception as e:
                    if n == 0:
                        print(
                            f"Warning: unnormalize failed ({e}); "
                            "falling back to normalized-space action L2."
                        )

                ## calculate l2 distance between the predicted action and ground truth action
                # Use actual action dimension (7 for UMI single arm, 14 for dual arm, etc.)
                action_dim = min(pred_actions.shape[-1], trajectory.shape[-1])
                l2_distance = torch.sqrt(
                    torch.sum(
                        (trajectory[:, :, :action_dim] - pred_actions[:, :, :action_dim]) ** 2,
                        dim=-1,
                    )
                )
                action_l2_distances.append(l2_distance.mean())

            if cfg.training.debug:
                break

    log_data = dict()
    if cfg.model.policy.action_model_params.predict_action:
        log_data[f"{name_label}val_action_l2_distances"] = (
            torch.stack(action_l2_distances).mean().item()
        )

    return log_data


def test_eef_trajectory_error(
    cfg,
    model,
    loader,
    it,
    output_dir,
    device,
    text_model=None,
    name_label="",
):
    """
    Compute end-effector trajectory error and final state distance.
    
    Supports both single-arm and bimanual (dual-arm) tasks:
    - Single-arm: action dim 7, uses robot0_eef_pos
    - Bimanual: action dim 14, uses robot0_eef_pos and robot1_eef_pos
    
    This function:
    1. Predicts actions from observations
    2. Integrates actions to get predicted end-effector trajectory
    3. Compares predicted trajectory with ground truth trajectory
    4. Computes trajectory error and final state distance
    
    Returns:
        dict with:
        - val_eef_trajectory_error: mean L2 error across all timesteps (meters)
        - val_final_state_distance: L2 distance at final timestep (meters)
        For bimanual tasks, also includes:
        - val_eef_trajectory_error_robot0/robot1: per-arm trajectory errors
        - val_final_state_distance_robot0/robot1: per-arm final state distances
    """
    trajectory_errors = []
    final_state_distances = []
    # For bimanual tasks
    trajectory_errors_robot0 = []
    trajectory_errors_robot1 = []
    final_state_distances_robot0 = []
    final_state_distances_robot1 = []
    
    is_bimanual = None  # Will be determined from data
    
    with torch.no_grad():
        for n, batch in enumerate(loader):
            if n % 10 == 0:
                print("test_eef_trajectory_error", n, len(loader))
            
            x = batch
            x = dict_apply(x, lambda x: x.to(device, non_blocking=True))
            actions = x["action"]
            
            # Detect if this is a bimanual task based on action dimension
            action_dim = actions.shape[-1]
            if is_bimanual is None:
                is_bimanual = action_dim == 14
                if is_bimanual:
                    print("  Detected bimanual task (action_dim=14)")
                else:
                    print(f"  Detected single-arm task (action_dim={action_dim})")
            
            # Save original obs before normalization (needed for robot_eef_pos)
            original_obs = {}
            if "robot0_eef_pos" in x["obs"]:
                original_obs["robot0_eef_pos"] = x["obs"]["robot0_eef_pos"].clone()
            if is_bimanual and "robot1_eef_pos" in x["obs"]:
                original_obs["robot1_eef_pos"] = x["obs"]["robot1_eef_pos"].clone()
            
            if cfg.model.policy.use_history_action:
                x = dict_apply(x, lambda x: x[:, 1:])
                if "robot0_eef_pos" in original_obs:
                    original_obs["robot0_eef_pos"] = original_obs["robot0_eef_pos"][:, 1:]
                if "robot1_eef_pos" in original_obs:
                    original_obs["robot1_eef_pos"] = original_obs["robot1_eef_pos"][:, 1:]
            
            x = resize_image(cfg, x)
            
            B, T, C, H, W = x["obs"]["image"].size()
            
            # Check if we have end-effector position in original obs (for UMI datasets)
            if "robot0_eef_pos" not in original_obs:
                # Skip if we don't have eef position data
                if n == 0:
                    print("Warning: robot0_eef_pos not found in obs, skipping trajectory error calculation")
                continue
            
            if is_bimanual and "robot1_eef_pos" not in original_obs:
                if n == 0:
                    print("Warning: robot1_eef_pos not found for bimanual task, skipping trajectory error calculation")
                continue
            
            if cfg.task.dataset.language_emb_model is not None:
                if "language" in x["obs"]:
                    language_goal = x["obs"]["language"]
                    del x["obs"]["language"]
                elif "language_latents" in x:
                    language_goal = x["language_latents"]
                    del x["language_latents"]
                else:
                    raise NotImplementedError
            else:
                language_goal = None
            
            (
                x_processed,
                real,
                _,
                c,
                text_latents,
                history_trajectory,
                trajectory,
                proprioception_input,
            ) = prepare_data_predict_action(
                cfg, x, actions, model, T, device, language_goal=language_goal
            )
            
            z, act_out = model.model.sample_tokens(
                bsz=B,
                cond=c,
                text_latents=text_latents,
                num_iter=cfg.model.policy.autoregressive_model_params.num_iter,
                cfg=cfg.model.policy.autoregressive_model_params.cfg,
                cfg_schedule=cfg.model.policy.autoregressive_model_params.cfg_schedule,
                temperature=cfg.model.policy.autoregressive_model_params.temperature,
                history_nactions=history_trajectory,
                nactions=trajectory,
                proprioception_input=proprioception_input,
                task_mode="policy_model",
            )
            
            if cfg.model.policy.action_model_params.predict_action:
                gt_dim = trajectory.shape[-1]
                pred_actions = act_out
                if pred_actions.shape[-1] != gt_dim:
                    pred_actions = pred_actions[:, :, :gt_dim]
                    if n == 0:
                        print(
                            f"Warning: action dim mismatch (pred={act_out.shape[-1]}, gt={gt_dim}); "
                            f"using first {gt_dim} dims for trajectory metrics."
                        )
                try:
                    # Unnormalize predicted actions
                    pred_actions = unnormalize_future_action(
                        normalizer=model.normalizer,
                        normalizer_type=model.normalizer_type,
                        actions=pred_actions,
                    )
                    
                    # Also unnormalize ground truth actions for comparison
                    gt_actions = unnormalize_future_action(
                        normalizer=model.normalizer,
                        normalizer_type=model.normalizer_type,
                        actions=trajectory,
                    )
                except Exception as e:
                    if n == 0:
                        print(
                            f"Warning: unnormalize failed ({e}); "
                            "falling back to normalized-space trajectory metrics."
                        )
                    gt_actions = trajectory
                
                if is_bimanual:
                    # Bimanual task: process both robot0 and robot1
                    # Action format: [pos0(3), rot0(3), gripper0(1), pos1(3), rot1(3), gripper1(1)] = 14D
                    
                    def compute_trajectory_error_for_arm(arm_idx, pos_start_idx, eef_key):
                        """Helper to compute trajectory error for one arm"""
                        gt_eef_full_trajectory = original_obs[eef_key]  # (B, T_full, 3)
                        initial_eef_pos = gt_eef_full_trajectory[:, 0, :]  # (B, 3)
                        gt_eef_trajectory = gt_eef_full_trajectory[:, 1:, :]  # (B, T_full-1, 3)
                        
                        # Extract position deltas for this arm
                        predicted_pos_deltas = pred_actions[:, :, pos_start_idx:pos_start_idx+3]  # (B, T_action, 3)
                        
                        # Debug info for first batch
                        if n == 0 and len(trajectory_errors) == 0:
                            pred_action_mag = torch.norm(predicted_pos_deltas[0], dim=-1).mean().item()
                            print(f"  Debug robot{arm_idx} - Predicted action magnitude: {pred_action_mag:.6f} m/step")
                        
                        # Integrate position deltas
                        predicted_trajectory = [initial_eef_pos]
                        for t in range(predicted_pos_deltas.shape[1]):
                            next_pos = predicted_trajectory[-1] + predicted_pos_deltas[:, t, :]
                            predicted_trajectory.append(next_pos)
                        predicted_trajectory = torch.stack(predicted_trajectory[1:], dim=1)  # (B, T_action, 3)
                        
                        # Align dimensions
                        min_T = min(predicted_trajectory.shape[1], gt_eef_trajectory.shape[1])
                        predicted_trajectory = predicted_trajectory[:, :min_T, :]
                        gt_eef_trajectory = gt_eef_trajectory[:, :min_T, :]
                        
                        # Compute trajectory error
                        trajectory_error_per_timestep = torch.sqrt(
                            torch.sum((predicted_trajectory - gt_eef_trajectory) ** 2, dim=-1)
                        )
                        mean_trajectory_error = trajectory_error_per_timestep.mean().item()
                        
                        # Final state distance
                        final_state_distance = torch.sqrt(
                            torch.sum((predicted_trajectory[:, -1, :] - gt_eef_trajectory[:, -1, :]) ** 2, dim=-1)
                        ).mean().item()
                        
                        return mean_trajectory_error, final_state_distance
                    
                    # Robot 0: position deltas at indices 0:3
                    traj_err_0, final_dist_0 = compute_trajectory_error_for_arm(0, 0, "robot0_eef_pos")
                    trajectory_errors_robot0.append(traj_err_0)
                    final_state_distances_robot0.append(final_dist_0)
                    
                    # Robot 1: position deltas at indices 7:10 (after robot0's 7D action)
                    traj_err_1, final_dist_1 = compute_trajectory_error_for_arm(1, 7, "robot1_eef_pos")
                    trajectory_errors_robot1.append(traj_err_1)
                    final_state_distances_robot1.append(final_dist_1)
                    
                    # Combined error (average of both arms)
                    trajectory_errors.append((traj_err_0 + traj_err_1) / 2)
                    final_state_distances.append((final_dist_0 + final_dist_1) / 2)
                    
                else:
                    # Single-arm task (original logic)
                    gt_eef_full_trajectory = original_obs["robot0_eef_pos"]  # (B, T_full, 3)
                    initial_eef_pos = gt_eef_full_trajectory[:, 0, :]  # (B, 3)
                    gt_eef_trajectory = gt_eef_full_trajectory[:, 1:, :]  # (B, T_full-1, 3)
                    
                    # Extract position deltas from predicted actions
                    # Action format: [pos_delta(3), rot_delta(3), gripper_delta(1)]
                    predicted_pos_deltas = pred_actions[:, :, :3]  # (B, T_action, 3)
                    gt_pos_deltas = gt_actions[:, :, :3]  # (B, T_action, 3)
                    
                    # Debug: Check action magnitudes (only for first batch, first sample)
                    if n == 0 and len(trajectory_errors) == 0:
                        pred_action_mag = torch.norm(predicted_pos_deltas[0], dim=-1).mean().item()
                        gt_action_mag = torch.norm(gt_pos_deltas[0], dim=-1).mean().item()
                        print(f"  Debug - Predicted action magnitude: {pred_action_mag:.6f} m/step")
                        print(f"  Debug - Ground truth action magnitude: {gt_action_mag:.6f} m/step")
                        print(f"  Debug - Initial eef pos: {initial_eef_pos[0].cpu().numpy()}")
                        print(f"  Debug - Final gt eef pos: {gt_eef_trajectory[0, -1].cpu().numpy()}")
                        print(f"  Debug - Trajectory length: {gt_eef_trajectory.shape[1]} steps")
                        total_gt_displacement = torch.norm(gt_eef_trajectory[0, -1] - initial_eef_pos[0]).item()
                        print(f"  Debug - Total GT displacement: {total_gt_displacement:.4f} m")
                    
                    # Integrate position deltas to get predicted trajectory
                    predicted_trajectory = [initial_eef_pos]
                    for t in range(predicted_pos_deltas.shape[1]):
                        next_pos = predicted_trajectory[-1] + predicted_pos_deltas[:, t, :]
                        predicted_trajectory.append(next_pos)
                    predicted_trajectory = torch.stack(predicted_trajectory[1:], dim=1)  # (B, T_action, 3)
                    
                    # Align dimensions
                    min_T = min(predicted_trajectory.shape[1], gt_eef_trajectory.shape[1])
                    predicted_trajectory = predicted_trajectory[:, :min_T, :]
                    gt_eef_trajectory = gt_eef_trajectory[:, :min_T, :]
                    
                    # Debug: Check predicted final position
                    if n == 0 and len(trajectory_errors) == 0:
                        pred_final_pos = predicted_trajectory[0, -1].cpu().numpy()
                        print(f"  Debug - Predicted final eef pos: {pred_final_pos}")
                        pred_total_displacement = torch.norm(predicted_trajectory[0, -1] - initial_eef_pos[0]).item()
                        print(f"  Debug - Total predicted displacement: {pred_total_displacement:.4f} m")
                    
                    # Compute trajectory error
                    trajectory_error_per_timestep = torch.sqrt(
                        torch.sum((predicted_trajectory - gt_eef_trajectory) ** 2, dim=-1)
                    )
                    mean_trajectory_error = trajectory_error_per_timestep.mean()
                    trajectory_errors.append(mean_trajectory_error.item())
                    
                    # Final state distance
                    final_predicted_pos = predicted_trajectory[:, -1, :]
                    final_gt_pos = gt_eef_trajectory[:, -1, :]
                    final_state_distance = torch.sqrt(
                        torch.sum((final_predicted_pos - final_gt_pos) ** 2, dim=-1)
                    ).mean()
                    final_state_distances.append(final_state_distance.item())
            
            if cfg.training.debug:
                break
    
    log_data = dict()
    if len(trajectory_errors) > 0:
        log_data[f"{name_label}val_eef_trajectory_error"] = np.mean(trajectory_errors)
        log_data[f"{name_label}val_final_state_distance"] = np.mean(final_state_distances)
        print(f"  Trajectory error: {np.mean(trajectory_errors):.4f} m")
        print(f"  Final state distance: {np.mean(final_state_distances):.4f} m")
        
        # Log per-arm metrics for bimanual tasks
        if is_bimanual and len(trajectory_errors_robot0) > 0:
            log_data[f"{name_label}val_eef_trajectory_error_robot0"] = np.mean(trajectory_errors_robot0)
            log_data[f"{name_label}val_eef_trajectory_error_robot1"] = np.mean(trajectory_errors_robot1)
            log_data[f"{name_label}val_final_state_distance_robot0"] = np.mean(final_state_distances_robot0)
            log_data[f"{name_label}val_final_state_distance_robot1"] = np.mean(final_state_distances_robot1)
            print(f"  Robot0 trajectory error: {np.mean(trajectory_errors_robot0):.4f} m")
            print(f"  Robot1 trajectory error: {np.mean(trajectory_errors_robot1):.4f} m")
    
    return log_data
