import sys

sys.path.extend([".", "src"])
import torch
import os
from einops import rearrange
import torch.nn.functional as F
import wandb
import numpy as np

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
    cfg, model, loader, it, output_dir, device, name_label="", plot_actions=False
):
    losses = dict()
    losses["fvd"] = AverageMeter()

    i3d = load_i3d_pretrained(device)
    real_embeddings = []
    pred_embeddings = []

    reals = []
    predictions = []

    n_examples = 4

    with torch.no_grad():
        for n, batch in enumerate(loader):
            if n % 10 == 0:
                print("test_video_fvd", n, len(loader))

            x = batch
            if n >= n_examples:
                break

            x = dict_apply(x, lambda x: x.to(device, non_blocking=True))
            actions = x["action"]

            if cfg.model.policy.use_history_action:
                x = dict_apply(x, lambda x: x[:, 1:])

            x = resize_image(cfg, x)

            B, T, C, H, W = x["obs"]["image"].size()
            k = min(n_examples, B)

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

            if len(predictions) < n_examples:
                reals.append(
                    torch.cat(
                        [
                            x[:, :, : x.size(2) // 2],
                            rearrange(real, "b t h w c -> b c t h w"),
                        ],
                        dim=2,
                    )
                )
                predictions.append(
                    torch.cat(
                        [
                            x[:, :, : x.size(2) // 2],
                            rearrange(pred, "b t h w c -> b c t h w"),
                        ],
                        dim=2,
                    )
                )

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
    real_vid = save_image_grid(
        reals.cpu().numpy(),
        os.path.join(output_dir, f"vis/{name_label}real_{it}.gif"),
        drange=[0, 255],
        grid_size=(reals.size(0) // 4, 4),
    )  # [4, 3, 8, 128, 128]
    pred_vid = save_image_grid(
        predictions.cpu().numpy(),
        os.path.join(output_dir, f"vis/{name_label}predicted_{it}.gif"),
        drange=[0, 255],
        grid_size=(predictions.size(0) // 4, 4),
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
                act_out = unnormalize_future_action(
                    normalizer=model.normalizer,
                    normalizer_type=model.normalizer_type,
                    actions=act_out,
                )
                trajectory = unnormalize_future_action(
                    normalizer=model.normalizer,
                    normalizer_type=model.normalizer_type,
                    actions=trajectory,
                )

                ## calculate l2 distance between the predicted action and ground truth action
                # Use actual action dimension (7 for UMI single arm, 14 for dual arm, etc.)
                action_dim = act_out.shape[-1]
                l2_distance = torch.sqrt(
                    torch.sum((trajectory[:, :, :action_dim] - act_out[:, :, :action_dim]) ** 2, dim=-1)
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
    
    This function:
    1. Predicts actions from observations
    2. Integrates actions to get predicted end-effector trajectory
    3. Compares predicted trajectory with ground truth trajectory
    4. Computes trajectory error and final state distance
    
    Returns:
        dict with:
        - val_eef_trajectory_error: mean L2 error across all timesteps (meters)
        - val_final_state_distance: L2 distance at final timestep (meters)
    """
    trajectory_errors = []
    final_state_distances = []
    
    with torch.no_grad():
        for n, batch in enumerate(loader):
            if n % 10 == 0:
                print("test_eef_trajectory_error", n, len(loader))
            
            x = batch
            x = dict_apply(x, lambda x: x.to(device, non_blocking=True))
            actions = x["action"]
            
            # Save original obs before normalization (needed for robot0_eef_pos)
            original_obs = {}
            if "robot0_eef_pos" in x["obs"]:
                original_obs["robot0_eef_pos"] = x["obs"]["robot0_eef_pos"].clone()
            
            if cfg.model.policy.use_history_action:
                x = dict_apply(x, lambda x: x[:, 1:])
                if "robot0_eef_pos" in original_obs:
                    original_obs["robot0_eef_pos"] = original_obs["robot0_eef_pos"][:, 1:]
            
            x = resize_image(cfg, x)
            
            B, T, C, H, W = x["obs"]["image"].size()
            
            # Check if we have end-effector position in original obs (for UMI datasets)
            if "robot0_eef_pos" not in original_obs:
                # Skip if we don't have eef position data
                if n == 0:
                    print("Warning: robot0_eef_pos not found in obs, skipping trajectory error calculation")
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
                # Unnormalize predicted actions
                act_out = unnormalize_future_action(
                    normalizer=model.normalizer,
                    normalizer_type=model.normalizer_type,
                    actions=act_out,
                )
                
                # Get ground truth end-effector trajectory from original (unnormalized) obs
                # original_obs["robot0_eef_pos"] shape is (B, T_full, 3) where T_full includes initial state
                gt_eef_full_trajectory = original_obs["robot0_eef_pos"]  # (B, T_full, 3)
                initial_eef_pos = gt_eef_full_trajectory[:, 0, :]  # (B, 3)
                gt_eef_trajectory = gt_eef_full_trajectory[:, 1:, :]  # (B, T_full-1, 3)
                
                # Extract position deltas from predicted actions
                # Action format: [pos_delta(3), rot_delta(3), gripper_delta(1)]
                predicted_pos_deltas = act_out[:, :, :3]  # (B, T_action, 3)
                
                # Integrate position deltas to get predicted trajectory
                # predicted_pos[t+1] = predicted_pos[t] + pos_delta[t]
                predicted_trajectory = [initial_eef_pos]  # Start with initial position
                for t in range(predicted_pos_deltas.shape[1]):
                    next_pos = predicted_trajectory[-1] + predicted_pos_deltas[:, t, :]
                    predicted_trajectory.append(next_pos)
                
                # Stack to get (B, T_action+1, 3), then take [:, 1:, :] to get (B, T_action, 3)
                predicted_trajectory = torch.stack(predicted_trajectory[1:], dim=1)  # (B, T_action, 3)
                
                # Align dimensions: take minimum length
                min_T = min(predicted_trajectory.shape[1], gt_eef_trajectory.shape[1])
                predicted_trajectory = predicted_trajectory[:, :min_T, :]
                gt_eef_trajectory = gt_eef_trajectory[:, :min_T, :]
                
                # Compute trajectory error: L2 distance at each timestep
                trajectory_error_per_timestep = torch.sqrt(
                    torch.sum((predicted_trajectory - gt_eef_trajectory) ** 2, dim=-1)
                )  # (B, min_T)
                
                # Mean error across all timesteps (in meters)
                mean_trajectory_error = trajectory_error_per_timestep.mean()
                trajectory_errors.append(mean_trajectory_error.item())
                
                # Final state distance: L2 distance at the last timestep (in meters)
                final_predicted_pos = predicted_trajectory[:, -1, :]  # (B, 3)
                final_gt_pos = gt_eef_trajectory[:, -1, :]  # (B, 3)
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
    
    return log_data
