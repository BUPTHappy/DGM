import os
import random
import pathlib
import sys
import click
import dill
import hydra
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from omegaconf import open_dict

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from unified_video_action.workspace.base_workspace import BaseWorkspace
from unified_video_action.common.pytorch_util import dict_apply
from unified_video_action.utils.data_utils import resize_image
from unified_video_action.eval.eval import prepare_data_predict_action


def _load_policy_and_cfg(
    ckpt_path, output_dir, device, act_diff_testing_steps, dataset_path
):
    payload = torch.load(open(ckpt_path, "rb"), map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]

    seed = cfg.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    with open_dict(cfg):
        cfg.output_dir = output_dir
        cfg.model.policy.autoregressive_model_params.act_diff_testing_steps = str(
            act_diff_testing_steps
        )
        if dataset_path is not None:
            if hasattr(cfg.task, "dataset"):
                if "dataset_path" in cfg.task.dataset:
                    cfg.task.dataset.dataset_path = dataset_path
                if "zarr_path" in cfg.task.dataset:
                    cfg.task.dataset.zarr_path = dataset_path

    cls = hydra.utils.get_class(cfg.model._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.ema_model if workspace.ema_model is not None else workspace.model
    policy.to(device)
    policy.eval()

    effective_steps = policy.model.diffactloss.gen_diffusion.num_timesteps
    print(
        f"[INFO] Effective action diffusion sampling steps: {effective_steps} "
        f"(requested: {act_diff_testing_steps})"
    )

    return cfg, policy


def _build_val_loader(cfg):
    if cfg.task.task_type == "multiple_datasets":
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        val_dataset = dataset.split_unused_episodes()
        return val_dataset.get_dataloader()

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    val_dataset = dataset.get_validation_dataset()
    return DataLoader(val_dataset, **cfg.val_dataloader)


@torch.no_grad()
def _get_action_pre_diffusion_latent(policy, z):
    diffact = policy.model.diffactloss
    act_model_type = diffact.act_model_type

    if act_model_type == "conv_fc":
        z = z.reshape(z.shape[0], diffact.n_frames, diffact.w * diffact.h, z.shape[-1])
        z = z.reshape(z.shape[0] * diffact.n_frames, diffact.w, diffact.h, z.shape[-1])
        z = z.permute(0, 3, 1, 2)
        z = diffact.conv(z)
        z = z.reshape(z.shape[0], -1)
        z = diffact.fc(z)
        z = z.reshape(-1, diffact.n_frames, z.shape[-1])
        z = z.permute(0, 2, 1)
        z = diffact.interpolate(z)
        z = z.permute(0, 2, 1)
        z = diffact.refine(z)
        return z

    if act_model_type == "conv_ori":
        z = z.reshape(z.shape[0], diffact.n_frames, diffact.w * diffact.h, z.shape[-1])
        z = z.reshape(z.shape[0], diffact.n_frames, diffact.w, diffact.h, z.shape[-1])
        z = z.permute(0, 4, 1, 2, 3)
        z = diffact.conv_transpose3d(z)
        z = diffact.avg_pool(z)
        z = z.reshape(z.shape[0], -1, z.shape[1])
        return z

    if act_model_type == "conv2":
        return diffact.conv(z)

    if act_model_type == "fc2":
        z = diffact.fc(z.transpose(1, 2))
        return z.transpose(1, 2)

    if act_model_type == "none":
        return z

    raise NotImplementedError(f"Unsupported act_model_type: {act_model_type}")


@torch.no_grad()
def _collect_z_refine(cfg, policy, loader, device, max_batches):
    if not hasattr(policy.model, "diffactloss"):
        raise RuntimeError("Current checkpoint does not have action diffusion head (diffactloss).")

    all_vals = []
    for n, batch in enumerate(loader):
        if n >= max_batches:
            break

        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
        actions = batch["action"]

        if cfg.model.policy.use_history_action:
            batch = dict_apply(batch, lambda x: x[:, 1:])

        batch = resize_image(cfg, batch)
        bsz, T, _, _, _ = batch["obs"]["image"].size()

        if cfg.task.dataset.language_emb_model is not None:
            if "language" in batch["obs"]:
                language_goal = batch["obs"]["language"]
                del batch["obs"]["language"]
            elif "language_latents" in batch:
                language_goal = batch["language_latents"]
                del batch["language_latents"]
            else:
                raise NotImplementedError("Language model enabled but no language input found.")
        else:
            language_goal = None

        (
            _x,
            _real,
            _latent_size,
            c,
            text_latents,
            history_trajectory,
            trajectory,
            proprioception_input,
        ) = prepare_data_predict_action(cfg, batch, actions, policy, T, device, language_goal=language_goal)

        # Reconstruct the latent that feeds the action diffusion head (before diffusion sampling).
        n_frames = policy.model.n_frames
        seq_len = policy.model.seq_len
        mask = torch.ones(bsz, n_frames, seq_len, device=device)
        tokens = torch.zeros(
            bsz,
            n_frames,
            seq_len,
            policy.model.token_embed_dim,
            device=device,
        )
        x_enc = policy.model.forward_mae_encoder(
            tokens,
            mask,
            c,
            text_latents=text_latents,
            history_nactions=history_trajectory,
            nactions=trajectory,
            proprioception_input=proprioception_input,
            task_mode="policy_model",
        )
        z_decoder = policy.model.forward_mae_decoder(x_enc, mask)
        z_refine = _get_action_pre_diffusion_latent(policy, z_decoder)

        all_vals.append(z_refine.detach().reshape(-1).cpu().numpy())

    if len(all_vals) == 0:
        raise RuntimeError("No z_refine collected. Try increasing --max_batches.")

    return np.concatenate(all_vals, axis=0)


def _plot_hist(values, label, out_path, bins=120):
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, density=True, alpha=0.7, label=label)
    plt.xlabel("z_refine value")
    plt.ylabel("density")
    plt.title("z_refine distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


@click.command()
@click.option(
    "--checkpoint",
    default="checkpoints/pusht_dgm.ckpt",
    type=str,
    show_default=True,
    help="Path to checkpoint.",
)
@click.option("--label", default="uva_2step", type=str, show_default=True)
@click.option(
    "--output-dir",
    default="out_diagram",
    type=str,
    show_default=True,
    help="Directory to save npz and png.",
)
@click.option("--device", default="cuda:0", type=str, show_default=True)
@click.option("--max-batches", default=50, type=int, show_default=True)
@click.option("--act-steps", default=2, type=int, show_default=True)
@click.option(
    "--dataset-path",
    default="data/pusht/pusht_cchi_v7_replay.zarr",
    type=str,
    show_default=True,
    help="Dataset path override for cfg.task.dataset.",
)
def main(checkpoint, label, output_dir, device, max_batches, act_steps, dataset_path):
    output_dir = os.path.abspath(output_dir)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Collecting z_refine from checkpoint: {checkpoint}")
    dataset_path = os.path.abspath(dataset_path)
    print(f"[INFO] Using dataset path: {dataset_path}")
    cfg, policy = _load_policy_and_cfg(
        checkpoint,
        output_dir,
        device,
        act_diff_testing_steps=act_steps,
        dataset_path=dataset_path,
    )
    loader = _build_val_loader(cfg)
    values = _collect_z_refine(cfg, policy, loader, device, max_batches)
    np.savez_compressed(
        os.path.join(output_dir, f"{label}_z_refine.npz"),
        z_refine=values,
        mean=values.mean(),
        std=values.std(),
    )
    print(f"[INFO] {label}: mean={values.mean():.6f}, std={values.std():.6f}")

    fig_path = os.path.join(output_dir, "z_refine_hist.png")
    _plot_hist(values, label, fig_path)
    print(f"[INFO] Saved histogram: {fig_path}")


if __name__ == "__main__":
    main()
