import os
import json
import torch
from torch import nn
import hydra
from omegaconf import DictConfig, OmegaConf
import unified_video_action.model.autoregressive.mar_con_unified as mar
import pathlib
from omegaconf import open_dict
import sys

# === CONFIGURATION ===
#LOG_DIR = "./logs/time_logs/pusht_ucgm_many_steps_sample"
NUM_WARMUP = 20  # iterations to warm up CUDA / JIT
NUM_ITERS = 200  # iterations to average over
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1

def get_pruning_rates():
    pruning_rates = {"restore_after_encoder_True": {}, "restore_after_encoder_False": {}}
    for pruning_rate in [0.25, 0.5, 0.75, 0.9, 0.99]:
        pruning_rates["restore_after_encoder_False"][pruning_rate] = {}
        pruning_rates["restore_after_encoder_True"][pruning_rate] = {}

        # gradual strategy
        enc_pruning_ratios = [pruning_rate/24]*12
        dec_pruning_ratios = [pruning_rate/24]*12
        pruning_ratios = [enc_pruning_ratios, dec_pruning_ratios]
        pruning_rates["restore_after_encoder_False"][pruning_rate]["gradual"] = pruning_ratios

        # only encoder strategy
        enc_pruning_ratios = [pruning_rate/12]*12
        dec_pruning_ratios = [0]*12
        pruning_ratios = [enc_pruning_ratios, dec_pruning_ratios]
        pruning_rates["restore_after_encoder_False"][pruning_rate]["encoder_gradual"] = pruning_ratios

        
        # all at once strategy
        enc_pruning_ratios = [pruning_rate] + [0]*11
        dec_pruning_ratios = [0]*12
        pruning_ratios = [enc_pruning_ratios, dec_pruning_ratios]
        pruning_rates["restore_after_encoder_False"][pruning_rate]["all_at_once"] = pruning_ratios

        # gradual strategy
        enc_pruning_ratios = [pruning_rate/12]*12
        dec_pruning_ratios = [pruning_rate/12]*12
        pruning_ratios = [enc_pruning_ratios, dec_pruning_ratios]
        pruning_rates["restore_after_encoder_True"][pruning_rate]["gradual"] = pruning_ratios
        
        # all at once strategy
        enc_pruning_ratios = [pruning_rate] + [0]*11
        dec_pruning_ratios = [pruning_rate] + [0]*11
        pruning_ratios = [enc_pruning_ratios, dec_pruning_ratios]
        pruning_rates["restore_after_encoder_True"][pruning_rate]["all_at_once"] = pruning_ratios
    return pruning_rates
import os
import torch
from torch import nn
from tqdm import tqdm

def run_profiled_sample_tokens(
    model: nn.Module,
    input_args: tuple,
    log_dir: str,
    num_warmup: int = 20,
    num_iters: int = 200,
    device: str = "cuda"
):
    """
    Run MAR model's sample_tokens with internal profiling of transformer blocks and diffactloss.sample_tokens.
    Collects timing stats over multiple iterations and writes a log file to log_dir.

    Args:
        model (nn.Module):       Your MAR model instance (e.g. MAR(...)).
        input_args (tuple):      Arguments to pass to model.sample_tokens, *excluding* `self`. 
                                 For example: (bsz, cond, text_latents, num_iter, cfg, cfg_schedule, 
                                 temperature, progress, history_nactions, nactions, proprio_input).
        log_dir (str):           Path to a directory where the timing log should be written. 
                                 (The function will create it if it doesn’t exist.)
        num_warmup (int):        Number of warmup iterations (these are run but *not* recorded).
        num_iters (int):         Number of iterations over which to collect timing data.
        device (str):            “cuda” or “cpu” (if you’re on CPU, it will fallback but mostly 
                                 CUDA‐timing is what this is designed for).

    Returns:
        last_output: The raw output from the final `model.sample_tokens(...)` call (e.g. the sampled tokens).
    """

    # 1) Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)
    LOG_PATH = os.path.join(log_dir, "sample_tokens_timing.log")

    # 2) Move model to device & set eval mode
    model = model.to(device)
    model.eval()

    # 3) Prepare a dict to hold per‐module times, and a list of hook handles to remove later
    module_times = {}
    handles = []

    # 4) Strore a small helper to build pre/post hooks
    def make_pre_hook(name):
        def hook(module, inputs):
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
                evt_start = torch.cuda.Event(enable_timing=True)
                evt_end = torch.cuda.Event(enable_timing=True)
                # Save them on the module so post‐hook can read them
                module._evt_start = evt_start
                module._evt_end = evt_end
                evt_start.record()
            else:
                # If you're on CPU, you could use time.time() instead, 
                # but for brevity we just record zero.
                module._start_time_cpu = 0.0
        return hook

    def make_post_hook(name):
        def hook(module, inputs, outputs):
            if device.startswith("cuda"):
                module._evt_end.record()
                torch.cuda.synchronize(device)
                elapsed = module._evt_start.elapsed_time(module._evt_end)
            else:
                elapsed = 0.0
            module_times[name].append(elapsed)
        return hook

    # 5) Hook every transformer block under encoder_blocks, if it exists
    if hasattr(model, "encoder_blocks"):
        for idx, block in enumerate(model.encoder_blocks):
            name = f"encoder_blocks.{idx}"
            module_times[name] = []
            handles.append(block.register_forward_pre_hook(make_pre_hook(name)))
            handles.append(block.register_forward_hook(make_post_hook(name)))

    # 6) Hook every transformer block under decoder_blocks, if it exists
    if hasattr(model, "decoder_blocks"):
        for idx, block in enumerate(model.decoder_blocks):
            name = f"decoder_blocks.{idx}"
            module_times[name] = []
            handles.append(block.register_forward_pre_hook(make_pre_hook(name)))
            handles.append(block.register_forward_hook(make_post_hook(name)))

    # 7) If your MAR class has “action_proj_cond”, “decoder_score_predictor”, “encoder_score_predictor”, hook those heads too:
    for head_name in ("action_proj_cond", "decoder_score_predictor", "encoder_score_predictor"):
        mod_head = getattr(model, head_name, None)
        if mod_head is not None:
            module_times[head_name] = []
            handles.append(mod_head.register_forward_pre_hook(make_pre_hook(head_name)))
            handles.append(mod_head.register_forward_hook(make_post_hook(head_name)))

    # 8) Hook diffactloss.sample_tokens, if diffactloss exists.
    #    We do this by monkey‐patching model.diffactloss.sample_tokens itself so we can measure it.
    if hasattr(model, "diffactloss"):
        name = "diffactloss.sample"
        module_times[name] = []
        original_diffact = model.diffactloss.sample

        def diffact_profiled(*args, **kwargs):
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
                evt_s = torch.cuda.Event(enable_timing=True)
                evt_e = torch.cuda.Event(enable_timing=True)
                evt_s.record()
                out = original_diffact(*args, **kwargs)
                evt_e.record()
                torch.cuda.synchronize(device)
                elapsed = evt_s.elapsed_time(evt_e)
            else:
                elapsed = 0.0
                out = original_diffact(*args, **kwargs)

            module_times[name].append(elapsed)
            return out

        model.diffactloss.sample = diffact_profiled

    # 9) Warm‐up loop: run sample_tokens num_warmup times (to load CUDA kernels, JIT, etc.), but do NOT record them.
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model.sample_tokens(*input_args)

    # 10) Now run the profiled loop num_iters times. We measure:
    #     - the “full sample_tokens” time per iteration, and
    #     - accumulate per‐module times via the hooks we installed.
    full_forward_times = []
    last_output = None

    for _ in tqdm(range(num_iters), desc="Profiling sample_tokens"):
        # (a) Measure full sample_tokens call
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
            evt_start = torch.cuda.Event(enable_timing=True)
            evt_end = torch.cuda.Event(enable_timing=True)
            evt_start.record()

            with torch.no_grad():
                last_output = model.sample_tokens(*input_args)

            evt_end.record()
            torch.cuda.synchronize(device)
            full_elapsed = evt_start.elapsed_time(evt_end)
        else:
            full_elapsed = 0.0
            with torch.no_grad():
                last_output = model.sample_tokens(*input_args)

        full_forward_times.append(full_elapsed)

    # 11) Remove all hooks now that we’re done
    for handle in handles:
        handle.remove()

    # 12) Compute averages
    avg_full = sum(full_forward_times) / len(full_forward_times)
    avg_module_times = {
        name: (sum(times) / len(times) if len(times) > 0 else 0.0)
        for name, times in module_times.items()
    }

    # 13) Write a simple text log in log_dir/sample_tokens_timing.log
    with open(LOG_PATH, "w") as f:
        f.write(f"Profiling sample_tokens over {num_iters} iterations (after {num_warmup} warmups)\n")
        f.write(f"Average full sample_tokens time: {avg_full:.3f} ms\n\n")
        f.write("---- Average time per submodule (ms) ----\n")
        for name, t in sorted(avg_module_times.items(), key=lambda x: -x[1]):
            f.write(f"{name:<40}: {t:8.3f} ms\n")

    print(f"[✓] Timing summary written to {LOG_PATH}")
    return last_output

@hydra.main(
    version_base=None,
    config_path="../unified_video_action/config",  # <== Relative to THIS script
    config_name="config"
)
def main(cfg: DictConfig):
    LOG_PATH = cfg.save_folder

    # cfg stuff copied from train.py
    OmegaConf.resolve(cfg)
    if cfg.model.policy.action_model_params.predict_action == False:
        cfg.checkpoint.topk.monitor_key = "video_fvd"
        cfg.checkpoint.topk.format_str = (
            "epoch={epoch:04d}-video_fvd={video_fvd:.3f}.ckpt"
        )
        cfg.checkpoint.topk.mode = "min"

    with open_dict(cfg):
        cfg.n_gpus = torch.cuda.device_count()
        cfg.model.policy.debug = cfg.training.debug

    # instatiate the model
    autoregressive_model_params = cfg.model.policy.autoregressive_model_params
    action_model_params = cfg.model.policy.action_model_params
    language_emb_model = cfg.task.dataset.language_emb_model

    # instantiate mar_base

    # Prepare inputs
    input_tuple = (
        BATCH_SIZE,  # bsz
        torch.randn(BATCH_SIZE, 4, 16, 16, 16, dtype=torch.float32, device=DEVICE),  # cond
        None,  # text_latents
        1,  # num_iter
        1,  # cfg
        "linear",  # cfg_schedule
        0.95,  # temperature
        None,  # history_nactions
        torch.randn(BATCH_SIZE, 16, 2, dtype=torch.float32, device=DEVICE),  # nactions
        None,  # proprioception_input
        "policy_model",  # task_mode
    )
    model = (
        mar.__dict__[autoregressive_model_params.model_size](
            img_size=autoregressive_model_params.img_size,
            vae_stride=autoregressive_model_params.vae_stride,
            patch_size=autoregressive_model_params.patch_size,
            vae_embed_dim=autoregressive_model_params.vae_embed_dim,
            mask_ratio_min=autoregressive_model_params.mask_ratio_min,
            label_drop_prob=autoregressive_model_params.label_drop_prob,
            attn_dropout=autoregressive_model_params.attn_dropout,
            proj_dropout=autoregressive_model_params.proj_dropout,
            diffloss_d=autoregressive_model_params.diffloss_d,
            diffloss_w=autoregressive_model_params.diffloss_w,
            diffloss_act_d=autoregressive_model_params.diffloss_act_d,
            diffloss_act_w=autoregressive_model_params.diffloss_act_w,
            num_sampling_steps=autoregressive_model_params.num_sampling_steps,
            diffusion_batch_mul=autoregressive_model_params.diffusion_batch_mul,
            grad_checkpointing=autoregressive_model_params.grad_checkpointing,
            predict_video=autoregressive_model_params.predict_video,
            act_diff_training_steps=autoregressive_model_params.act_diff_training_steps,
            act_diff_testing_steps=autoregressive_model_params.act_diff_testing_steps,
            action_model_params=action_model_params,
            use_history_action=cfg.model.policy.use_history_action,
            action_mask_ratio=cfg.model.policy.action_mask_ratio,
            use_proprioception=cfg.model.policy.use_proprioception,
            predict_wrist_img=cfg.model.policy.predict_wrist_img,
            different_history_freq=cfg.model.policy.different_history_freq,
            predict_proprioception=cfg.model.policy.predict_proprioception,
            task_name=cfg.task.name,
            language_emb_model=language_emb_model,
            shape_meta=cfg.task.shape_meta,
            token_pruning=autoregressive_model_params.token_pruning,
            use_ucgm=autoregressive_model_params.use_ucgm,
        )
        .to(DEVICE)
        .eval()
    )

    run_profiled_sample_tokens(model=model, 
                              input_args=input_tuple, 
                              log_dir=LOG_PATH, 
                              num_warmup=NUM_WARMUP, 
                              num_iters=NUM_ITERS, 
                              device=DEVICE)
    


if __name__ == "__main__":
    main()
