"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace

Training with Bayesian Optimization:
python train.py --config-name=train_diffusion_lowdim_workspace \
    bayesian_optimization.enabled=true \
    bayesian_optimization.start_epoch=100 \
    bayesian_optimization.interval=10
"""
# import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True)
import torch
import os
import sys
import hydra
from omegaconf import OmegaConf
import pathlib
from unified_video_action.workspace.base_workspace import BaseWorkspace
from unified_video_action.utils.load_env import load_env_runner
import wandb
import numpy as np
import logging
from omegaconf import open_dict
import dill

logging.getLogger('unified_video_action.codecs.imagecodecs_numcodecs').setLevel(logging.ERROR)


# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


import wandb

if "WANDB_API_KEY" in os.environ:
    wandb.login(key=os.environ["WANDB_API_KEY"])


@hydra.main(
    version_base=None,
    config_path=str(
        pathlib.Path(__file__).parent.joinpath("unified_video_action", "config")
    ),
)
def main(cfg: OmegaConf):
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
        
        # Add Bayesian optimization configuration if not present
        if not hasattr(cfg, 'bayesian_optimization'):
            cfg.bayesian_optimization = OmegaConf.create({
                'enabled': False,
                'start_epoch': cfg.training.num_epochs // 2,  # Start at 50% of training
                'interval': 10,  # Optimize every 10 epochs
                'max_trials': 15,  # Reduced trials for training integration
                'n_test': 5,  # Reduced test count for faster evaluation
                'device': 'cuda:0',
                'output_dir': './bayesian_optimization_logs'
            })

    if cfg.training.debug:
        cfg.dataloader.batch_size = 2
        cfg.val_dataloader.batch_size = 2
        cfg.dataloader.shuffle = False
        cfg.val_dataloader.shuffle = False

        if "env_runner" in cfg.task: 
            cfg.task.env_runner.max_steps = 20

        if "dataloader_cfg" in cfg.task.dataset:
            cfg.task.dataset.dataloader_cfg.batch_size = 2


    cls = hydra.utils.get_class(cfg.model._target_)
    workspace: BaseWorkspace = cls(cfg)

    if cfg.payload:
        print(f"Loading checkpoint from: {cfg.payload}")
        payload = torch.load(open(cfg.payload, "rb"), pickle_module=dill)
        
        # 判断加载类型：预训练模型 vs 恢复训练
        # 预训练模型通常不包含训练状态信息，恢复训练的checkpoint包含epoch, global_step等
        is_pretrained_model = "epoch" not in payload.get("pickles", {}) and "global_step" not in payload.get("pickles", {})
        
        if is_pretrained_model:
            print("[加载类型] 预训练模型 - 将drop整个diffactloss网络")
            diffhead_finetuning = True
        else:
            print("[加载类型] 恢复训练checkpoint - 尝试完整加载")
            diffhead_finetuning = False
            
        if cfg.training.use_ema:
            workspace.load_payload_new(payload, exclude_keys=None, include_keys=None, diffhead_finetuning=diffhead_finetuning, strict=False)
        else:
            workspace.load_payload_new(payload, exclude_keys=['ema_model'], include_keys=None, diffhead_finetuning=diffhead_finetuning, strict=False)

    if cfg.freeze_submodules:
        workspace.freeze_submodules(action_only=True)

    workspace.run()


if __name__ == "__main__":
    print(sys.argv)
    for arg in sys.argv:
        if "local_rank" in arg:  # For deepspeed compatibility
            sys.argv.remove(arg)
    main()
