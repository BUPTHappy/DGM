if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import math
import torch
from omegaconf import OmegaConf, open_dict
import pathlib
import copy
import random
import tqdm
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader
import numpy as np
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, DistributedDataParallelKwargs
import pickle

from unified_video_action.workspace.base_workspace import BaseWorkspace
from unified_video_action.policy.unified_video_action_policy import (
    UnifiedVideoActionPolicy,
)
from unified_video_action.dataset.base_dataset import BaseImageDataset
from unified_video_action.dataset.umi_multi_dataset import UmiMultiDataset
from unified_video_action.common.checkpoint_util import TopKCheckpointManager
from unified_video_action.common.pytorch_util import dict_apply
from unified_video_action.model.autoregressive.ema_model import EMAModel
from unified_video_action.model.common.lr_scheduler import get_scheduler
from unified_video_action.utils.load_env import load_env_runner, env_rollout
from unified_video_action.eval.eval import test_video_fvd, test_action_l2
from unified_video_action.utils.data_utils import resize_image

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainUnifiedVideoActionWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure policy model
        language_emb_model = cfg.task.dataset.language_emb_model
        if (
            "deepspeed_config" in cfg.training
            and cfg.training.deepspeed_config is not None
        ):
            language_emb_model = (
                None  # HACK: When training umi dataset on multiple nodes
            )
        self.model: UnifiedVideoActionPolicy = hydra.utils.instantiate(
            cfg.model.policy,
            task_name=cfg.task.name,
            task_modes=cfg.task.task_modes,
            normalizer_type=cfg.task.dataset.normalizer_type,
            language_emb_model=language_emb_model,
        )

        self.ema_model: UnifiedVideoActionPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer_parameters = cfg.model.policy.optimizer
        self.optimizer = self.model.get_optimizer(**cfg.model.policy.optimizer)

        # configure training state
        self.global_step = 0
        self.epoch = 0
        
        # Initialize Bayesian optimizer if enabled
        self.bayesian_optimizer = None
        if hasattr(cfg, 'bayesian_optimization') and cfg.bayesian_optimization.enabled:
            from unified_video_action.optimization.training_bayesian_optimizer import TrainingBayesianOptimizer
            self.bayesian_optimizer = TrainingBayesianOptimizer(
                config=cfg,
                start_epoch=cfg.bayesian_optimization.start_epoch,
                interval=cfg.bayesian_optimization.interval,
                max_trials=cfg.bayesian_optimization.max_trials,
                final_trials=cfg.bayesian_optimization.final_trials,
                n_test=cfg.bayesian_optimization.n_test,
                final_n_test=cfg.bayesian_optimization.final_n_test,
                device=cfg.bayesian_optimization.device,
                output_dir=cfg.bayesian_optimization.output_dir,
                use_best_checkpoint_for_final=cfg.bayesian_optimization.use_best_checkpoint_for_final,
                optimization_mode=cfg.bayesian_optimization.optimization_mode
            )
            print(f"Bayesian optimization enabled: start_epoch={cfg.bayesian_optimization.start_epoch}, interval={cfg.bayesian_optimization.interval}")
            
            # Print initial model parameters for reference
            print(f"\n{'='*50}")
            print(f"INITIAL MODEL PARAMETERS:")
            print(f"{'='*50}")
            if hasattr(self.model, 'autoregressive_model_params'):
                autoregressive_params = self.model.autoregressive_model_params
                print(f"Initial num_sampling_steps: {getattr(autoregressive_params, 'num_sampling_steps', 'N/A')}")
                print(f"Initial cfg: {getattr(autoregressive_params, 'cfg', 'N/A')}")
                print(f"Initial temperature: {getattr(autoregressive_params, 'temperature', 'N/A')}")
                print(f"Initial window_size: {getattr(autoregressive_params, 'window_size', 'N/A')}")
                print(f"Initial lambda_local: {getattr(autoregressive_params, 'lambda_local', 'N/A')}")
                print(f"Initial use_ucgm: {getattr(autoregressive_params, 'use_ucgm', 'N/A')}")
                
                if hasattr(autoregressive_params, 'ucgmts_config'):
                    ucgmts_config = autoregressive_params.ucgmts_config
                    print(f"Initial ucgmts_config:")
                    print(f"  transport_type: {getattr(ucgmts_config, 'transport_type', 'N/A')}")
                    print(f"  consistc_ratio: {getattr(ucgmts_config, 'consistc_ratio', 'N/A')}")
                    print(f"  scaled_cbl_eps: {getattr(ucgmts_config, 'scaled_cbl_eps', 'N/A')}")
                    print(f"  ema_decay_rate: {getattr(ucgmts_config, 'ema_decay_rate', 'N/A')}")
                    print(f"  rfba_gap_steps: {getattr(ucgmts_config, 'rfba_gap_steps', 'N/A')}")
                    print(f"  extrapol_ratio: {getattr(ucgmts_config, 'extrapol_ratio', 'N/A')}")
            print(f"{'='*50}")
    
    def freeze_submodules(self, action_only=False):
        # freeze submodules except the action diffusion head
        # Let's say you want to train only model.classifier
        if self.cfg.training.use_ema:
            models = [self.model, self.ema_model]
        else:
            models = [self.model]

        for model in models:
            # Freeze everything
            model.model.eval()
            for param in model.model.parameters():
                param.requires_grad = False
            
            print(f"Model type: {type(model.model)}")
            # Unfreeze only diffloss and diffactloss
            if not action_only:
                if hasattr(model.model, "diffloss"):
                    model.model.diffloss.train()
                    for param in model.model.diffloss.parameters():
                        param.requires_grad = True
                    print("Unfreezing diffloss")
            if hasattr(model.model, "diffactloss"):
                model.model.diffactloss.train()
                for param in model.model.diffactloss.parameters():
                    param.requires_grad = True
                print("Unfreezing diffactloss")
            
    def test_rollout(self):
        """
        Minimal rollout: build dataset -> get/set normalizer -> env rollout.
        No accelerator, no wandb, no training bits.
        """
        import copy
        import hydra

        cfg = copy.deepcopy(self.cfg)

        dataset = hydra.utils.instantiate(cfg.task.dataset)
        normalizer = dataset.get_normalizer()


        policy = getattr(self, "ema_model", None) or self.model
        policy = getattr(policy, "module", policy)
        policy.set_normalizer(normalizer)
        policy.to("cuda" if torch.cuda.is_available() else "cpu")
        policy.eval()

        accelerator = Accelerator(
                log_with="wandb", mixed_precision=self.cfg.training.mixed_precision
        )
        if accelerator.is_main_process:
            env_runners = load_env_runner(cfg, self.output_dir)
            with torch.no_grad():
                runner_log = env_rollout(cfg, env_runners, policy)

            print(runner_log)

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        if (
            "deepspeed_config" in cfg.training
            and cfg.training.deepspeed_config is not None
        ):
            deepspeed_plugin = DeepSpeedPlugin(
                hf_ds_config=cfg.training.deepspeed_config
            )
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
            accelerator = Accelerator(
                log_with="wandb",
                mixed_precision=self.cfg.training.mixed_precision,
                deepspeed_plugin=deepspeed_plugin,
                kwargs_handlers=[ddp_kwargs],
            )
        else:
            accelerator = Accelerator(
                log_with="wandb", mixed_precision=self.cfg.training.mixed_precision
            )

        if accelerator.is_main_process:
            cfg.logging.name = self.output_dir.split("/")[-1]
            wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
            wandb_cfg.pop("project")
            wandb_cfg["resume"] = "allow"

            accelerator.init_trackers(
                project_name=cfg.logging.project,
                config=OmegaConf.to_container(cfg, resolve=True),
                init_kwargs={"wandb": wandb_cfg},
            )

        if cfg.task.task_type == "multiple_datasets":
            dataset: UmiMultiDataset
            dataset = hydra.utils.instantiate(cfg.task.dataset)
            train_dataloader = dataset.get_dataloader()
            val_dataset = dataset.split_unused_episodes()
            val_dataloader = val_dataset.get_dataloader()
            dataset.set_datasets_attribute("random_img_sampling", True)
            print(
                "train dataset:",
                len(dataset),
                "train dataloader:",
                len(train_dataloader),
            )
            print(
                "val dataset:", len(val_dataset), "val dataloader:", len(val_dataloader)
            )
        else:
            # configure dataset
            dataset: BaseImageDataset
            dataset = hydra.utils.instantiate(cfg.task.dataset)
            train_dataloader = DataLoader(dataset, **cfg.dataloader)

            # configure validation dataset
            val_dataset = dataset.get_validation_dataset()
            val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
            print(
                "train dataset:",
                len(dataset),
                "train dataloader:",
                len(train_dataloader),
            )
            print(
                "val dataset:", len(val_dataset), "val dataloader:", len(val_dataloader)
            )

            # compute normalizer on the main process and save to disk
            normalizer_path = os.path.join(self.output_dir, "normalizer.pkl")
            if accelerator.is_main_process:
                normalizer = dataset.get_normalizer()
                pickle.dump(normalizer, open(normalizer_path, "wb"))

        if (
            "deepspeed_config" not in cfg.training
            or cfg.training.deepspeed_config is None
        ):
            accelerator.wait_for_everyone()

        # load normalizer on all processes
        if cfg.task.task_type == "single_dataset":
            normalizer = pickle.load(open(normalizer_path, "rb"))

            self.model.set_normalizer(normalizer)
            if cfg.training.use_ema:
                self.ema_model.set_normalizer(normalizer)
        

        # configure lr scheduler
        self.lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs)
            // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1,
        )

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                accelerator.print(f"Resuming from checkpoint {lastest_ckpt_path}") 
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # configure env
        if (
            cfg.model.policy.action_model_params.predict_action
            and "env_runner" in cfg.task and accelerator.is_main_process and cfg.training.max_train_steps is None
        ):
            env_runners = load_env_runner(cfg, self.output_dir)

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"), **cfg.checkpoint.topk
        )

        # accelerator
        (
            train_dataloader,
            val_dataloader,
            self.model,
            self.optimizer,
            self.lr_scheduler,
        ) = accelerator.prepare(
            train_dataloader,
            val_dataloader,
            self.model,
            self.optimizer,
            self.lr_scheduler,
        )

        device = self.model.device

        if self.ema_model is not None:
            self.ema_model.to(device)

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        #print(f"self.model.normalizer.params_dict.action.scale {self.model.normalizer.params_dict.action.scale}")
        #print(f"self.ema_model.normalizer.params_dict.action.scale {self.ema_model.normalizer.params_dict.action.scale}")
        total_params       = sum(p.numel() for p in self.model.parameters())
        trainable_params   = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        trainable_percent  = 100.0 * trainable_params / total_params

        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
            f"({trainable_percent:.2f}%)")

        for local_epoch_idx in range(cfg.training.num_epochs):
            step_log = dict()
            train_losses = list()
            with tqdm.tqdm(
                train_dataloader,
                desc=f"Training epoch {self.epoch}",
                leave=False,
                mininterval=cfg.training.tqdm_interval_sec,
            ) as tepoch:
                for batch_idx, batch in enumerate(tepoch):

                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    # resize image
                    batch = resize_image(cfg, batch)
                    # compute loss
                    if (
                        "deepspeed_config" in cfg.training
                        and cfg.training.deepspeed_config is not None
                    ): 
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): # You might need to change the device_type to str(device) for other versions of torch
                            raw_loss, (loss_diffusion, loss_action) = self.model(batch)
                    else:
                        raw_loss, (loss_diffusion, loss_action) = self.model(batch)
                    
                    # backward pass
                    accelerator.backward(raw_loss)
                        
                    scale = accelerator.scaler.get_scale()
                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.lr_scheduler.step()

                    # update ema
                    if cfg.training.use_ema:
                        ema.step(accelerator.unwrap_model(self.model))

                    # logging
                    raw_loss_cpu = raw_loss.item()

                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)

                    if cfg.model.policy.autoregressive_model_params.predict_video:
                        loss_diffusion_cpu = loss_diffusion.item()
                    else:
                        loss_diffusion_cpu = 0.0

                    if cfg.model.policy.action_model_params.predict_action:
                        loss_action_cpu = loss_action.item()
                    else:
                        loss_action_cpu = 0.0

                    step_log = {
                        "AMP scale": scale,
                        "train_loss": raw_loss_cpu,
                        "diffusion_loss": loss_diffusion_cpu,
                        "action_loss": loss_action_cpu,
                        "global_step": self.global_step,
                        "epoch": self.epoch,
                        "lr": self.lr_scheduler.get_last_lr()[0],
                    }

                    is_last_batch = batch_idx == (len(train_dataloader) - 1)
                    if not is_last_batch:
                        accelerator.log(step_log, step=self.global_step)
                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) and batch_idx >= (
                        cfg.training.max_train_steps - 1
                    ):
                        break


            train_loss = np.mean(train_losses)
            step_log["train_loss"] = train_loss

            # ========= eval for this epoch ==========
            policy = accelerator.unwrap_model(self.model)
            if cfg.training.use_ema:
                policy = self.ema_model
            policy.eval()

            if cfg.training.max_train_steps is None:
                # ========= evaluate val action error =========
                if (
                    cfg.model.policy.action_model_params.predict_action
                    and "env_runner" not in cfg.task
                ):
                    ## if has similartor, skip this
                    act_log = test_action_l2(
                        cfg,
                        policy,
                        val_dataloader,
                        local_epoch_idx,
                        self.output_dir,
                        device,
                    )
                    step_log.update(act_log)

                # ========= simulator: run rollout =========            
                if (
                    cfg.model.policy.action_model_params.predict_action
                    and "env_runner" in cfg.task and accelerator.is_main_process
                ):
                    if (self.epoch % cfg.training.rollout_every) == 0:
                        runner_log = env_rollout(cfg, env_runners, policy)
                        step_log.update(runner_log)

                # ========= checkpoint ==========
                if (
                    self.epoch % cfg.training.checkpoint_every
                ) == 0 and accelerator.is_main_process:
                    # unwrap the model to save ckpt
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)

                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()

                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace("/", "_")
                        metric_dict[new_key] = value

                    # save topk checkpoints
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                    # recover the DDP model
                    self.model = model_ddp
            accelerator.wait_for_everyone()
            # ========= eval end for this epoch ==========
            
            # ========= Bayesian Optimization ==========
            if (self.bayesian_optimizer is not None and 
                self.bayesian_optimizer.should_optimize(self.epoch) and 
                accelerator.is_main_process):
                
                print(f"\n{'='*60}")
                print(f"Starting Bayesian optimization at epoch {self.epoch}")
                print(f"{'='*60}")
                
                # Get current checkpoint path - try multiple possible locations
                checkpoint_path = None
                possible_paths = [
                    os.path.join(self.output_dir, "checkpoints", "latest.ckpt"),
                    os.path.join(self.output_dir, "checkpoints", "last.ckpt"),
                    os.path.join(self.output_dir, "checkpoints", f"epoch={self.epoch:04d}-test_mean_score={step_log.get('test_mean_score', 0.0):.3f}.ckpt")
                ]
                
                # Also try to find any checkpoint file in the checkpoints directory
                checkpoints_dir = os.path.join(self.output_dir, "checkpoints")
                if os.path.exists(checkpoints_dir):
                    checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.ckpt')]
                    if checkpoint_files:
                        # Use the most recent checkpoint file
                        checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
                        possible_paths.append(os.path.join(checkpoints_dir, checkpoint_files[0]))
                
                # Wait a bit to ensure checkpoint is fully written
                import time
                time.sleep(3)
                
                for path in possible_paths:
                    if os.path.exists(path):
                        # Check if checkpoint is valid before using it
                        if self.bayesian_optimizer._is_checkpoint_valid(path):
                            checkpoint_path = path
                            break
                        else:
                            print(f"Skipping invalid checkpoint: {path}")
                
                # If no valid checkpoint found, try to use the most recent one anyway (fallback)
                if checkpoint_path is None:
                    checkpoints_dir = os.path.join(self.output_dir, "checkpoints")
                    if os.path.exists(checkpoints_dir):
                        checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.ckpt')]
                        if checkpoint_files:
                            # Use the most recent checkpoint file as fallback
                            checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
                            fallback_path = os.path.join(checkpoints_dir, checkpoint_files[0])
                            print(f"Using fallback checkpoint (validation failed but file exists): {fallback_path}")
                            checkpoint_path = fallback_path
                
                if checkpoint_path and os.path.exists(checkpoint_path):
                    print(f"Using checkpoint: {checkpoint_path}")
                    # Run Bayesian optimization
                    checkpoints_dir = os.path.join(self.output_dir, "checkpoints")
                    optimization_result = self.bayesian_optimizer.run_optimization(checkpoint_path, self.epoch, checkpoints_dir)
                    
                    if optimization_result is not None:
                        best_params, best_score = optimization_result
                        
                        # Apply best parameters to model with score comparison
                        # Always use the main model, not EMA model, for parameter updates
                        policy = accelerator.unwrap_model(self.model)
                        
                        print(f"Policy type: {type(policy)}")
                        print(f"Policy attributes: {[attr for attr in dir(policy) if not attr.startswith('_')]}")
                        
                        success = self.bayesian_optimizer.apply_best_params_to_model(
                            policy, 
                            best_params, 
                            workspace=self,
                            current_checkpoint_path=checkpoint_path,
                            optimized_score=best_score
                        )
                        
                        if success:
                            print(f" Successfully applied optimized parameters to model")
                            print(f"   Optimized score: {best_score:.4f}")
                            
                            # CRITICAL: Also update workspace cfg to ensure parameters are saved in checkpoint
                            if hasattr(self, 'cfg') and self.cfg is not None:
                                with open_dict(self.cfg.model.policy.autoregressive_model_params):
                                    if "ucgmts_config" not in self.cfg.model.policy.autoregressive_model_params:
                                        self.cfg.model.policy.autoregressive_model_params.ucgmts_config = OmegaConf.create({})
                                    
                                    self.cfg.model.policy.autoregressive_model_params.use_ucgm = True
                                    self.cfg.model.policy.autoregressive_model_params.num_sampling_steps = best_params['num_sampling_steps']
                                    self.cfg.model.policy.autoregressive_model_params.cfg = best_params['cfg']
                                    self.cfg.model.policy.autoregressive_model_params.temperature = best_params['temperature']
                                    self.cfg.model.policy.autoregressive_model_params.window_size = best_params['window_size']
                                    self.cfg.model.policy.autoregressive_model_params.lambda_local = best_params['lambda_local']
                                    
                                    self.cfg.model.policy.autoregressive_model_params.ucgmts_config.transport_type = best_params['ucgmts_config']['transport_type']
                                    self.cfg.model.policy.autoregressive_model_params.ucgmts_config.consistc_ratio = best_params['ucgmts_config']['consistc_ratio']
                                    self.cfg.model.policy.autoregressive_model_params.ucgmts_config.scaled_cbl_eps = best_params['ucgmts_config']['scaled_cbl_eps']
                                    self.cfg.model.policy.autoregressive_model_params.ucgmts_config.ema_decay_rate = best_params['ucgmts_config']['ema_decay_rate']
                                    self.cfg.model.policy.autoregressive_model_params.ucgmts_config.rfba_gap_steps = best_params['ucgmts_config']['rfba_gap_steps']
                                    self.cfg.model.policy.autoregressive_model_params.ucgmts_config.extrapol_ratio = best_params['ucgmts_config']['extrapol_ratio']
                                    
                                    print("✓ Updated workspace.cfg with optimized parameters")
                            
                            # Verify parameters were actually applied by checking the model
                            print(f"\n{'='*50}")
                            print(f"VERIFICATION: Checking model parameters after application:")
                            print(f"{'='*50}")
                            if hasattr(policy, 'autoregressive_model_params'):
                                autoregressive_params = policy.autoregressive_model_params
                                print(f" Verified num_sampling_steps: {autoregressive_params.num_sampling_steps}")
                                print(f" Verified cfg: {autoregressive_params.cfg}")
                                print(f" Verified temperature: {autoregressive_params.temperature}")
                                print(f" Verified window_size: {autoregressive_params.window_size}")
                                print(f" Verified lambda_local: {autoregressive_params.lambda_local}")
                                print(f" Verified use_ucgm: {autoregressive_params.use_ucgm}")
                                
                                if hasattr(autoregressive_params, 'ucgmts_config'):
                                    ucgmts_config = autoregressive_params.ucgmts_config
                                    print(f" Verified ucgmts_config:")
                                    print(f"    transport_type: {ucgmts_config.transport_type}")
                                    print(f"    consistc_ratio: {ucgmts_config.consistc_ratio}")
                                    print(f"    rfba_gap_steps: {ucgmts_config.rfba_gap_steps}")
                                    print(f"    extrapol_ratio: {ucgmts_config.extrapol_ratio}")
                                
                                # Also verify the actual model components
                                if hasattr(policy, 'model') and hasattr(policy.model, 'diffactloss'):
                                    diffactloss = policy.model.diffactloss
                                    print(f" Verified DiffActLoss num_sampling_steps: {diffactloss.num_sampling_steps}")
                                    if hasattr(diffactloss, 'ucgmts'):
                                        ucgmts = diffactloss.ucgmts
                                        print(f" Verified UCGMTS parameters:")
                                        print(f"    consistc_ratio: {getattr(ucgmts, 'cor', 'N/A')}")
                                        print(f"    scaled_cbl_eps: {getattr(ucgmts, 'huc', 'N/A')}")
                                        print(f"    ema_decay_rate: {getattr(ucgmts, 'emd', 'N/A')}")
                                        print(f"    (transport_type is fixed and not optimized)")
                                        print(f"    (rfba_gap_steps and extrapol_ratio are passed during sampling)")
                            print(f"{'='*50}")
                            
                            # Log optimization results
                            optimization_log = {
                                "bayesian_optimization_score": self.bayesian_optimizer.best_score,
                                "bayesian_optimization_epoch": self.epoch,
                                "bayesian_optimization_params": best_params
                            }
                            step_log.update(optimization_log)
                        else:
                            print(f"Failed to apply optimized parameters to model")
                    else:
                        print(f"Bayesian optimization failed at epoch {self.epoch}")
                else:
                    print(f"No valid checkpoint found for Bayesian optimization. Searched paths:")
                    for path in possible_paths:
                        if os.path.exists(path):
                            print(f"  - {path} (exists but invalid)")
                        else:
                            print(f"  - {path} (not found)")
                    print(f"Skipping Bayesian optimization at epoch {self.epoch}")
                    print(f"Will retry at next optimization interval")
            
            policy.model.diffactloss.train()
            # policy.train()
            accelerator.log(step_log, step=self.global_step)
            self.global_step += 1
            self.epoch += 1

        accelerator.end_training()
        
        # Final Bayesian optimization after training completion
        if (self.bayesian_optimizer is not None and 
            accelerator.is_main_process and 
            self.epoch >= cfg.training.num_epochs):
            
            print(f"\n{'='*60}")
            print(f"FINAL BAYESIAN OPTIMIZATION")
            print(f"{'='*60}")
            print(f"Training completed! Running final optimization with maximum resources...")
            
            # Get the final checkpoint
            checkpoint_path = None
            possible_paths = [
                os.path.join(self.output_dir, "checkpoints", "latest.ckpt"),
                os.path.join(self.output_dir, "checkpoints", "last.ckpt"),
            ]
            
            # Also try to find any checkpoint file in the checkpoints directory
            checkpoints_dir = os.path.join(self.output_dir, "checkpoints")
            if os.path.exists(checkpoints_dir):
                checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.ckpt')]
                if checkpoint_files:
                    # Use the most recent checkpoint file
                    checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
                    possible_paths.append(os.path.join(checkpoints_dir, checkpoint_files[0]))
            
            for path in possible_paths:
                if os.path.exists(path) and self.bayesian_optimizer._is_checkpoint_valid(path):
                    checkpoint_path = path
                    break
            
            if checkpoint_path:
                print(f"Using final checkpoint: {checkpoint_path}")
                # Run final optimization
                checkpoints_dir = os.path.join(self.output_dir, "checkpoints")
                best_params = self.bayesian_optimizer.run_optimization(checkpoint_path, self.epoch, checkpoints_dir)
                
                if best_params:
                    print(f"\n{'='*60}")
                    print(f"FINAL OPTIMIZATION COMPLETED!")
                    print(f"{'='*60}")
                    print(f"Best parameters found: {best_params}")
                    
                    # Apply final parameters using the EXACT method that works
                    print(f"Applying final optimized parameters using exact method...")
                    
                    # Convert Bayesian optimization params to manual evaluation params
                    manual_params = {
                        'use_ucgm': True,
                        'num_sampling_steps': best_params['num_sampling_steps'],
                        'stochasticity_rate': best_params['consistc_ratio'],  # This is the key!
                        'window_size': best_params['window_size'],
                        'lambda_local': best_params['lambda_local']
                    }
                    
                    # Apply parameters EXACTLY as eval_sim.py does
                    with open_dict(self.cfg.model.policy.autoregressive_model_params):
                        # Create ucgmts_config if it doesn't exist (exactly as eval_sim.py does)
                        if "ucgmts_config" not in self.cfg.model.policy.autoregressive_model_params:
                            self.cfg.model.policy.autoregressive_model_params.ucgmts_config = OmegaConf.create({})
                            self.cfg.model.policy.autoregressive_model_params.ucgmts_config.transport_type = "Linear"
                            self.cfg.model.policy.autoregressive_model_params.ucgmts_config.scaled_cbs_eps = 0.0
                            self.cfg.model.policy.autoregressive_model_params.ucgmts_config.ema_decay_rate = 0.0
                        
                        # Apply stochasticity_rate (consistc_ratio)
                        if manual_params['stochasticity_rate'] is not None:
                            self.cfg.model.policy.autoregressive_model_params.ucgmts_config.consistc_ratio = manual_params['stochasticity_rate']
                        
                        # Apply num_sampling_steps and rfba_gap_steps (exactly as eval_sim.py does)
                        if manual_params['num_sampling_steps']:
                            self.cfg.model.policy.autoregressive_model_params.num_sampling_steps = manual_params['num_sampling_steps']
                            if manual_params['num_sampling_steps'] <= 2:
                                self.cfg.model.policy.autoregressive_model_params.ucgmts_config.rfba_gap_steps = [0.001, 0.5] 
                            else:
                                self.cfg.model.policy.autoregressive_model_params.ucgmts_config.rfba_gap_steps = [0.001, 0.001]
                        
                        # Apply local attention parameters
                        if manual_params['window_size'] is not None:
                            self.cfg.model.policy.autoregressive_model_params.window_size = manual_params['window_size']
                        if manual_params['lambda_local'] is not None:
                            self.cfg.model.policy.autoregressive_model_params.lambda_local = manual_params['lambda_local']
                    
                    # Apply use_ucgm flag (exactly as eval_sim.py does)
                    if manual_params['use_ucgm']:
                        OmegaConf.set_struct(self.cfg, False)  # Allow new keys
                        self.cfg.model.policy.autoregressive_model_params.use_ucgm = True
                        print("Using UCGM mode")
                    
                    # Apply pruning settings (exactly as eval_sim.py does)
                    OmegaConf.set_struct(self.cfg, False)  # Allow new keys
                    self.cfg.model.policy.autoregressive_model_params.pruning_ratios = None
                    self.cfg.model.policy.autoregressive_model_params.token_pruning = False
                    self.cfg.model.policy.autoregressive_model_params.restore_after_encoder = False
                    
                    print("✓ Applied parameters exactly as eval_sim.py does")
                    
                    # Apply parameters to model components
                    print(f"Applying parameters to model components...")
                    model = self.model
                    
                    # Apply to autoregressive_model_params
                    if hasattr(model, 'autoregressive_model_params'):
                        autoregressive_params = model.autoregressive_model_params
                    elif hasattr(model, 'model') and hasattr(model.model, 'autoregressive_model_params'):
                        autoregressive_params = model.model.autoregressive_model_params
                    else:
                        print(" Cannot find autoregressive_model_params")
                        return False
                    
                    autoregressive_params.use_ucgm = True
                    autoregressive_params.num_sampling_steps = manual_params['num_sampling_steps']
                    autoregressive_params.window_size = manual_params['window_size']
                    autoregressive_params.lambda_local = manual_params['lambda_local']
                    
                    if not hasattr(autoregressive_params, 'ucgmts_config'):
                        autoregressive_params.ucgmts_config = OmegaConf.create({})
                    
                    autoregressive_params.ucgmts_config.transport_type = "Linear"
                    autoregressive_params.ucgmts_config.consistc_ratio = manual_params['stochasticity_rate']
                    autoregressive_params.ucgmts_config.scaled_cbs_eps = 0.0
                    autoregressive_params.ucgmts_config.ema_decay_rate = 0.0
                    autoregressive_params.ucgmts_config.rfba_gap_steps = [0.001, 0.5]  # Exactly as eval_sim.py sets it
                    
                    print("✓ Updated autoregressive_model_params")
                    
                    # Apply to actual model components
                    if hasattr(model, 'model') and hasattr(model.model, 'diffactloss'):
                        diffactloss = model.model.diffactloss
                        diffactloss.num_sampling_steps = manual_params['num_sampling_steps']
                        
                        if hasattr(diffactloss, 'ucgmts'):
                            ucgmts = diffactloss.ucgmts
                            ucgmts.transport_type = "Linear"
                            ucgmts.consistc_ratio = manual_params['stochasticity_rate']
                            ucgmts.scaled_cbs_eps = 0.0
                            ucgmts.ema_decay_rate = 0.0
                            ucgmts.rfba_gap_steps = [0.001, 0.5]  # Exactly as eval_sim.py sets it
                            print("✓ Updated UCGMTS model components")
                    
                    # Also update EMA model if it exists
                    if hasattr(self, 'ema_model') and self.ema_model is not None:
                        print(f"Updating EMA model...")
                        ema_model = self.ema_model
                        
                        if hasattr(ema_model, 'autoregressive_model_params'):
                            ema_autoregressive_params = ema_model.autoregressive_model_params
                        elif hasattr(ema_model, 'model') and hasattr(ema_model.model, 'autoregressive_model_params'):
                            ema_autoregressive_params = ema_model.model.autoregressive_model_params
                        else:
                            print("Cannot find EMA autoregressive_model_params")
                            return False
                        
                        ema_autoregressive_params.use_ucgm = True
                        ema_autoregressive_params.num_sampling_steps = manual_params['num_sampling_steps']
                        ema_autoregressive_params.window_size = manual_params['window_size']
                        ema_autoregressive_params.lambda_local = manual_params['lambda_local']
                        
                        if not hasattr(ema_autoregressive_params, 'ucgmts_config'):
                            ema_autoregressive_params.ucgmts_config = OmegaConf.create({})
                        
                        ema_autoregressive_params.ucgmts_config.transport_type = "Linear"
                        ema_autoregressive_params.ucgmts_config.consistc_ratio = manual_params['stochasticity_rate']
                        ema_autoregressive_params.ucgmts_config.scaled_cbs_eps = 0.0
                        ema_autoregressive_params.ucgmts_config.ema_decay_rate = 0.0
                        ema_autoregressive_params.ucgmts_config.rfba_gap_steps = [0.001, 0.5]  # Exactly as eval_sim.py sets it
                        
                        # Also update EMA model components
                        if hasattr(ema_model, 'model') and hasattr(ema_model.model, 'diffactloss'):
                            ema_diffactloss = ema_model.model.diffactloss
                            ema_diffactloss.num_sampling_steps = manual_params['num_sampling_steps']
                            
                            if hasattr(ema_diffactloss, 'ucgmts'):
                                ema_ucgmts = ema_diffactloss.ucgmts
                                ema_ucgmts.transport_type = "Linear"
                                ema_ucgmts.consistc_ratio = manual_params['stochasticity_rate']
                                ema_ucgmts.scaled_cbs_eps = 0.0
                                ema_ucgmts.ema_decay_rate = 0.0
                                ema_ucgmts.rfba_gap_steps = [0.001, 0.5]  # Exactly as eval_sim.py sets it
                        
                        print("✓ Updated EMA model")
                    
                    print(f"✓ Final optimized parameters applied to model using exact method!")
                    
                    # Save the final optimized model with a special name
                    final_model_path = os.path.join(self.output_dir, "checkpoints", 
                                                  f"final_optimized_model_score={self.bayesian_optimizer.best_score:.3f}.ckpt")
                    print(f"Saving final optimized model to: {final_model_path}")
                    self.save_checkpoint(path=final_model_path)
                    print(f"✓ Final optimized model saved successfully!")
                    print(f"✓ This checkpoint should achieve the same performance as manual evaluation!")
                else:
                    print(f"Final optimization failed")
            else:
                print(f"No valid checkpoint found for final optimization")
        
        # Print Bayesian optimization summary if enabled
        if self.bayesian_optimizer is not None and accelerator.is_main_process:
            summary = self.bayesian_optimizer.get_optimization_summary()
            print(f"\n{'='*60}")
            print(f"BAYESIAN OPTIMIZATION SUMMARY")
            print(f"{'='*60}")
            print(f"Total optimizations performed: {summary['total_optimizations']}")
            print(f"Best score achieved: {summary['best_score']:.4f}")
            if summary['best_params']:
                print(f"Best parameters: {summary['best_params']}")
            print(f"Optimization logs saved to: {self.bayesian_optimizer.output_dir}")
            print(f"{'='*60}")