#!/usr/bin/env python3
"""
将贝叶斯优化得到的参数应用到checkpoint中，这样可以确保评估时使用完全相同的参数。
"""

import torch
import dill
from omegaconf import OmegaConf, open_dict

def apply_bayesian_params_to_checkpoint(input_checkpoint, output_checkpoint):
    """
    将贝叶斯优化得到的参数应用到checkpoint
    """
    
    # 从贝叶斯优化结果加载最优参数（只保存有效的4个参数）
    best_params = {
        "num_sampling_steps": 2,
        "stochasticity_rate": 0.8225843668899645,  # consistc_ratio
        "window_size": 12,
        "lambda_local": 0.13521376899022036,
    }
    
    print("="*60)
    print("APPLYING BAYESIAN OPTIMIZATION PARAMETERS TO CHECKPOINT")
    print("="*60)
    print(f"Input: {input_checkpoint}")
    print(f"Output: {output_checkpoint}")
    print("\nApplying parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint...")
    payload = torch.load(open(input_checkpoint, "rb"), pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    
    # Apply parameters to config
    print("\nApplying parameters to config...")
    
    with open_dict(cfg.model.policy.autoregressive_model_params):
        # Create ucgmts_config if it doesn't exist (exactly as eval_sim.py does)
        if "ucgmts_config" not in cfg.model.policy.autoregressive_model_params:
            cfg.model.policy.autoregressive_model_params.ucgmts_config = OmegaConf.create({})
            cfg.model.policy.autoregressive_model_params.ucgmts_config.transport_type = "Linear"
            cfg.model.policy.autoregressive_model_params.ucgmts_config.scaled_cbs_eps = 0.0
            cfg.model.policy.autoregressive_model_params.ucgmts_config.ema_decay_rate = 0.0
        
        # Apply UCGM settings
        cfg.model.policy.autoregressive_model_params.use_ucgm = True
        
        # Apply main parameters (exactly as eval_sim.py does)
        if best_params['num_sampling_steps']:
            cfg.model.policy.autoregressive_model_params.num_sampling_steps = best_params['num_sampling_steps']
            if best_params['num_sampling_steps'] <= 2:
                cfg.model.policy.autoregressive_model_params.ucgmts_config.rfba_gap_steps = [0.001, 0.5] 
            else:
                cfg.model.policy.autoregressive_model_params.ucgmts_config.rfba_gap_steps = [0.001, 0.001]
        
        # Apply local attention parameters
        if best_params['window_size'] is not None:
            cfg.model.policy.autoregressive_model_params.window_size = best_params['window_size']
        if best_params['lambda_local'] is not None:
            cfg.model.policy.autoregressive_model_params.lambda_local = best_params['lambda_local']
        
        # Apply stochasticity_rate (consistc_ratio)
        if best_params['stochasticity_rate'] is not None:
            cfg.model.policy.autoregressive_model_params.ucgmts_config.consistc_ratio = best_params['stochasticity_rate']
    
    # Update checkpoint payload
    payload["cfg"] = cfg
    
    # Save modified checkpoint
    print("\nSaving modified checkpoint...")
    with open(output_checkpoint, "wb") as f:
        torch.save(payload, f, pickle_module=dill)
    
    print("\nVerification - Config parameters after modification:")
    params = cfg.model.policy.autoregressive_model_params
    print(f"  use_ucgm: {params.use_ucgm}")
    print(f"  num_sampling_steps: {params.num_sampling_steps}")
    print(f"  window_size: {params.window_size}")
    print(f"  lambda_local: {params.lambda_local}")
    print(f"  ucgmts_config:")
    print(f"    consistc_ratio: {params.ucgmts_config.consistc_ratio}")
    print(f"    rfba_gap_steps: {params.ucgmts_config.rfba_gap_steps}")
    
    print(f"\n✓ Checkpoint saved to: {output_checkpoint}")
    print("\n现在你可以用这个checkpoint复现贝叶斯优化的0.9964结果了！")
    print("\n使用方法：")
    print(f"  python eval_sim.py \\")
    print(f"    --checkpoint {output_checkpoint} \\")
    print(f"    --output_dir results/ \\")
    print(f"    --use_ucgm \\")
    print(f"    --device cuda:0")

if __name__ == "__main__":
    import sys
    import shutil
    import pathlib
    
    input_ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/push_t_dit_hybrid.ckpt"
    output_ckpt = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/push_t_dit_hybrid_bayesian.ckpt"
    
    # 自动创建备份
    input_path = pathlib.Path(input_ckpt)
    if input_path.exists():
        backup_path = input_path.parent / f"{input_path.stem}_original_backup{input_path.suffix}"
        if not backup_path.exists():
            print(f"\n🔄 Creating backup: {backup_path}")
            shutil.copy2(input_ckpt, backup_path)
            print(f"✓ Backup created successfully!")
        else:
            print(f"\n⚠️  Backup already exists: {backup_path}")
            print(f"   (Skipping backup creation)")
    
    apply_bayesian_params_to_checkpoint(input_ckpt, output_ckpt)

