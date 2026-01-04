#!/usr/bin/env python3
"""
统计模型三个主要部分的参数量：
1. VAE部分 (encoder + decoder + quant_conv + post_quant_conv)
2. Transformer部分 (encoder + decoder + 投影层等)
3. Diffusion head部分 (diffloss.net + diffactloss中的网络)
"""

import torch
import torch.nn as nn
from collections import OrderedDict
import hydra
from omegaconf import OmegaConf
import sys
import pathlib

# 添加项目路径
ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)

from unified_video_action.policy.unified_video_action_policy import UnifiedVideoActionPolicy


def count_parameters(model, only_trainable=False):
    """
    统计模型参数量，支持nn.Module和nn.Parameter
    
    Args:
        model: 要统计的模型或参数
        only_trainable: 如果为True，只统计requires_grad=True的参数；如果为False，统计所有参数
    """
    import torch.nn as nn
    if isinstance(model, nn.Parameter):
        # 如果是单个Parameter对象，直接返回其参数量
        if only_trainable:
            return model.numel() if model.requires_grad else 0
        else:
            return model.numel()
    elif isinstance(model, nn.Module):
        # 如果是Module，统计所有参数
        if only_trainable:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in model.parameters())
    else:
        # 其他情况（如tensor），尝试直接获取numel
        try:
            if only_trainable:
                return model.numel() if hasattr(model, 'requires_grad') and model.requires_grad else 0
            else:
                return model.numel()
        except:
            return 0


def count_parameters_by_module(model, module_name_prefix=""):
    """递归统计模块参数量"""
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if module_name_prefix == "" or name.startswith(module_name_prefix):
                total += param.numel()
    return total


def format_number(num):
    """格式化数字，便于阅读"""
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
    else:
        return str(num)


def analyze_model_parameters(policy_model):
    """
    分析模型三个部分的参数量
    
    Args:
        policy_model: UnifiedVideoActionPolicy实例
    
    Returns:
        dict: 包含各部分参数量的字典
    """
    results = {
        "vae": 0,
        "transformer": 0,
        "diffusion_head": 0,
        "other": 0,
        "total": 0
    }
    
    # 获取MAR模型
    mar_model = policy_model.model
    
    # 1. VAE部分
    print(f"\n=== VAE部分 ===")
    if hasattr(policy_model, 'vae_model'):
        if policy_model.vae_model is not None:
            vae_model = policy_model.vae_model
            # VAE包括encoder、decoder、quant_conv、post_quant_conv
            # 注意：VAE参数通常被冻结（requires_grad=False），但我们需要统计所有参数
            vae_params = count_parameters(vae_model, only_trainable=False)
            results["vae"] = vae_params
            print(f"总参数量: {vae_params:,} ({format_number(vae_params)})")
            
            # 详细统计VAE各部分
            if hasattr(vae_model, 'encoder'):
                encoder_params = count_parameters(vae_model.encoder, only_trainable=False)
                print(f"  - Encoder: {encoder_params:,} ({format_number(encoder_params)})")
            
            if hasattr(vae_model, 'decoder'):
                decoder_params = count_parameters(vae_model.decoder, only_trainable=False)
                print(f"  - Decoder: {decoder_params:,} ({format_number(decoder_params)})")
            
            if hasattr(vae_model, 'quant_conv'):
                quant_conv_params = count_parameters(vae_model.quant_conv, only_trainable=False)
                print(f"  - Quant Conv: {quant_conv_params:,} ({format_number(quant_conv_params)})")
            
            if hasattr(vae_model, 'post_quant_conv'):
                post_quant_conv_params = count_parameters(vae_model.post_quant_conv, only_trainable=False)
                print(f"  - Post Quant Conv: {post_quant_conv_params:,} ({format_number(post_quant_conv_params)})")
        else:
            print("警告: vae_model 属性存在但值为 None")
            print("提示: VAE模型会在实例化时自动创建，如果路径不存在可能无法加载")
            results["vae"] = 0
    else:
        print("警告: policy_model 没有 vae_model 属性")
        print("提示: VAE模型会在实例化时自动创建，如果路径不存在可能无法加载")
        results["vae"] = 0
    
    # 2. Transformer部分
    print(f"\n=== Transformer部分 ===")
    transformer_params = 0
    
    # Encoder blocks
    if hasattr(mar_model, 'encoder_blocks'):
        encoder_blocks_params = count_parameters(mar_model.encoder_blocks)
        transformer_params += encoder_blocks_params
        print(f"  - Encoder Blocks: {encoder_blocks_params:,} ({format_number(encoder_blocks_params)})")
    
    # Local causal encoder blocks
    if hasattr(mar_model, 'local_causal_encoder_blocks'):
        local_causal_params = count_parameters(mar_model.local_causal_encoder_blocks)
        transformer_params += local_causal_params
        print(f"  - Local Causal Encoder Blocks: {local_causal_params:,} ({format_number(local_causal_params)})")
    
    # Decoder blocks
    if hasattr(mar_model, 'decoder_blocks'):
        decoder_blocks_params = count_parameters(mar_model.decoder_blocks)
        transformer_params += decoder_blocks_params
        print(f"  - Decoder Blocks: {decoder_blocks_params:,} ({format_number(decoder_blocks_params)})")
    
    # 投影层
    projection_params = 0
    projection_modules = [
        'z_proj', 'z_proj_cond', 'z_proj_wrist',
        'action_proj_cond', 'history_action_proj_cond',
        'proprioception_proj_cond', 'proprioception_image_proj_cond',
        'text_proj_cond', 'proj_cond_x_layer',
        'decoder_embed', 'feature_fusion'
    ]
    for module_name in projection_modules:
        if hasattr(mar_model, module_name):
            module = getattr(mar_model, module_name)
            if module is not None:
                params = count_parameters(module)
                projection_params += params
                print(f"  - {module_name}: {params:,} ({format_number(params)})")
    
    transformer_params += projection_params
    
    # 位置编码
    pos_embed_params = 0
    pos_embed_modules = [
        'temporal_pos_embed', 'spatial_pos_embed',
        'decoder_temporal_pos_embed', 'decoder_spatial_pos_embed',
        'decoder_text_pos_embed', 'text_pos_embed',
        'diffusion_temporal_embed', 'diffusion_spatial_embed'
    ]
    for module_name in pos_embed_modules:
        if hasattr(mar_model, module_name):
            module = getattr(mar_model, module_name)
            if module is not None:
                params = count_parameters(module)
                pos_embed_params += params
                print(f"  - {module_name}: {params:,} ({format_number(params)})")
    
    transformer_params += pos_embed_params
    
    # 归一化层
    norm_params = 0
    norm_modules = [
        'z_proj_ln', 'encoder_norm', 'local_causal_encoder_norm',
        'decoder_norm'
    ]
    for module_name in norm_modules:
        if hasattr(mar_model, module_name):
            module = getattr(mar_model, module_name)
            if module is not None:
                params = count_parameters(module)
                norm_params += params
                print(f"  - {module_name}: {params:,} ({format_number(params)})")
    
    transformer_params += norm_params
    
    # Fake latent参数
    fake_latent_params = 0
    fake_latent_modules = [
        'fake_latent_x', 'fake_action_latent', 'fake_latent_wrist_x',
        'fake_latent_history_action', 'fake_latent'
    ]
    for module_name in fake_latent_modules:
        if hasattr(mar_model, module_name):
            module = getattr(mar_model, module_name)
            if module is not None:
                params = count_parameters(module)
                fake_latent_params += params
                print(f"  - {module_name}: {params:,} ({format_number(params)})")
    
    transformer_params += fake_latent_params
    
    results["transformer"] = transformer_params
    print(f"Transformer总参数量: {transformer_params:,} ({format_number(transformer_params)})")
    
    # 3. Diffusion head部分
    print(f"\n=== Diffusion Head部分 ===")
    diffusion_params = 0
    
    # Video diffusion loss
    if hasattr(mar_model, 'diffloss') and mar_model.diffloss is not None:
        if hasattr(mar_model.diffloss, 'net'):
            diffloss_params = count_parameters(mar_model.diffloss.net)
            diffusion_params += diffloss_params
            print(f"  - Video DiffLoss Net: {diffloss_params:,} ({format_number(diffloss_params)})")
    
    # Wrist video diffusion loss
    if hasattr(mar_model, 'diffloss_wrist') and mar_model.diffloss_wrist is not None:
        if hasattr(mar_model.diffloss_wrist, 'net'):
            diffloss_wrist_params = count_parameters(mar_model.diffloss_wrist.net)
            diffusion_params += diffloss_wrist_params
            print(f"  - Wrist Video DiffLoss Net: {diffloss_wrist_params:,} ({format_number(diffloss_wrist_params)})")
    
    # Action diffusion loss
    if hasattr(mar_model, 'diffactloss') and mar_model.diffactloss is not None:
        # DiffActLoss可能包含多个网络组件
        diffactloss_params = 0
        
        # 检查是否有net属性（类似DiffLoss）
        if hasattr(mar_model.diffactloss, 'net'):
            net_params = count_parameters(mar_model.diffactloss.net)
            diffactloss_params += net_params
            print(f"    - Action DiffLoss Net: {net_params:,} ({format_number(net_params)})")
        
        # 检查是否有conv、fc等组件（conv_fc类型）
        if hasattr(mar_model.diffactloss, 'conv'):
            conv_params = count_parameters(mar_model.diffactloss.conv)
            diffactloss_params += conv_params
            print(f"    - Action Conv: {conv_params:,} ({format_number(conv_params)})")
        
        if hasattr(mar_model.diffactloss, 'fc'):
            fc_params = count_parameters(mar_model.diffactloss.fc)
            diffactloss_params += fc_params
            print(f"    - Action FC: {fc_params:,} ({format_number(fc_params)})")
        
        if hasattr(mar_model.diffactloss, 'interpolate'):
            interpolate_params = count_parameters(mar_model.diffactloss.interpolate)
            diffactloss_params += interpolate_params
            print(f"    - Action Interpolate: {interpolate_params:,} ({format_number(interpolate_params)})")
        
        if hasattr(mar_model.diffactloss, 'refine'):
            refine_params = count_parameters(mar_model.diffactloss.refine)
            diffactloss_params += refine_params
            print(f"    - Action Refine: {refine_params:,} ({format_number(refine_params)})")
        
        if hasattr(mar_model.diffactloss, 'ucgmts'):
            ucgmts_params = count_parameters(mar_model.diffactloss.ucgmts)
            diffactloss_params += ucgmts_params
            print(f"    - UCGMTS: {ucgmts_params:,} ({format_number(ucgmts_params)})")
        
        diffusion_params += diffactloss_params
        print(f"  - Action DiffLoss Total: {diffactloss_params:,} ({format_number(diffactloss_params)})")
    
    # Proprioception diffusion loss
    if hasattr(mar_model, 'diffproploss') and mar_model.diffproploss is not None:
        diffproploss_params = count_parameters(mar_model.diffproploss)
        diffusion_params += diffproploss_params
        print(f"  - Proprioception DiffLoss: {diffproploss_params:,} ({format_number(diffproploss_params)})")
    
    results["diffusion_head"] = diffusion_params
    print(f"Diffusion Head总参数量: {diffusion_params:,} ({format_number(diffusion_params)})")
    
    # 4. 其他部分（如果有）
    total_mar_params = count_parameters(mar_model)
    other_params = total_mar_params - transformer_params - diffusion_params
    results["other"] = other_params
    
    # 总参数量
    if hasattr(policy_model, 'vae_model') and policy_model.vae_model is not None:
        total_params = results["vae"] + total_mar_params
    else:
        total_params = total_mar_params
    
    results["total"] = total_params
    
    return results


def print_summary(results):
    """打印参数量对比总结"""
    print("\n" + "="*70)
    print("参数量统计总结")
    print("="*70)
    
    vae = results["vae"]
    transformer = results["transformer"]
    diffusion = results["diffusion_head"]
    other = results["other"]
    total = results["total"]
    
    print(f"\n{'部分':<25} {'参数量':<25} {'百分比':<15}")
    print("-" * 70)
    
    if vae > 0:
        vae_pct = (vae / total) * 100
        print(f"{'VAE':<25} {vae:>20,} ({format_number(vae):>8}) {vae_pct:>6.2f}%")
    
    transformer_pct = (transformer / total) * 100
    print(f"{'Transformer':<25} {transformer:>20,} ({format_number(transformer):>8}) {transformer_pct:>6.2f}%")
    
    diffusion_pct = (diffusion / total) * 100
    print(f"{'Diffusion Head':<25} {diffusion:>20,} ({format_number(diffusion):>8}) {diffusion_pct:>6.2f}%")
    
    if other > 0:
        other_pct = (other / total) * 100
        print(f"{'其他':<25} {other:>20,} ({format_number(other):>8}) {other_pct:>6.2f}%")
    
    print("-" * 70)
    print(f"{'总计':<25} {total:>20,} ({format_number(total):>8}) {'100.00%':>15}")
    print("="*70)
    
    # 计算Transformer和Diffusion的比例（不包括VAE）
    mar_total = transformer + diffusion + other
    if mar_total > 0:
        print(f"\nMAR模型内部比例（不包括VAE）:")
        print(f"  {'Transformer:':<25} {transformer:>20,} ({format_number(transformer):>8}) {(transformer/mar_total)*100:>6.2f}%")
        print(f"  {'Diffusion Head:':<25} {diffusion:>20,} ({format_number(diffusion):>8}) {(diffusion/mar_total)*100:>6.2f}%")
        if other > 0:
            print(f"  {'其他:':<25} {other:>20,} ({format_number(other):>8}) {(other/mar_total)*100:>6.2f}%")
    
    # 打印直观对比
    print(f"\n直观对比:")
    if vae > 0 and transformer > 0:
        print(f"  VAE参数量是Transformer的 {vae/transformer:.2f}x")
    if transformer > 0 and diffusion > 0:
        print(f"  Transformer参数量是Diffusion Head的 {transformer/diffusion:.2f}x")
    if vae > 0 and diffusion > 0:
        print(f"  VAE参数量是Diffusion Head的 {vae/diffusion:.2f}x")


@hydra.main(version_base=None, config_path="unified_video_action/config", config_name="uva_pusht")
def main(cfg):
    """
    主函数
    
    注意：此脚本直接从代码和配置文件实例化模型来统计参数量，
    不需要加载checkpoint权重。参数量只取决于模型结构，与权重值无关。
    """
    import os
    # 设置环境变量以获取完整错误堆栈
    os.environ['HYDRA_FULL_ERROR'] = '1'
    
    print("="*60)
    print("模型参数量统计")
    print("="*60)
    print("\n说明：直接从代码和配置统计参数量，不需要checkpoint权重")
    print("="*60)
    
    # 实例化模型（不需要加载权重）
    print("\n正在从配置文件实例化模型...")
    try:
        language_emb_model = cfg.task.dataset.language_emb_model if hasattr(cfg.task.dataset, 'language_emb_model') else None
        
        # 确保task_modes是列表类型
        task_modes = cfg.task.task_modes if hasattr(cfg.task, 'task_modes') else []
        if task_modes is None:
            task_modes = []
        elif isinstance(task_modes, (int, float)):
            # 如果是数字，转换为空列表
            print(f"警告: task_modes为数字类型，将转换为空列表")
            task_modes = []
        else:
            # ListConfig, list, tuple等都转换为列表
            try:
                task_modes = list(task_modes)
            except (TypeError, ValueError):
                print(f"警告: 无法将task_modes转换为列表，使用空列表")
                task_modes = []
        
        # 确保normalizer_type存在
        normalizer_type = cfg.task.dataset.normalizer_type if hasattr(cfg.task.dataset, 'normalizer_type') else None
        
        policy_model: UnifiedVideoActionPolicy = hydra.utils.instantiate(
            cfg.model.policy,
            task_name=cfg.task.name,
            task_modes=task_modes,
            normalizer_type=normalizer_type,
            language_emb_model=language_emb_model,
        )
        
        print("模型结构加载完成！\n")
        
        # 分析参数量
        results = analyze_model_parameters(policy_model)
        
        # 打印总结
        print_summary(results)
        
        return results
    except Exception as e:
        print(f"\n错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

