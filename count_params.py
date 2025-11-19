#!/usr/bin/env python3
"""
简化版ckpt参数量统计脚本
"""

import torch
import sys
import os


def count_params(ckpt_path):
    """统计ckpt的参数量"""
    print(f"加载模型: {ckpt_path}")
    
    # 加载checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # 获取state_dict
    state_dicts = {}
    if isinstance(ckpt, dict):
        # 处理state_dicts结构（多个模型）
        if 'state_dicts' in ckpt:
            state_dicts = ckpt['state_dicts']
            print(f"\n找到 {len(state_dicts)} 个模型:")
            for key in state_dicts.keys():
                print(f"  - {key}")
        # 处理单个state_dict
        elif 'state_dict' in ckpt:
            state_dicts = {'model': ckpt['state_dict']}
        elif 'model' in ckpt:
            state_dicts = {'model': ckpt['model']}
        else:
            state_dicts = {'model': ckpt}
    else:
        state_dicts = {'model': ckpt}
    
    # 统计每个模型的参数
    print("\n" + "=" * 60)
    total_all = 0
    for model_name, state_dict in state_dicts.items():
        total = 0
        for name, param in state_dict.items():
            if hasattr(param, 'numel'):
                total += param.numel()
        
        total_all += total
        
        # 输出每个模型的参数量
        print(f"\n{model_name}:")
        print(f"  参数量: {total:,}")
        if total >= 1e9:
            print(f"  等于: {total/1e9:.2f}B")
        elif total >= 1e6:
            print(f"  等于: {total/1e6:.2f}M")
    
    # 输出总计
    print("\n" + "=" * 60)
    print(f"总参数量: {total_all:,}")
    if total_all >= 1e9:
        print(f"等于: {total_all/1e9:.2f}B")
    elif total_all >= 1e6:
        print(f"等于: {total_all/1e6:.2f}M")
    
    # 文件大小
    size_gb = os.path.getsize(ckpt_path) / (1024**3)
    print(f"文件大小: {size_gb:.2f} GB")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python count_params.py model.ckpt")
        sys.exit(1)
    
    count_params(sys.argv[1])