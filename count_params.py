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
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt
    
    # 统计参数
    total = 0
    for name, param in state_dict.items():
        if hasattr(param, 'numel'):
            total += param.numel()
    
    # 输出结果
    print(f"\n总参数量: {total:,}")
    
    # 转换为M/B
    if total >= 1e9:
        print(f"等于: {total/1e9:.2f}B")
    elif total >= 1e6:
        print(f"等于: {total/1e6:.2f}M")
    
    # 文件大小
    size_gb = os.path.getsize(ckpt_path) / (1024**3)
    print(f"文件大小: {size_gb:.2f} GB")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python count_params.py model.ckpt")
        sys.exit(1)
    
    count_params(sys.argv[1])