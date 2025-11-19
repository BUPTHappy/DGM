#!/usr/bin/env python3
"""
调试脚本：查看ckpt文件结构
"""

import torch
import sys


def inspect_ckpt(ckpt_path):
    """查看ckpt文件的详细结构"""
    print(f"加载: {ckpt_path}\n")
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    print("=" * 60)
    print("Checkpoint 类型:", type(ckpt))
    print("=" * 60)
    
    if isinstance(ckpt, dict):
        print("\n顶层键:")
        for key in ckpt.keys():
            print(f"  - {key}: {type(ckpt[key])}")
        
        # 检查常见的state_dict位置
        for key in ['state_dict', 'model', 'model_state_dict', 'ema_model']:
            if key in ckpt:
                print(f"\n'{key}' 的内容:")
                sd = ckpt[key]
                if isinstance(sd, dict):
                    print(f"  类型: dict")
                    print(f"  键的数量: {len(sd)}")
                    print(f"  前5个键:")
                    for i, k in enumerate(list(sd.keys())[:5]):
                        val = sd[k]
                        if hasattr(val, 'shape'):
                            print(f"    {k}: shape={val.shape}, numel={val.numel()}")
                        else:
                            print(f"    {k}: {type(val)}")
                else:
                    print(f"  类型: {type(sd)}")
                    if hasattr(sd, 'state_dict'):
                        print(f"  有 state_dict() 方法")
    else:
        print(f"\n不是字典类型，是: {type(ckpt)}")
        if hasattr(ckpt, 'state_dict'):
            print("有 state_dict() 方法")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python inspect_ckpt.py model.ckpt")
        sys.exit(1)
    
    inspect_ckpt(sys.argv[1])