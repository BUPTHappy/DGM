#!/bin/bash

# Test script to verify imports and setup
# 测试脚本验证导入和设置

set -e

# Change to project root directory
cd "$(dirname "$0")/../.."

echo "=========================================="
echo "Testing Import and Setup"
echo "=========================================="
echo "Working directory: $(pwd)"

# Test Python imports
echo "Testing Python imports..."
python -c "
import sys
import os
sys.path.insert(0, os.getcwd())

try:
    from unified_video_action.dataset import get_dataset
    print('✓ get_dataset import successful')
except ImportError as e:
    print(f'✗ get_dataset import failed: {e}')

try:
    from unified_video_action.model.autoregressive.mar_con_unified import MAR
    print('✓ MAR import successful')
except ImportError as e:
    print(f'✗ MAR import failed: {e}')

try:
    from unified_video_action.model.autoregressive.diffusion_action_loss import DiffActLoss
    print('✓ DiffActLoss import successful')
except ImportError as e:
    print(f'✗ DiffActLoss import failed: {e}')

try:
    from unified_video_action.model.autoregressive.cross_attention_diffusion import CrossAttentionAdaLN
    print('✓ CrossAttentionAdaLN import successful')
except ImportError as e:
    print(f'✗ CrossAttentionAdaLN import failed: {e}')

print('Import test completed!')
"

# Test GPU availability
echo ""
echo "Testing GPU availability..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('No CUDA GPUs available')
"

# Test checkpoint file
echo ""
echo "Testing checkpoint file..."
if [ -f "checkpoints/pusht.ckpt" ]; then
    echo "✓ checkpoints/pusht.ckpt exists"
    ls -lh checkpoints/pusht.ckpt
else
    echo "✗ checkpoints/pusht.ckpt not found"
    echo "Available checkpoints:"
    ls -la checkpoints/ || echo "No checkpoints directory"
fi

# Test dataset
echo ""
echo "Testing dataset..."
if [ -d "data/pusht/" ]; then
    echo "✓ data/pusht/ directory exists"
    ls -la data/pusht/ | head -5
else
    echo "✗ data/pusht/ directory not found"
    echo "Available data directories:"
    ls -la data/ || echo "No data directory"
fi

echo ""
echo "=========================================="
echo "Setup test completed!"
echo "=========================================="

