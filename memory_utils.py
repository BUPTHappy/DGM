"""
Memory optimization utilities for UVA evaluation.
"""

import torch
import gc
import psutil
import os
from typing import Optional, Dict, Any


def print_memory_stats(stage: str = ""):
    """Print current memory usage statistics."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        
        print(f"\n{'='*50}")
        print(f"GPU Memory Stats {stage}")
        print(f"{'='*50}")
        print(f"Allocated: {allocated:.2f} GB")
        print(f"Reserved: {reserved:.2f} GB")
        print(f"Max Allocated: {max_allocated:.2f} GB")
        
        # Get GPU memory info
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / 1024**3
            print(f"GPU {i} Total: {total_memory:.2f} GB")
    
    # System memory
    memory = psutil.virtual_memory()
    print(f"System RAM: {memory.used/1024**3:.2f} GB / {memory.total/1024**3:.2f} GB ({memory.percent:.1f}%)")
    print(f"{'='*50}\n")


def cleanup_memory():
    """Aggressive memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def set_memory_optimization_env():
    """Set environment variables for memory optimization."""
    # Disable expandable segments to avoid internal assertion failures
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    # Additional memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,garbage_collection_threshold:0.6"


def enable_gradient_checkpointing(model):
    """Enable gradient checkpointing for the model if supported."""
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("✓ Enabled gradient checkpointing")
    elif hasattr(model, 'enable_gradient_checkpointing'):
        model.enable_gradient_checkpointing()
        print("✓ Enabled gradient checkpointing")
    else:
        print("⚠ Gradient checkpointing not supported for this model")


def optimize_model_for_eval(model):
    """Optimize model for evaluation (reduce memory usage)."""
    model.eval()
    
    # Enable gradient checkpointing if available
    enable_gradient_checkpointing(model)
    
    # Set to half precision if supported
    if hasattr(model, 'half'):
        try:
            model = model.half()
            print("✓ Converted model to half precision")
        except Exception as e:
            print(f"⚠ Could not convert to half precision: {e}")
    
    return model


def get_optimal_chunk_size(available_memory_gb: float, sequence_length: int) -> int:
    """
    Calculate optimal chunk size based on available memory.
    
    Args:
        available_memory_gb: Available GPU memory in GB
        sequence_length: Length of the sequence to process
    
    Returns:
        Optimal chunk size
    """
    # Conservative estimation: each frame needs ~0.5GB
    estimated_memory_per_frame = 0.5
    
    # Calculate max frames we can process at once
    max_frames = int(available_memory_gb / estimated_memory_per_frame)
    
    # Don't exceed sequence length
    optimal_chunk_size = min(max_frames, sequence_length)
    
    # Ensure at least 1 frame
    optimal_chunk_size = max(1, optimal_chunk_size)
    
    return optimal_chunk_size


class MemoryMonitor:
    """Context manager for monitoring memory usage."""
    
    def __init__(self, stage_name: str):
        self.stage_name = stage_name
    
    def __enter__(self):
        print(f"\nStarting: {self.stage_name}")
        print_memory_stats(f"Before {self.stage_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        cleanup_memory()
        print_memory_stats(f"After {self.stage_name}")
        if exc_type is not None:
            print(f"Error in {self.stage_name}: {exc_val}")
