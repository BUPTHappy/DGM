#!/usr/bin/env python3
"""
Test script to verify the simplified baseline evaluation.
This script tests the baseline evaluation without complex config creation.
"""

import os
import sys
import tempfile
import subprocess
import json
from pathlib import Path

def test_baseline_evaluation(checkpoint_path, device="cuda:0"):
    """
    Test baseline evaluation using eval_sim.py directly.
    This mimics what the simplified evaluate_model_with_params does.
    """
    print(f"Testing baseline evaluation with checkpoint: {checkpoint_path}")
    print(f"Using device: {device}")
    
    # Create temp output directory
    temp_output_dir = tempfile.mkdtemp(prefix="test_baseline_")
    print(f"Temp output dir: {temp_output_dir}")
    
    # Build command - use eval_sim.py with --use_ucgm, no additional parameters
    cmd = [
        "python", "eval_sim.py",
        "--checkpoint", checkpoint_path,
        "--output_dir", temp_output_dir,
        "--device", device,
        "--use_ucgm"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1] if ":" in device else "0"
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=300  # 5 minutes timeout
        )
        
        if result.returncode != 0:
            print(f"Evaluation failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return None
        
        # Parse results
        eval_log_path = os.path.join(temp_output_dir, f'eval_log_{os.path.basename(checkpoint_path)}.json')
        
        if not os.path.exists(eval_log_path):
            print(f"Eval log not found: {eval_log_path}")
            return None
        
        with open(eval_log_path, 'r') as f:
            eval_results = json.load(f)
        
        # Extract score
        if "test_mean_score" in eval_results:
            score = eval_results["test_mean_score"]
        elif "test/pusht_mean_score" in eval_results:
            score = eval_results["test/pusht_mean_score"]
        elif "test/libero10_mean_score" in eval_results:
            score = eval_results["test/libero10_mean_score"]
        else:
            score_keys = [k for k in eval_results.keys() if "score" in k.lower()]
            if score_keys:
                score = eval_results[score_keys[0]]
            else:
                print(f"No score found, available keys: {list(eval_results.keys())}")
                return None
        
        print(f"Baseline score: {score}")
        print(f"Available keys in eval results: {list(eval_results.keys())}")
        
        # Cleanup
        subprocess.run(["rm", "-rf", temp_output_dir], check=False)
        
        return float(score)
        
    except subprocess.TimeoutExpired:
        print("Evaluation timeout")
        return None
    except Exception as e:
        print(f"Evaluation error: {e}")
        return None

if __name__ == "__main__":
    # Test with the checkpoint from your logs
    checkpoint_path = "/home/zhuliu/workspace/Towards-Generalizable-Embodied-AI-via-Diffusion-Based-Robotic-Policies/checkpoints/pusht_final_5/checkpoints/latest.ckpt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please update the checkpoint path in the script")
        sys.exit(1)
    
    score = test_baseline_evaluation(checkpoint_path)
    
    if score is not None:
        print(f"\n✅ Baseline evaluation successful!")
        print(f"Score: {score:.4f}")
        print(f"This should be similar to the baseline score in your Bayesian optimization logs")
    else:
        print(f"\n❌ Baseline evaluation failed!")
        print("Check the error messages above for debugging")
