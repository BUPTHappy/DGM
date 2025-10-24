#!/usr/bin/env python3
"""
Test script to verify the checkpoint loading fix.
This script tests that UCGM parameters are properly loaded without being dropped.
"""

import os
import sys
import tempfile
import subprocess
import json
from pathlib import Path

def test_checkpoint_loading(checkpoint_path, device="cuda:0"):
    """
    Test checkpoint loading to verify UCGM parameters are not dropped.
    """
    print(f"Testing checkpoint loading with: {checkpoint_path}")
    print(f"Using device: {device}")
    
    # Create temp output directory
    temp_output_dir = tempfile.mkdtemp(prefix="test_checkpoint_loading_")
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
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        # Check if we see the "Dropped" messages
        if "[Resume Training] Dropped" in result.stdout:
            print("\n❌ PROBLEM FOUND: UCGM parameters are still being dropped!")
            print("This means the checkpoint loading fix didn't work properly.")
            return False
        else:
            print("\n✅ SUCCESS: No UCGM parameters were dropped!")
            print("Checkpoint loading is working correctly.")
            
            # Parse results if available
            eval_log_path = os.path.join(temp_output_dir, f'eval_log_{os.path.basename(checkpoint_path)}.json')
            if os.path.exists(eval_log_path):
                with open(eval_log_path, 'r') as f:
                    eval_results = json.load(f)
                
                if "test_mean_score" in eval_results:
                    score = eval_results["test_mean_score"]
                    print(f"Test score: {score:.4f}")
                    
                    if score > 0.5:  # Reasonable threshold
                        print("✅ Score looks reasonable - model is working properly!")
                    else:
                        print("⚠️  Score is low - there might still be issues")
            
            return True
        
    except subprocess.TimeoutExpired:
        print("Evaluation timeout")
        return False
    except Exception as e:
        print(f"Evaluation error: {e}")
        return False
    finally:
        # Cleanup
        subprocess.run(["rm", "-rf", temp_output_dir], check=False)

if __name__ == "__main__":
    # Test with the checkpoint from your logs
    checkpoint_path = "/home/zhuliu/workspace/Towards-Generalizable-Embodied-AI-via-Diffusion-Based-Robotic-Policies/checkpoints/pusht_final_5/checkpoints/epoch=0190-test_mean_score=0.978.ckpt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please update the checkpoint path in the script")
        sys.exit(1)
    
    success = test_checkpoint_loading(checkpoint_path)
    
    if success:
        print(f"\n🎉 Checkpoint loading fix is working!")
        print("UCGM parameters are no longer being dropped.")
        print("Baseline evaluation should now show reasonable scores.")
    else:
        print(f"\n❌ Checkpoint loading fix needs more work.")
        print("UCGM parameters are still being dropped.")
