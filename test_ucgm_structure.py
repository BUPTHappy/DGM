#!/usr/bin/env python3
"""
Test script to verify the UCGM model structure fix.
This script tests that the model is initialized with UCGM structure to match the checkpoint.
"""

import os
import sys
import tempfile
import subprocess
import json
from pathlib import Path

def test_ucgm_model_structure(checkpoint_path, device="cuda:0"):
    """
    Test that model is initialized with UCGM structure to match checkpoint.
    """
    print(f"Testing UCGM model structure with: {checkpoint_path}")
    print(f"Using device: {device}")
    
    # Create temp output directory
    temp_output_dir = tempfile.mkdtemp(prefix="test_ucgm_structure_")
    print(f"Temp output dir: {temp_output_dir}")
    
    # Build command - use eval_sim.py with --use_ucgm
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
        
        # Check for specific success indicators
        success_indicators = [
            "Using UCGM mode",
            "Using diffloss:  <class 'unified_video_action.model.autoregressive.diffusion_loss_ucgm.DiffLossUCGM'>",
            "Using diffactloss:  <class 'unified_video_action.model.autoregressive.diffusion_action_loss_ucgm.DiffActLossUCGM'>",
            "UCGMTS config values:",
            "Loading model",
            "Loading ema_model"
        ]
        
        failed_indicators = [
            "Unexpected key(s) in state_dict",
            "Error(s) in loading state_dict",
            "RuntimeError"
        ]
        
        # Check for success indicators
        success_count = sum(1 for indicator in success_indicators if indicator in result.stdout)
        failed_count = sum(1 for indicator in failed_indicators if indicator in result.stdout)
        
        print(f"\nSuccess indicators found: {success_count}/{len(success_indicators)}")
        print(f"Failed indicators found: {failed_count}")
        
        if failed_count > 0:
            print("\n❌ PROBLEM FOUND: Model structure mismatch!")
            print("The model is not initialized with UCGM structure to match the checkpoint.")
            return False
        elif success_count >= 4:  # Most success indicators present
            print("\n✅ SUCCESS: Model structure matches checkpoint!")
            print("UCGM model is properly initialized and checkpoint loads successfully.")
            
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
        else:
            print("\n⚠️  UNCLEAR: Some indicators missing, but no clear errors.")
            print("The model might be working, but verification is incomplete.")
            return False
        
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
    
    success = test_ucgm_model_structure(checkpoint_path)
    
    if success:
        print(f"\n🎉 UCGM model structure fix is working!")
        print("Model is properly initialized with UCGM structure.")
        print("Checkpoint loading should now work without errors.")
    else:
        print(f"\n❌ UCGM model structure fix needs more work.")
        print("Model structure still doesn't match the checkpoint.")
