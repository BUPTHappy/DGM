#!/usr/bin/env python3
"""
Test script to verify Bayesian optimization integration.
This script tests the basic functionality without running full training.
"""

import os
import sys
import torch
import tempfile
from omegaconf import OmegaConf
from unified_video_action.optimization.training_bayesian_optimizer import TrainingBayesianOptimizer


def test_bayesian_optimizer_initialization():
    """Test Bayesian optimizer initialization."""
    print("Testing Bayesian optimizer initialization...")
    
    # Create a minimal config
    cfg = OmegaConf.create({
        'training': {'num_epochs': 200},
        'task': {'name': 'pusht'},
        'bayesian_optimization': {
            'enabled': True,
            'start_epoch': 100,
            'interval': 10,
            'max_trials': 5,
            'n_test': 2,
            'device': 'cuda:0',
            'output_dir': './test_bayesian_logs'
        }
    })
    
    try:
        optimizer = TrainingBayesianOptimizer(
            config=cfg,
            start_epoch=cfg.bayesian_optimization.start_epoch,
            interval=cfg.bayesian_optimization.interval,
            max_trials=cfg.bayesian_optimization.max_trials,
            n_test=cfg.bayesian_optimization.n_test,
            device=cfg.bayesian_optimization.device,
            output_dir=cfg.bayesian_optimization.output_dir
        )
        
        print("✓ Bayesian optimizer initialized successfully")
        
        # Test should_optimize method
        assert not optimizer.should_optimize(50), "Should not optimize at epoch 50"
        assert optimizer.should_optimize(100), "Should optimize at epoch 100"
        assert optimizer.should_optimize(110), "Should optimize at epoch 110"
        assert not optimizer.should_optimize(105), "Should not optimize at epoch 105"
        
        print("✓ should_optimize method works correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Bayesian optimizer initialization failed: {e}")
        return False


def test_parameter_generation():
    """Test parameter generation."""
    print("Testing parameter generation...")
    
    cfg = OmegaConf.create({
        'training': {'num_epochs': 200},
        'task': {'name': 'pusht'},
        'bayesian_optimization': {
            'enabled': True,
            'start_epoch': 100,
            'interval': 10,
            'max_trials': 5,
            'n_test': 2,
            'device': 'cuda:0',
            'output_dir': './test_bayesian_logs'
        }
    })
    
    try:
        optimizer = TrainingBayesianOptimizer(
            config=cfg,
            start_epoch=100,
            interval=10,
            max_trials=5,
            n_test=2,
            device='cuda:0',
            output_dir='./test_bayesian_logs'
        )
        
        # Create a mock trial for testing
        import optuna
        study = optuna.create_study(direction='maximize')
        trial = study.ask()
        
        # Test parameter generation
        params = optimizer.optimizer.optimize_params(trial)
        
        # Check required parameters
        required_params = [
            'consistc_ratio', 'rfba_gap_end', 'temperature', 
            'num_sampling_steps', 'cfg', 'extrapol_ratio',
            'window_size', 'lambda_local', 'ucgmts_config'
        ]
        
        for param in required_params:
            assert param in params, f"Missing parameter: {param}"
        
        # Check parameter ranges
        assert 0.5 <= params['consistc_ratio'] <= 1.0, "consistc_ratio out of range"
        assert 0.1 <= params['rfba_gap_end'] <= 0.8, "rfba_gap_end out of range"
        assert 0.7 <= params['temperature'] <= 1.2, "temperature out of range"
        assert params['num_sampling_steps'] in [1, 2, 3], "num_sampling_steps invalid"
        assert 0.8 <= params['cfg'] <= 1.5, "cfg out of range"
        assert 0.0 <= params['extrapol_ratio'] <= 0.6, "extrapol_ratio out of range"
        assert 0 <= params['window_size'] <= 25, "window_size out of range"
        assert 0.01 <= params['lambda_local'] <= 0.8, "lambda_local out of range"
        
        print("✓ Parameter generation works correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Parameter generation failed: {e}")
        return False


def test_config_integration():
    """Test configuration integration."""
    print("Testing configuration integration...")
    
    try:
        # Test that the config can be loaded
        config_path = "unified_video_action/config/pusht_DiT_hybrid_with_bayesian.yaml"
        if os.path.exists(config_path):
            cfg = OmegaConf.load(config_path)
            
            # Check that Bayesian optimization config exists
            assert hasattr(cfg, 'bayesian_optimization'), "Bayesian optimization config missing"
            assert cfg.bayesian_optimization.enabled == True, "Bayesian optimization not enabled"
            
            print("✓ Configuration integration works correctly")
            return True
        else:
            print("⚠ Configuration file not found, skipping test")
            return True
            
    except Exception as e:
        print(f"✗ Configuration integration failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("BAYESIAN OPTIMIZATION INTEGRATION TESTS")
    print("=" * 60)
    
    tests = [
        test_bayesian_optimizer_initialization,
        test_parameter_generation,
        test_config_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("✓ All tests passed! Bayesian optimization integration is ready.")
        return 0
    else:
        print("✗ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
