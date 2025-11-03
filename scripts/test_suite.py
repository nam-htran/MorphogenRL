# scripts/test_suite.py
import argparse
import os
import sys
import yaml
from types import SimpleNamespace

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing import convert_weight
from scripts import demo as demo_script
from scripts import train_ppo
from scripts import train_acl
from scripts import train_marl
from run import ray_init_and_run # Import the wrapper for Ray

def print_header(text, char='='):
    width = 80
    print("\n" + char * width)
    print(f"{text:^{width}}")
    print(char * width)

def run_test(func, args_namespace, test_name):
    print_header(f"RUNNING TEST: {test_name.upper()}", char='-')
    if args_namespace:
        # Print args but hide potentially long lists/dicts for cleaner logs
        clean_args = {k: v if not isinstance(v, (dict, list)) else f"<{type(v).__name__}>" for k, v in vars(args_namespace).items()}
        print(f"Args: {clean_args}")
        
    try:
        if test_name.startswith("MARL"): # MARL requires Ray initialization
            result = ray_init_and_run(func, args_namespace)
        else:
            result = func(args_namespace) if args_namespace else func()
        print(f"✅ SUCCESS: {test_name} completed.")
        return result, True
    except KeyboardInterrupt:
        print(f"❌ FAILURE: Test '{test_name}' was interrupted by the user.")
        return None, False
    except Exception:
        print(f"❌ FAILURE: {test_name} failed.")
        import traceback
        traceback.print_exc()
        return None, False

def load_main_config():
    """Load main configuration from main_pipeline.yaml."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'main_pipeline.yaml')
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found at {config_path}")
        sys.exit(1)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(args=None):
    print_header("STARTING TRANSFER LEARNING PIPELINE TEST SUITE")
    
    main_config = load_main_config()
    failed_tests = []

    # START CHANGE: Define default parameters that might be missing from YAML
    default_test_params = {
        'render': False,
        'fullscreen': False,
        'width': None,
        'height': None,
        'seed': None, # Let the test run without a fixed seed unless specified
        'pretrained_model_path': None
    }
    # END CHANGE

    # --- Test 1: Prerequisite - Weight conversion ---
    _, success = run_test(convert_weight.convert_tf1_to_pytorch, None, "Prerequisite: Convert Weight")
    if not success:
        failed_tests.append("Convert Weight")
        
    # --- SEQUENTIAL TRANSFER LEARNING TEST ---
    ppo_model_path, acl_model_path = None, None
    
    # --- Step 1: PPO Pre-training (Micro-run) ---
    if 'PPO' in main_config and not failed_tests:
        # START CHANGE: Combine defaults with loaded config
        ppo_config = {**default_test_params, **main_config['PPO']}
        # END CHANGE
        
        ppo_config.update({
            'run_id': "test_suite_transfer_ppo",
            'total_timesteps': 512, 
            'n_envs': 2,
            'save_freq': 1024,
        })
        ppo_args = SimpleNamespace(**ppo_config)
        ppo_model_path, success = run_test(train_ppo.main, ppo_args, "PPO Pre-training Stage")
        if not success:
            failed_tests.append("PPO Pre-training")

    # --- Step 2: ACL Generalization (Micro-run with Transfer) ---
    if 'ACL' in main_config and ppo_model_path and not failed_tests:
        # START CHANGE: Combine defaults with loaded config
        acl_config = {**default_test_params, **main_config['ACL']}
        # END CHANGE
        
        acl_config.update({
            'run_id': "test_suite_transfer_acl",
            'total_stages': 1,
            'student_steps_per_stage': 256,
            'n_envs': 2,
            'pretrained_model_path': ppo_model_path, 
        })
        if 'ppo_config' in acl_config and 'learning_starts' in acl_config['ppo_config']:
            acl_config['ppo_config']['learning_starts'] = 100

        acl_args = SimpleNamespace(**acl_config)
        acl_model_path, success = run_test(train_acl.main, acl_args, "ACL Transfer Stage")
        if not success:
            failed_tests.append("ACL Transfer")

    # --- Step 3: MARL Cooperation (Micro-run with Transfer) ---
    if 'MARL' in main_config and acl_model_path and not failed_tests:
        # START CHANGE: Combine defaults with loaded config
        marl_config = {**default_test_params, **main_config['MARL']}
        # END CHANGE
        
        marl_config.update({
            'run_id': "test_suite_transfer_marl",
            'iterations': 1,
            'num_workers': 1,
            'num_gpus': 0,
            'use_tune': False,
            'shared_policy': True,
            'pretrained_model_path': acl_model_path,
        })
        marl_args = SimpleNamespace(**marl_config)
        _, success = run_test(train_marl.main, marl_args, "MARL Transfer Stage")
        if not success:
            failed_tests.append("MARL Transfer")

    # --- Final Summary ---
    print_header("TEST SUITE SUMMARY")
    if not failed_tests:
        print("🎉🎉🎉 Full Transfer Learning Pipeline test passed successfully! 🎉🎉🎉")
    else:
        print("🔥🔥🔥 ERRORS DETECTED DURING TESTING. 🔥🔥🔥")
        print(f"Failed stages: {', '.join(failed_tests)}")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the project's integrated test suite.")
    main(None)