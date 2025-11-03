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
from scripts import watch

def print_header(text, char='='):
    width = 80
    print("\n" + char * width)
    print(f"{text:^{width}}")
    print(char * width)

def run_test(func, args_namespace, test_name):
    print_header(f"RUNNING TEST: {test_name.upper()}", char='-')
    if args_namespace:
        print(f"Args: {vars(args_namespace)}")
    try:
        # START CHANGE: Handle Ray-based functions
        if test_name.startswith("Train MARL"):
            from run import ray_init_and_run
            result = ray_init_and_run(func, args_namespace)
        else:
            result = func(args_namespace) if args_namespace else func()
        # END CHANGE
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
    print_header("STARTING FULL PIPELINE TEST SUITE")
    overall_success = True
    
    main_config = load_main_config()
    failures = []

    # --- Test 1: Weight conversion ---
    _, success = run_test(convert_weight.convert_tf1_to_pytorch, None, "Convert Weight")
    if not success: overall_success = False; failures.append("Convert Weight")

    # --- Test 2: Demo environment ---
    # Use PPO config for demo environment settings if available
    demo_base_config = main_config.get('PPO', {})
    demo_args = SimpleNamespace(
        env=demo_base_config.get('env', 'parkour'), 
        body=demo_base_config.get('body', 'classic_bipedal'), 
        steps=50, 
        fullscreen=False, # Use windowed mode for automated tests
        width=800, height=600,
        roughness=None, stump_height=None, stump_width=None, obstacle_spacing=None,
        input_vector=None, water_level=None, creepers_width=None, creepers_height=None, creepers_spacing=None
    )
    _, success = run_test(demo_script.main, demo_args, "Demo Environment")
    if not success: overall_success = False; failures.append("Demo Environment")

    # --- Test 3: PPO training (short) and playback ---
    if 'PPO' in main_config:
        ppo_config = main_config['PPO'].copy()
        # Override for a quick test run
        ppo_config.update({
            'run_id': "test_suite_ppo",
            'total_timesteps': 256,
            'save_freq': 512, # Ensure it doesn't save mid-run
            'n_envs': 2,
            'render': False,
        })
        ppo_args = SimpleNamespace(**ppo_config)
        
        model_path, success = run_test(train_ppo.main, ppo_args, "Train PPO (short)")
        if not success:
            overall_success = False
            failures.append("Train PPO (short)")
        elif model_path:
            watch_args = SimpleNamespace(
                model_path=model_path,
                framework='sb3',
                num_episodes=1,
                env=ppo_args.env,
                body=ppo_args.body,
                fullscreen=False, width=800, height=600,
                timeout=100,
                fast_forward=True,
                roughness=None, stump_height=None, stump_width=None, obstacle_spacing=None,
                input_vector=None, water_level=None, creepers_width=None, creepers_height=None, creepers_spacing=None
            )
            _, watch_success = run_test(watch.main, watch_args, "Watch Trained PPO Model")
            if not watch_success: 
                overall_success = False
                failures.append("Watch Trained PPO Model")

    # --- Test 4: ACL training (short) ---
    if 'ACL' in main_config:
        acl_config = main_config['ACL'].copy()
        # Override for a quick test run
        acl_config.update({
            'run_id': "test_suite_acl",
            'total_stages': 2,
            'student_steps_per_stage': 128,
            'eval_episodes': 1,
            'n_envs': 2,
            'render': False,
            'learning_starts': 100 # For SAC, to start learning faster in tests
        })
        # If ppo_config exists, update learning_starts there
        if 'ppo_config' in acl_config:
            acl_config['ppo_config']['learning_starts'] = 100

        acl_args = SimpleNamespace(**acl_config)
        _, success = run_test(train_acl.main, acl_args, "Train ACL (short)")
        if not success: 
            overall_success = False
            failures.append("Train ACL (short)")

    # --- Test 5: MARL training (short) ---
    if 'MARL' in main_config:
        marl_config = main_config['MARL'].copy()
        # Override for a quick test run
        marl_config.update({
            'run_id': "test_suite_marl",
            'iterations': 2,
            'num_workers': 1,
            'num_gpus': 0,
            'use_tune': False,
        })
        marl_args = SimpleNamespace(**marl_config)
        
        _, success = run_test(train_marl.main, marl_args, "Train MARL (short)")
        if not success: 
            overall_success = False
            failures.append("Train MARL (short)")

    # --- Final Summary ---
    print_header("TEST SUITE SUMMARY")
    if overall_success:
        print("🎉🎉🎉 All tests passed successfully! 🎉🎉🎉")
    else:
        print("🔥🔥🔥 ERRORS DETECTED DURING TESTING. 🔥🔥🔥")
        print(f"Failed tests: {', '.join(failures)}")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the project's integrated test suite.")
    main(None)