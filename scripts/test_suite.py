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
        result = func(args_namespace) if args_namespace else func()
        print(f"✅ SUCCESS: {test_name} completed.")
        return result, True
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
    print_header("STARTING FULL PIPELINE TEST")
    overall_success = True
    
    main_config = load_main_config()

    # Test 1: Weight conversion
    _, success = run_test(convert_weight.convert_tf1_to_pytorch, None, "Convert Weight")
    if not success:
        overall_success = False

    # Test 2: Demo environment
    demo_args = SimpleNamespace(
        env='parkour', body='classic_bipedal', steps=50, 
        # START CHANGE: Set fullscreen to True
        fullscreen=True, 
        # END CHANGE
        width=None, height=None, # width/height will be ignored if fullscreen is True
        roughness=None, stump_height=None, stump_width=None, obstacle_spacing=None,
        input_vector=None, water_level=None, creepers_width=None, creepers_height=None, creepers_spacing=None
    )
    _, success = run_test(demo_script.main, demo_args, "Demo Parkour Env")
    if not success:
        overall_success = False

    # Test 3: PPO training (short) and playback
    ppo_config = main_config.get('PPO', {})
    ppo_config.update({
        'run_id': "test_suite_ppo",
        'total_timesteps': 256,
        'save_freq': 256,
        'n_envs': 2,
        'render': False,
        'fullscreen': False,
        'width': None,
        'height': None
    })
    ppo_args = SimpleNamespace(**ppo_config)
    
    model_path, success = run_test(train_ppo.main, ppo_args, "Train PPO (short)")
    if not success:
        overall_success = False
    elif model_path:
        watch_args = SimpleNamespace(
            model_path=model_path,
            framework='sb3',
            num_episodes=1,
            env=ppo_args.env,
            body=ppo_args.body,
            # START CHANGE: Set fullscreen to True
            fullscreen=True,
            # END CHANGE
            width=None, height=None,
            timeout=100,
            fast_forward=True,
            roughness=None, stump_height=None, stump_width=None, obstacle_spacing=None,
            input_vector=None, water_level=None, creepers_width=None, creepers_height=None, creepers_spacing=None
        )
        _, watch_success = run_test(watch.main, watch_args, "Watch Trained PPO Model")
        if not watch_success:
            overall_success = False

    # Test 4: ACL training (short)
    acl_config = main_config.get('ACL', {})
    acl_config.update({
        'run_id': "test_suite_acl",
        'total_stages': 2,
        'student_steps_per_stage': 128,
        'eval_episodes': 1,
        'mastery_threshold': 50.0,
        'n_envs': 2,
        'render': False,
        'fullscreen': False,
        'width': None,
        'height': None
    })
    acl_args = SimpleNamespace(**acl_config)
    _, success = run_test(train_acl.main, acl_args, "Train ACL (short)")
    if not success:
        overall_success = False

    # Test 5: MARL training (short) and playback
    marl_config = main_config.get('MARL', {})
    marl_config.update({
        'run_id': "test_suite_marl",
        'iterations': 2,
        'num_workers': 1,
        'num_gpus': 0,
        'check_env': False,
        'width': None,
        'height': None,
        'fullscreen': False,
        'use_tune': False
    })
    marl_args = SimpleNamespace(**marl_config)
    
    marl_checkpoint_path, success = run_test(train_marl.main, marl_args, "Train MARL (short)")
    if not success:
        overall_success = False
    elif marl_checkpoint_path:
        watch_marl_args = SimpleNamespace(
            model_path=marl_checkpoint_path,
            framework='rllib',
            num_episodes=1,
            timeout=100,
            fast_forward=True,
            mode=marl_args.mode,
            n_agents=marl_args.n_agents,
            body=marl_args.body,
            # START CHANGE: Set fullscreen to True
            fullscreen=True,
            # END CHANGE
            width=None, height=None,
            env='parkour', roughness=None, stump_height=None, stump_width=None, 
            obstacle_spacing=None, input_vector=None, water_level=None, 
            creepers_width=None, creepers_height=None, creepers_spacing=None
        )
        _, watch_marl_success = run_test(watch.main, watch_marl_args, "Watch Trained MARL Model")
        if not watch_marl_success:
            overall_success = False

    print_header("TEST SUITE SUMMARY")
    if overall_success:
        print("🎉🎉🎉 All pipeline tests passed successfully! 🎉🎉🎉")
    else:
        print("🔥🔥🔥 ERRORS DETECTED DURING TESTING. 🔥🔥🔥")
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the project's integrated test suite.")
    main(None)