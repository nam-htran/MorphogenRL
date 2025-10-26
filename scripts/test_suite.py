# scripts/test_suite.py
import argparse
import os
import sys
from types import SimpleNamespace

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing import convert_weight
from scripts import demo as demo_script
from scripts import train_ppo
from scripts import train_acl
from scripts import train_marl
from scripts import watch
from utils.shared_args import available_bodies

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

def main(args=None):
    print_header("STARTING FULL PIPELINE TEST")
    overall_success = True

    # Test 1: Weight conversion
    _, success = run_test(convert_weight.convert_tf1_to_pytorch, None, "Convert Weight")
    if not success: overall_success = False

    # Test 2: Demo environment
    demo_args = SimpleNamespace(
        env='parkour', body='classic_bipedal', steps=50, fullscreen=True, width=600, height=400,
        roughness=None, stump_height=None, stump_width=None, obstacle_spacing=None,
        input_vector=None, water_level=None, creepers_width=None, creepers_height=None, creepers_spacing=None
    )
    _, success = run_test(demo_script.main, demo_args, "Demo Parkour Env")
    if not success: overall_success = False

    # Test 3: PPO training (short) and playback
    ppo_args = SimpleNamespace(
        run_id="test_suite_ppo", total_timesteps=128, body='classic_bipedal', env='parkour',
        save_freq=128, n_envs=1, render=False, width=None, height=None, fullscreen=True,
        roughness=None, stump_height=None, stump_width=None, obstacle_spacing=None,
        input_vector=None, water_level=None, creepers_width=None, creepers_height=None, creepers_spacing=None
    )
    model_path, success = run_test(train_ppo.main, ppo_args, "Train PPO (short)")
    if not success:
        overall_success = False
    elif model_path:
        watch_args = SimpleNamespace(
            model_path=model_path, framework='sb3', num_episodes=1, n_agents=1,
            env='parkour', body='classic_bipedal', fullscreen=True, width=200, height=150,
            roughness=None, stump_height=None, stump_width=None, obstacle_spacing=None,
            input_vector=None, water_level=None, creepers_width=None, creepers_height=None, creepers_spacing=None,
            timeout=100, auto_skip_stuck=False, fast_forward=True
        )
        _, watch_success = run_test(watch.main, watch_args, "Watch Trained PPO Model")
        if not watch_success: overall_success = False

    # Test 4: ACL training (short)
    acl_args = SimpleNamespace(
        run_id="test_suite_acl", 
        body='climbing_profile_chimpanzee', 
        env='parkour',
        total_stages=2,
        student_steps_per_stage=64,
        eval_episodes=1,
        mastery_threshold=50.0,
        difficulty_increment=0.1,
        render=False, fullscreen=True,
        width=None, height=None, roughness=None, stump_height=None, stump_width=None,
        obstacle_spacing=None, input_vector=None, water_level=None, creepers_width=None,
        creepers_height=None, creepers_spacing=None
    )
    _, success = run_test(train_acl.main, acl_args, "Train ACL (short)")
    if not success: overall_success = False

    # Test 5: MARL training (short)
    marl_args = SimpleNamespace(
        run_id="test_suite_marl", mode='interactive', n_agents=2, body='classic_bipedal',
        iterations=2, num_workers=1, check_env=False, width=None, height=None, fullscreen=True
    )
    _, success = run_test(train_marl.main, marl_args, "Train MARL (short)")
    if not success: overall_success = False

    print_header("TEST SUITE SUMMARY")
    if overall_success:
        print("🎉🎉🎉 All pipeline tests passed successfully! 🎉🎉🎉")
    else:
        print("🔥🔥🔥 ERRORS DETECTED DURING TESTING. 🔥🔥🔥")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the project's integrated test suite.")
    main(None)
