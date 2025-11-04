import argparse
import os
import sys
import yaml
from types import SimpleNamespace
import numpy as np
import time 

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TeachMyAgent.environments.envs.bodies.BodiesEnum import BodiesEnum
from utils.env_utils import build_and_setup_env, setup_render_window # Import setup_render_window
from preprocessing import convert_weight
from scripts import train_ppo, train_acl
from run import ray_init_and_run

def print_header(text, char='='):
    width = 80
    print("\n" + char * width)
    print(f"{text:^{width}}")
    print(char * width)

def run_test(func, test_name, args_namespace=None):
    print_header(f"RUNNING TEST: {test_name.upper()}", char='-')
    try:
        if args_namespace:
            result = func(args_namespace)
        else:
            result = func()
        print(f"✅ SUCCESS: {test_name} completed.")
        return result, True
    except Exception:
        print(f"❌ FAILURE: {test_name} failed.")
        import traceback
        traceback.print_exc()
        return None, False

# --- Test Functions ---
def test_body_creation():
    """Attempts to instantiate every body in BodiesEnum."""
    print("Testing instantiation of all agent bodies...")
    all_bodies = [e.name for e in BodiesEnum]
    created_bodies = []
    for body_name in all_bodies:
        try:
            body_class = BodiesEnum[body_name].value
            if "density" in body_class.__init__.__code__.co_varnames:
                 body_class(scale=30.0, density=1.0)
            else:
                 body_class(scale=30.0)
            created_bodies.append(body_name)
        except Exception as e:
            print(f"  - Failed to create '{body_name}': {e}")
            raise
    print(f"Successfully created {len(created_bodies)}/{len(all_bodies)} bodies.")
    return True

def test_env_parameterization():
    """Tests if the parkour environment can be parameterized correctly."""
    print("Testing parkour environment parameterization...")
    env_args = SimpleNamespace(horizon=100)
    build_and_setup_env('parkour', 'classic_bipedal', {}, args=env_args).close()
    
    custom_params = {"input_vector": np.array([0.1, -0.2, 0.3]), "water_level": 0.5, "creepers_height": 0.0}
    env = build_and_setup_env('parkour', 'classic_bipedal', custom_params, args=env_args)
    assert env.unwrapped.creepers_height == 0.0, "Creeper height not set correctly"
    env.close()
    
    print("Environment parameterization seems correct.")
    return True

def test_reward_function_sanity():
    """Runs a few steps and checks if the reward is a valid float."""
    print("Running reward function sanity check...")
    env_args = SimpleNamespace(horizon=100)
    env = build_and_setup_env('parkour', 'classic_bipedal', {}, args=env_args)
    env.reset()
    for _ in range(5):
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(action)
        is_float = isinstance(reward, float) or np.issubdtype(type(reward), np.floating)
        assert is_float, f"Reward is not a float, but {type(reward)}"
    env.close()
    print("Reward function returns valid float values.")
    return True

# START CHANGE: Updated rendering test to be more interactive and support fullscreen
def test_rendering_popup(args):
    """
    Tests if the rendering window can open, run with random actions, and close.
    """
    print("Testing rendering window popup...")
    env = None
    try:
        # Use args from the test_suite command to control rendering
        render_args = SimpleNamespace(
            fullscreen=args.render_fullscreen, 
            width=None if args.render_fullscreen else 800, # Use a larger default window
            height=None if args.render_fullscreen else 600,
            horizon=200 # Run for a bit longer
        )
        env = build_and_setup_env('parkour', 'classic_bipedal', {}, render_mode="human", args=render_args)
        
        # Use the utility to setup window size correctly
        setup_render_window(env, render_args)

        env.reset()
        if args.render_fullscreen:
            print("  - A fullscreen window should appear for ~4 seconds...")
        else:
            print(f"  - A {render_args.width}x{render_args.height} window should appear for ~4 seconds...")
        
        for i in range(render_args.horizon):
            # Take a random action
            action = env.action_space.sample()
            env.step(action)
            
            # Render the environment
            env.render()
            
            if env.unwrapped.viewer and env.unwrapped.viewer.window and env.unwrapped.viewer.window.has_exit:
                print("  - Window closed by user during test.")
                break
            time.sleep(1.0 / 50.0) # Limit to 50 FPS
        
        print("  - Rendering test loop completed without crashing.")
        return True
    finally:
        if env:
            env.close()
            print("  - Environment and rendering window closed successfully.")
# END CHANGE

def main(args): 
    print_header("STARTING EXPANDED PROJECT TEST SUITE")
    failed_tests = []
    
    print_header("GROUP 1: SANITY CHECKS", char='*')
    
    if args.test_render:
        # Pass the args object to the test function
        _, success = run_test(lambda: test_rendering_popup(args), "Rendering Window Popup")
        if not success: failed_tests.append("Rendering Window Popup")
    else:
        print("\nSkipping rendering test. Use the --test-render flag to run it.")

    _, success = run_test(test_body_creation, "Body Creation")
    if not success: failed_tests.append("Body Creation")
    
    _, success = run_test(test_env_parameterization, "Environment Parameterization")
    if not success: failed_tests.append("Environment Parameterization")

    _, success = run_test(test_reward_function_sanity, "Reward Function Sanity Check")
    if not success: failed_tests.append("Reward Function Sanity Check")

    print_header("GROUP 2: INTEGRATION TESTS", char='*')

    _, success = run_test(convert_weight.convert_tf1_to_pytorch, "Prerequisite: Convert Weight")
    if not success: failed_tests.append("Convert Weight")
        
    ppo_model_path = None
    
    if not failed_tests:
        ppo_args = SimpleNamespace(
            env='parkour', body='classic_bipedal', run_id="test_suite_ppo",
            total_timesteps=512, n_envs=2, save_freq=1024, seed=42, horizon=256,
            reward_shaping={'progress_multiplier': 1.0}, ppo_config={'n_steps': 128},
            render=False 
        )
        ppo_model_path, success = run_test(train_ppo.main, "PPO Micro-Training", args_namespace=ppo_args)
        if not success: failed_tests.append("PPO Micro-Training")

    if ppo_model_path and not failed_tests:
        acl_args = SimpleNamespace(
            env='parkour', body='classic_bipedal', run_id="test_suite_acl",
            total_stages=1, student_steps_per_stage=256, n_envs=2,
            pretrained_model_path=ppo_model_path, mastery_threshold=1000,
            eval_episodes=1, horizon=128, seed=42,
            difficulty_dims=['water_level'], initial_difficulty=[0.1], mastery_order=['water_level'],
            difficulty_increments={'water_level': 0.1},
            render=False
        )
        _, success = run_test(train_acl.main, "ACL Micro-Training (with Transfer)", args_namespace=acl_args)
        if not success: failed_tests.append("ACL Micro-Training")
    
    print_header("TEST SUITE SUMMARY")
    if not failed_tests:
        print("🎉🎉🎉 All tests passed successfully! 🎉🎉🎉")
    else:
        print("🔥🔥🔥 ERRORS DETECTED DURING TESTING. 🔥🔥🔥")
        print(f"Failed stages: {', '.join(failed_tests)}")
        sys.exit(1)