# ===== .\scripts\test_suite.py =====
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
from utils.env_utils import build_and_setup_env, setup_render_window 
from preprocessing import convert_weight
from scripts import train_ppo, train_acl, train_marl
# START CHANGE: Remove import of deleted function
from run import ray_init_and_run, load_config
# END CHANGE

def print_header(text, char='='):
    width = 80
    print("\n" + char * width)
    print(f"{text:^{width}}")
    print(char * width)

def run_test(func, test_name, args_namespace=None):
    print_header(f"RUNNING TEST: {test_name.upper()}", char='-')
    try:
        is_no_args_func = 'args_namespace' not in func.__code__.co_varnames and func.__code__.co_argcount == 0
        result = func() if is_no_args_func else func(args_namespace)
        print(f"✅ SUCCESS: {test_name} completed.")
        return result, True
    except Exception:
        print(f"❌ FAILURE: {test_name} failed.")
        import traceback
        traceback.print_exc()
        return None, False

# --- Sanity check functions remain the same ---
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
    env_config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'environments', 'parkour_default.yaml')
    env_config = load_config(env_config_path)
    env_args = SimpleNamespace(**env_config, **env_config.get('env_params', {}))
    env_args.horizon = 100

    build_and_setup_env(env_args.env, env_args.body, {}, args=env_args).close()
    
    custom_params = {"input_vector": np.array([0.1, -0.2, 0.3]), "water_level": 0.5, "creepers_height": 0.0}
    env = build_and_setup_env(env_args.env, env_args.body, custom_params, args=env_args)
    assert env.unwrapped.creepers_height == 0.0, "Creeper height not set correctly"
    env.close()
    
    print("Environment parameterization seems correct.")
    return True

def test_reward_function_sanity():
    """Runs a few steps and checks if the reward is a valid float."""
    print("Running reward function sanity check...")
    env_config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'environments', 'parkour_default.yaml')
    env_config = load_config(env_config_path)
    env_args = SimpleNamespace(**env_config, **env_config.get('env_params', {}))
    env_args.horizon = 100

    env = build_and_setup_env(env_args.env, env_args.body, {}, args=env_args)
    env.reset()
    for _ in range(5):
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(action)
        is_float = isinstance(reward, float) or np.issubdtype(type(reward), np.floating)
        assert is_float, f"Reward is not a float, but {type(reward)}"
    env.close()
    print("Reward function returns valid float values.")
    return True

def test_rendering_popup(args):
    """
    Tests if the rendering window can open, run with random actions, and close.
    """
    print("Testing rendering window popup...")
    env = None
    try:
        env_config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'environments', 'parkour_default.yaml')
        env_config = load_config(env_config_path)

        render_args = SimpleNamespace(
            fullscreen=args.render_fullscreen, 
            width=None if args.render_fullscreen else 800, 
            height=None if args.render_fullscreen else 600
        )
        
        env_render_args = SimpleNamespace(**env_config, **env_config.get('env_params', {}), **vars(render_args))
        env_render_args.horizon = 200
        
        env = build_and_setup_env(env_render_args.env, env_render_args.body, {}, render_mode="human", args=env_render_args)
        setup_render_window(env, render_args)

        env.reset()
        if args.render_fullscreen:
            print("  - A fullscreen window should appear for ~4 seconds...")
        else:
            print(f"  - A {render_args.width}x{render_args.height} window should appear for ~4 seconds...")
        
        for i in range(200):
            action = env.action_space.sample()
            env.step(action)
            env.render()
            
            if env.unwrapped.viewer and env.unwrapped.viewer.window and env.unwrapped.viewer.window.has_exit:
                print("  - Window closed by user during test.")
                break
            time.sleep(1.0 / 50.0) 
        
        print("  - Rendering test loop completed without crashing.")
        return True
    finally:
        if env:
            env.close()
            print("  - Environment and rendering window closed successfully.")

# REMOVED: The test_dynamic_threshold_calculation function is no longer needed.

def main(args): 
    print_header("STARTING EXPANDED PROJECT TEST SUITE")
    failed_tests = []
    
    config_dir = os.path.join(os.path.dirname(__file__), '..', 'configs')
    env_config_path = os.path.join(config_dir, 'environments', 'parkour_default.yaml')
    ppo_config_path = os.path.join(config_dir, 'algorithms', 'ppo_default.yaml')
    acl_config_path = os.path.join(config_dir, 'algorithms', 'acl_default.yaml')
    marl_config_path = os.path.join(config_dir, 'algorithms', 'marl_default.yaml')
    
    env_config = load_config(env_config_path)
    ppo_config = {**env_config, **load_config(ppo_config_path), **env_config.get('env_params', {})}
    acl_config = {**env_config, **load_config(acl_config_path), **env_config.get('env_params', {})}
    marl_config = {**env_config, **load_config(marl_config_path), **env_config.get('env_params', {})}

    print_header("GROUP 1: SANITY CHECKS", char='*')
    
    if args.test_render:
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
        
    if not failed_tests:
        ppo_test_config = ppo_config.copy()
        ppo_test_config['total_timesteps'] = 512
        ppo_test_config['n_envs'] = 2
        if 'ppo_config' in ppo_test_config:
            ppo_test_config['ppo_config']['n_steps'] = 128
        
        ppo_test_config['run_id'] = "test_suite_ppo"
        ppo_test_config['render'] = False
        ppo_args = SimpleNamespace(**ppo_test_config)
        
        _, success = run_test(train_ppo.main, "PPO Micro-Training", args_namespace=ppo_args)
        if not success: failed_tests.append("PPO Micro-Training")
    
    # START CHANGE: Removed dynamic threshold test. Rewrote ACL test to be independent.
    if not failed_tests:
        acl_test_config = acl_config.copy()
        acl_test_config['total_stages'] = 1
        acl_test_config['student_steps_per_stage'] = 256
        acl_test_config['n_envs'] = 2
        
        acl_test_config['run_id'] = "test_suite_acl"
        # REMOVED: pretrained_model_path is no longer used
        # Set a fixed, low threshold for testing purposes
        acl_test_config['mastery_threshold'] = 200.0
        acl_test_config['render'] = False

        # Turn off dynamic threshold calculation for the test
        if 'dynamic_threshold_settings' in acl_test_config:
            acl_test_config['dynamic_threshold_settings']['enabled'] = False

        acl_args = SimpleNamespace(**acl_test_config)
        
        _, success = run_test(train_acl.main, "ACL Micro-Training (from scratch)", args_namespace=acl_args)
        if not success: failed_tests.append("ACL Micro-Training")
    # END CHANGE
    
    if not failed_tests:
        marl_test_config = marl_config.copy()
        marl_test_config['iterations'] = 2
        marl_test_config['run_id'] = "test_suite_marl"
        marl_test_config['num_workers'] = 0 
        marl_test_config['num_gpus'] = 0

        marl_args = SimpleNamespace(**marl_test_config)

        _, success = run_test(lambda: ray_init_and_run(train_marl.main, marl_args), "MARL Micro-Training")
        if not success:
            failed_tests.append("MARL Micro-Training")
    
    print_header("TEST SUITE SUMMARY")
    if not failed_tests:
        print("🎉🎉🎉 All tests passed successfully! 🎉🎉🎉")
    else:
        print("🔥🔥🔥 ERRORS DETECTED DURING TESTING. 🔥🔥🔥")
        print(f"Failed stages: {', '.join(failed_tests)}")
        sys.exit(1)