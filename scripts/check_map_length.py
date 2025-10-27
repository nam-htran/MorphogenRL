# scripts/check_map_length.py
import argparse
import os
import sys
import time
import numpy as np
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import load_config
from utils.env_utils import build_and_setup_env, collect_env_params
from stable_baselines3 import PPO

def print_header(title):
    """Prints a standardized header."""
    width = 80
    print("\n" + "#"*width + f"\n##{title:^76}##\n" + "#"*width)

def add_check_length_args(parser):
    """Adds CLI arguments for the map length check."""
    parser.add_argument('--config', type=str, default='configs/main_pipeline.yaml', help="Path to the main pipeline configuration file.")
    parser.add_argument('--stage', type=str, required=True, choices=['PPO', 'ACL', 'MARL'], help="Which stage configuration to use for the test (e.g., PPO).")
    parser.add_argument('--model_path', type=str, default=None, help="(Optional) Path to a trained SB3 model (.zip) to run instead of the heuristic.")
    parser.add_argument('--timeout', type=int, default=10000, help="Maximum number of steps for this check before stopping.")
    parser.add_argument('--no-render', action='store_true', help="Run without rendering the environment.")
    parser.add_argument('--fast', action='store_true', help="Run rendering as fast as possible without delay.")
    return parser

def main(args):
    """
    Runs an intelligent exploratory agent to estimate the number of steps 
    required to finish a map based on a pipeline configuration.
    """
    print_header(f"CHECKING MAP LENGTH FOR STAGE: {args.stage}")

    # --- Configuration Loading ---
    config = load_config(args.config)
    stage_config = config.get(args.stage)

    if not stage_config:
        print(f"ERROR: Stage '{args.stage}' not found in '{args.config}'")
        sys.exit(1)
        
    stage_args = SimpleNamespace(**stage_config)
    
    stage_args.horizon = args.timeout 
    stage_args.render = not args.no_render
    stage_args.width = 1280
    stage_args.height = 720
    stage_args.fullscreen = False

    print("Building environment with parameters from stage '{}'".format(args.stage))
    user_params = collect_env_params(stage_args.env, stage_args)
    env = build_and_setup_env(stage_args.env, stage_args.body, user_params, 
                              render_mode="human" if stage_args.render else None, 
                              args=stage_args)

    # --- Agent Setup ---
    model = None
    if args.model_path:
        if os.path.exists(args.model_path):
            print(f"\nLoading model from: {args.model_path}")
            model = PPO.load(args.model_path)
        else:
            print(f"WARNING: Model not found at '{args.model_path}'. Using smart heuristic agent.")

    if model is None:
        print("\nUsing a smart heuristic agent to explore the map.")
        action_size = env.action_space.shape[0]
        
        # Define multiple strategies to move forward for classic_bipedal
        CANDIDATE_ACTIONS = [
            np.array([0.2, 0.1, -0.2, 0.1]),   # Strategy 1: Standard walk
            np.array([0.3, 0.0, -0.1, 0.0]),   # Strategy 2: Leaning forward
            np.array([-0.1, 0.3, 0.1, 0.3]),  # Strategy 3: High knee walk
            lambda: env.action_space.sample() # Strategy 4: Random action to get unstuck
        ]
        if action_size < 4: # Fallback for other bodies
            CANDIDATE_ACTIONS = [lambda: env.action_space.sample()]

    # --- Simulation Loop ---
    obs, info = env.reset()
    
    step = 0
    total_reward = 0
    done = False
    
    # Variables for smart heuristic
    current_action_index = 0
    last_check_step = 0
    last_check_pos_x = 0
    max_x_pos = -float('inf')
    steps_at_max_x = 0
    
    # Constants for stuck detection
    STUCK_CHECK_INTERVAL = 150
    STUCK_PROGRESS_THRESHOLD = 0.1 # meters

    print("\nStarting simulation... Press Ctrl+C to stop.")
    try:
        while not done:
            if stage_args.render:
                env.render()
                if not args.fast:
                    time.sleep(1./60.)

            # --- Action Selection ---
            if model:
                act, _ = model.predict(obs, deterministic=True)
            else:
                # Use the smart heuristic agent
                action_strategy = CANDIDATE_ACTIONS[current_action_index]
                act = action_strategy() if callable(action_strategy) else action_strategy

            # --- Environment Step ---
            obs, reward, terminated, truncated, info = env.step(act)
            done = terminated or truncated
            total_reward += reward
            step += 1
            
            # --- Progress Tracking for Heuristic Agent ---
            current_pos_x = env.unwrapped.agent_body.reference_head_object.position.x
            if current_pos_x > max_x_pos:
                max_x_pos = current_pos_x
                steps_at_max_x = step

            if not model and (step - last_check_step > STUCK_CHECK_INTERVAL):
                progress = current_pos_x - last_check_pos_x
                if progress < STUCK_PROGRESS_THRESHOLD:
                    print(f"  -> Agent stuck at step {step} (progress: {progress:.2f}m). Changing strategy...")
                    current_action_index = (current_action_index + 1) % len(CANDIDATE_ACTIONS)
                
                last_check_step = step
                last_check_pos_x = current_pos_x

            if step >= args.timeout:
                print("TIMEOUT: Reached maximum steps defined for the check.")
                break

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        env.close()

    # --- Final Report ---
    print_header("CHECK COMPLETE")
    print(f"Total Simulation Steps: {step}")
    print(f"Total Reward: {total_reward:.2f}")

    print("\nPerformance Metrics:")
    print(f"  - Maximum X-Position Reached: {max_x_pos:.2f} meters")
    print(f"  - Steps to Reach Max Position: {steps_at_max_x}")
    
    if truncated and not terminated:
        print("\nResult: Episode was TRUNCATED (hit the timeout). The map is likely longer.")
    elif terminated:
        print("\nResult: Episode was TERMINATED (agent fell or reached the goal).")
    else:
        print("\nResult: Simulation stopped manually.")

    print("\nRECOMMENDATION:")
    print(f"Based on this run, a good starting 'horizon' value is around {int(steps_at_max_x * 1.2)}.")
    print("(This is based on steps to reach the furthest point + 20% buffer).")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Check the approximate number of steps to complete a map.")
    parser = add_check_length_args(parser)
    args = parser.parse_args()
    main(args)