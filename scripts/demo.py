# scripts/demo.py
import argparse
import time
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import TeachMyAgent.environments
from utils.env_utils import build_and_setup_env, collect_env_params, setup_render_window
from utils.shared_args import add_common_args, add_environment_args, add_render_args

def add_demo_args(parser):
    parser = add_common_args(parser)
    parser = add_environment_args(parser)
    parser = add_render_args(parser)
    parser.add_argument('--steps', type=int, default=2000, help="Number of steps to run in the demo.")
    return parser

def main(args):
    print(f"--- Demo Configuration ---\nEnvironment: {args.env}\nBody: {args.body}\n---------------------------")
    env = None
    try:
        user_params = collect_env_params(args.env, args)
        env = build_and_setup_env(args.env, args.body, user_params, render_mode="human")
        setup_render_window(env, args)
        obs, info = env.reset(seed=np.random.randint(1000))
        
        for step in range(args.steps):
            env.render()

            # Safe exit if window is closed
            if env.unwrapped.viewer and env.unwrapped.viewer.window and env.unwrapped.viewer.window.has_exit:
                print("Window closed. Exiting demo.")
                break

            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            time.sleep(1.0 / 60.0)

            if terminated or truncated:
                print(f"Episode ended after {step+1} steps. Resetting environment.")
                obs, info = env.reset()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    finally:
        if env:
            env.close()
            print("Environment closed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run an environment demo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser = add_demo_args(parser)
    args = parser.parse_args()
    main(args)
