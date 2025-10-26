# scripts/watch.py
import argparse
import time
import os
import sys
import numpy as np

# Add root folder to import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.env_utils import build_and_setup_env, collect_env_params, setup_render_window
from utils.shared_args import add_common_args, add_environment_args, add_render_args
from stable_baselines3 import PPO


class UserInterrupt(Exception):
    """Raised when the user manually stops playback."""
    pass


def add_watch_args(parser):
    """Add CLI arguments for model playback."""
    parser.add_argument('--model_path', type=str, required=True, help="Path to trained model (.zip).")

    parser = add_common_args(parser)
    parser = add_environment_args(parser)
    parser = add_render_args(parser)

    watch_group = parser.add_argument_group('Playback Parameters')
    watch_group.add_argument('--num_episodes', type=int, default=10, help="Number of episodes to watch.")
    watch_group.add_argument('--timeout', type=int, default=2000, help="Max steps per episode (safety timeout).")
    watch_group.add_argument('--auto-skip-stuck', action='store_true', help="Automatically skip if agent gets stuck.")
    watch_group.add_argument('--fast-forward', '-ff', action='store_true', help="Skip frame delays for fast playback.")
    watch_group.add_argument('--framework', type=str, default='sb3', choices=['sb3', 'rllib'], help="Model framework.")
    watch_group.add_argument('--n_agents', type=int, default=1, help="Number of agents (for future MARL support).")
    return parser


def main(args):
    """Load and replay a trained agent."""
    print(f"--- Model Playback ---\nModel: {args.model_path}\nEnv: {args.env}\nBody: {args.body}\n--------------------")

    if not os.path.exists(args.model_path):
        print(f"ERROR: Model not found at '{args.model_path}'")
        sys.exit(1)

    STUCK_CHECK_STEPS = 150
    STUCK_PROGRESS_THRESHOLD = 0.1

    env = None
    try:
        user_params = collect_env_params(args.env, args)
        env = build_and_setup_env(args.env, args.body, user_params, render_mode="human")
        setup_render_window(env, args)

        print("Loading model...")
        model = PPO.load(args.model_path)
        print("Model loaded successfully.")

        for i in range(args.num_episodes):
            obs, info = env.reset()
            done = False
            total_reward = 0
            ep_len = 0
            print(f"\n--- Starting Episode {i + 1}/{args.num_episodes} ---")

            last_check_pos_x = env.unwrapped.agent_body.reference_head_object.position.x
            last_check_step = 0

            while not done:
                env.render()
                viewer = env.unwrapped.viewer
                if viewer is None or viewer.window is None or viewer.window.has_exit:
                    print("Render window closed. Stopping playback.")
                    raise UserInterrupt

                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                total_reward += reward
                ep_len += 1

                if args.auto_skip_stuck and (ep_len - last_check_step > STUCK_CHECK_STEPS):
                    current_pos_x = env.unwrapped.agent_body.reference_head_object.position.x
                    progress = current_pos_x - last_check_pos_x
                    if progress < STUCK_PROGRESS_THRESHOLD:
                        print(f"  -> Agent stuck (moved {progress:.2f} in {STUCK_CHECK_STEPS} steps), skipping episode.")
                        done = True
                    last_check_pos_x = current_pos_x
                    last_check_step = ep_len

                if not done and ep_len >= args.timeout:
                    print(f"  -> Episode stopped after exceeding {args.timeout} steps (timeout).")
                    done = True

                if not args.fast_forward:
                    time.sleep(1.0 / 60.0)

            print(f"Episode {i + 1} finished. Reward: {total_reward:.2f}, Length: {ep_len}")

    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C).")
        raise UserInterrupt
    finally:
        if env:
            env.close()
            print("Environment closed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Replay a trained agent.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = add_watch_args(parser)
    args = parser.parse_args()

    try:
        main(args)
    except UserInterrupt:
        print("Playback stopped by user.")
        sys.exit(0)
