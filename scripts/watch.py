# scripts/watch.py
import argparse
import time
import os
import sys
import numpy as np
import gymnasium as gym
import zipfile
import tempfile

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.env_utils import build_and_setup_env, collect_env_params, setup_render_window
from utils.shared_args import add_common_args, add_environment_args, add_render_args

class UserInterrupt(Exception):
    pass

def add_watch_args(parser):
    parser.add_argument('--model_path', type=str, required=True, help="Path to trained model (.zip).")
    parser = add_common_args(parser)
    parser = add_environment_args(parser)
    parser = add_render_args(parser)
    watch_group = parser.add_argument_group('Playback Parameters')
    watch_group.add_argument('--num-episodes', type=int, default=10, help="Number of episodes to watch.")
    watch_group.add_argument('--timeout', type=int, default=5000, help="Max steps per episode.") # Tăng timeout
    watch_group.add_argument('--fast-forward', '-ff', action='store_true', help="Skip frame delays for fast playback.")
    return parser

def main(args):
    print(f"--- Model Playback ---\nModel: {args.model_path}\nEnv: {args.env}\nBody: {args.body}\n--------------------")

    if not os.path.exists(args.model_path):
        print(f"ERROR: Model not found at '{args.model_path}'")
        sys.exit(1)

    def env_constructor():
        env = build_and_setup_env(args.env, args.body, 
                                  collect_env_params(args.env, args), 
                                  render_mode="human", args=args)
        setup_render_window(env, args)
        return env

    venv = None
    temp_dir = None
    try:
        venv = make_vec_env(env_constructor, n_envs=1)

        with zipfile.ZipFile(args.model_path, 'r') as archive:
            if 'vecnormalize.pkl' in archive.namelist():
                print("Found VecNormalize stats inside model.zip. Extracting...")
                # Tạo một thư mục tạm để giải nén
                temp_dir = tempfile.mkdtemp()
                archive.extract('vecnormalize.pkl', path=temp_dir)
                stats_path = os.path.join(temp_dir, 'vecnormalize.pkl')

                # 4. Load thống kê và bọc môi trường venv
                venv = VecNormalize.load(stats_path, venv)
                venv.training = False  # Đặt ở chế độ evaluation
                venv.norm_reward = False
                print("Successfully loaded and applied VecNormalize stats.")
            else:
                print("WARNING: No VecNormalize stats found inside model.zip. The model might not have been trained with normalization.")

        print("Loading model...")
        model = PPO.load(args.model_path)
        print("Model loaded successfully.")
        
        for i in range(args.num_episodes):
            obs = venv.reset()
            done = False
            total_reward = 0
            ep_len = 0
            print(f"\n--- Starting Episode {i + 1}/{args.num_episodes} ---")

            while not done:
                venv.render()
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = venv.step(action)
                
                done = dones[0]
                if done:
                    total_reward = infos[0].get('episode', {}).get('r', 0)
                
                ep_len += 1

                if not args.fast_forward:
                    time.sleep(1.0 / 60.0)
                
                if not done and ep_len >= args.timeout:
                     print(f"  -> Timeout reached ({args.timeout} steps).")
                     done = True

            print(f"Episode {i + 1} finished. Reward: {total_reward:.2f}, Length: {ep_len}")

    except (KeyboardInterrupt, UserInterrupt):
        print("\nPlayback stopped by user.")
    except Exception as e:
        if "window" in str(e).lower() or "closed" in str(e).lower():
             print("\nPlayback window closed.")
        else:
             print(f"An error occurred: {e}")
             import traceback
             traceback.print_exc()
    finally:
        if venv:
            venv.close()
            print("Environment closed.")
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Replay a trained agent.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = add_watch_args(parser)
    args = parser.parse_args()
    main(args)