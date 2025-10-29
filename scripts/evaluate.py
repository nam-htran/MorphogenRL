# scripts/evaluate.py
import argparse
import os
import sys
import time
import numpy as np
import zipfile
import tempfile
import shutil
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from utils.env_utils import build_and_setup_env, collect_env_params, setup_render_window
from utils.shared_args import add_common_args, add_environment_args, add_render_args
from utils.seeding import set_seed

def add_evaluation_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Adds evaluation-specific arguments to the parser."""
    parser = add_common_args(parser)
    parser = add_environment_args(parser)
    parser = add_render_args(parser)
    
    eval_group = parser.add_argument_group('Evaluation Parameters')
    eval_group.add_argument('--model_path', type=str, required=True, help="Path to the trained model (.zip file).")
    eval_group.add_argument('--num_episodes', type=int, default=20, help="Number of episodes to run for evaluation.")
    eval_group.add_argument('--render', action='store_true', help="Render the environment during evaluation.")
    eval_group.add_argument('--horizon', type=int, default=5000, help="Max steps per episode.")
    
    # Note: A separate seed is used here for evaluation to be independent of training seed
    eval_group.add_argument('--eval_seed', type=int, default=42, help="Base seed for evaluation episodes.")
    return parser

def evaluate_agent(args: argparse.Namespace):
    """Loads a trained agent and evaluates its performance over multiple episodes."""
    print(f"--- Starting Evaluation ---\nModel: {args.model_path}\nEnv: {args.env}\nBody: {args.body}\n--------------------")

    if not os.path.exists(args.model_path):
        print(f"ERROR: Model not found at '{args.model_path}'")
        sys.exit(1)
        
    set_seed(args.eval_seed)
    
    episode_rewards = []
    episode_lengths = []
    temp_dir = tempfile.mkdtemp()

    try:
        # Load VecNormalize stats from the zip file if they exist
        stats_path: Optional[str] = None
        with zipfile.ZipFile(args.model_path, 'r') as archive:
            if 'vecnormalize.pkl' in archive.namelist():
                archive.extract('vecnormalize.pkl', path=temp_dir)
                stats_path = os.path.join(temp_dir, 'vecnormalize.pkl')
                print("Found and extracted VecNormalize stats.")

        for i in range(args.num_episodes):
            print(f"\n--- Starting Evaluation Episode {i + 1}/{args.num_episodes} ---")
            
            # Create a new env for each episode with a different seed
            current_seed = args.eval_seed + i
            def env_constructor():
                set_seed(current_seed)
                env = build_and_setup_env(args.env, args.body, 
                                          collect_env_params(args.env, args), 
                                          render_mode="human" if args.render else None, 
                                          args=args)
                if args.render:
                    setup_render_window(env, args)
                return env

            venv = make_vec_env(env_constructor, n_envs=1)
            
            if stats_path:
                venv = VecNormalize.load(stats_path, venv)
                venv.training = False
                venv.norm_reward = False

            model = PPO.load(args.model_path, env=venv)

            obs = venv.reset()
            done = False
            total_reward = 0
            ep_len = 0
            
            while not done:
                if args.render:
                    venv.render()
                    time.sleep(1.0 / 60.0)

                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = venv.step(action)
                
                ep_len += 1
            
            final_reward = info[0].get('episode', {}).get('r', 0)
            episode_rewards.append(final_reward)
            episode_lengths.append(ep_len)
            print(f"Episode {i + 1} finished. Reward: {final_reward:.2f}, Length: {ep_len}")

            venv.close() # Close env to free up resources

        # --- Final Report ---
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print(f"Ran {args.num_episodes} episodes with base seed {args.eval_seed}.")
        print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"Mean Length: {np.mean(episode_lengths):.2f}")
        print("="*50)

    except KeyboardInterrupt:
        print("\nEvaluation stopped by user.")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained agent.")
    parser = add_evaluation_args(parser)
    args = parser.parse_args()
    evaluate_agent(args)