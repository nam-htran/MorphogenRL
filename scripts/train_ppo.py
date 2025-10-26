# scripts/train_ppo.py
import os
import sys
import argparse
import time

# Add root folder to import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

import TeachMyAgent.environments
from utils.env_utils import build_and_setup_env, collect_env_params, setup_render_window
from utils.shared_args import add_common_args, add_environment_args, add_render_args


def add_ppo_args(parser):
    parser = add_common_args(parser)
    parser = add_environment_args(parser)
    parser = add_render_args(parser)
    train_group = parser.add_argument_group('PPO Training Parameters')
    train_group.add_argument('--total_timesteps', type=int, default=1_000_000)
    train_group.add_argument('--run_id', type=str, default='ppo_run1')
    train_group.add_argument('--save_freq', type=int, default=50_000)
    train_group.add_argument('--n_envs', type=int, default=4)
    train_group.add_argument('--render', action='store_true')
    return parser


def main(args):
    output_base_dir = f"output/ppo/{args.run_id}"
    model_dir = os.path.join(output_base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 50 + f"\nTraining PPO: {args.env} with body {args.body}\nSaving to: {output_base_dir}\n" + "=" * 50)

    train_render_mode = "human" if args.render else None
    if args.render:
        args.n_envs = 1

    user_params = collect_env_params(args.env, args)
    env_lambda = lambda: build_and_setup_env(args.env, args.body, user_params, render_mode=train_render_mode)
    env = make_vec_env(env_lambda, n_envs=args.n_envs)

    if args.render:
        setup_render_window(env.envs[0], args)

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=os.path.join(output_base_dir, "logs"))
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // args.n_envs, 1),
        save_path=model_dir,
        name_prefix="ppo_model"
    )

    try:
        model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback, progress_bar=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    final_model_path = os.path.join(model_dir, "ppo_model_final.zip")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    env.close()
    return final_model_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train PPO (Stable Baselines 3).")
    parser = add_ppo_args(parser)
    args = parser.parse_args()
    main(args)
