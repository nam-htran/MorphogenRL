# scripts/train_ppo.py
import os
import sys
import argparse
import time
from typing import Callable, Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

import TeachMyAgent.environments
from utils.env_utils import build_and_setup_env, collect_env_params, setup_render_window
from utils.shared_args import add_common_args, add_environment_args, add_render_args

def add_ppo_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = add_common_args(parser)
    parser = add_environment_args(parser)
    parser = add_render_args(parser)
    train_group = parser.add_argument_group('PPO Training Parameters')
    train_group.add_argument('--total_timesteps', type=int, default=1_000_000)
    train_group.add_argument('--run_id', type=str, default='ppo_run1')
    train_group.add_argument('--save_freq', type=int, default=100_000)
    train_group.add_argument('--n_envs', type=int, default=16)
    train_group.add_argument('--render', action='store_true')
    parser.add_argument('--horizon', type=int, default=5000)
    return parser

# START CHANGE: Simplify schedule parsing to align with current SB3 and YAML usage
def get_linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Returns a linear schedule function."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def parse_schedule(config_value: Union[dict, float]):
    """
    Parses a schedule configuration.
    If it's a dict, create a schedule function.
    If it's a float, return it directly.
    """
    if isinstance(config_value, dict):
        schedule_type = config_value.get("type")
        if schedule_type == "linear":
            initial_val = float(config_value.get("initial_value", 0.0))
            return get_linear_schedule(initial_val)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
    # If it's already a float or can be converted to one, just return it.
    return float(config_value)
# END CHANGE

def main(args: argparse.Namespace) -> str:
    output_base_dir = f"output/ppo/{args.run_id}"
    model_dir = os.path.join(output_base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 50 + f"\nTraining PPO: {args.env} with body {args.body}\nSaving to: {output_base_dir}\n" + "=" * 50)

    train_render_mode = "human" if args.render else None
    if args.render:
        args.n_envs = 1

    # START CHANGE: Pass reward shaping params to the environment constructor
    reward_shaping_params = getattr(args, 'reward_shaping', {})
    
    user_params = collect_env_params(args.env, args)
    # The environment will receive reward_shaping_params through kwargs (**reward_shaping_params)
    env_lambda = lambda: build_and_setup_env(args.env, args.body, user_params, render_mode=train_render_mode, args=args, **reward_shaping_params)
    # END CHANGE
    
    print("Creating vectorized and normalized environment...")
    env = make_vec_env(env_lambda, n_envs=args.n_envs, seed=getattr(args, 'seed', None))
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    print("Environment created successfully.")

    if args.render:
        setup_render_window(env.envs[0], args)
    
    ppo_kwargs = getattr(args, 'ppo_config', {})
    
    # START CHANGE: Correctly handle both direct float and schedule dict for learning_rate
    if 'learning_rate' in ppo_kwargs:
        ppo_kwargs['learning_rate'] = parse_schedule(ppo_kwargs.pop('learning_rate'))
    if 'clip_range' in ppo_kwargs:
         ppo_kwargs['clip_range'] = parse_schedule(ppo_kwargs.pop('clip_range'))
    # END CHANGE

    print("Using PPO hyperparameters:", ppo_kwargs)

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=os.path.join(output_base_dir, "logs"), device="auto", **ppo_kwargs)
    
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
    
    stats_path = os.path.join(output_base_dir, "vecnormalize.pkl")
    env.save(stats_path)
    print(f"VecNormalize stats saved to: {stats_path}")

    env.close()

    return final_model_path