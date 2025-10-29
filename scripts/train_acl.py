# scripts/train_acl.py
import os
import sys
import time
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
# START FIX: Thêm import typing cho type hint
from typing import Callable, Union
# END FIX

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import TeachMyAgent.environments
from utils.env_utils import build_and_setup_env, setup_render_window
from utils.shared_args import add_common_args, add_environment_args, add_render_args

PARAM_SPACE_LIMS = np.array([
    [-1.0, 1.0],
    [-1.0, 1.0],
    [-1.0, 1.0],
    [0.0, 0.8],
])

EASY_PARAMS = np.array([0.0, 0.0, 0.0, 0.0])

# START FIX: Sao chép các hàm xử lý schedule từ train_ppo.py
def get_linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Returns a linear schedule function for learning rate."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def parse_schedule(config_value: Union[dict, float, str]):
    """Safely parses a schedule configuration from a dictionary."""
    if isinstance(config_value, dict):
        schedule_type = config_value.get("type")
        if schedule_type == "linear":
            initial_val = float(config_value["initial_value"])
            return get_linear_schedule(initial_val)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
    return float(config_value)
# END FIX

def add_acl_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = add_common_args(parser)
    parser = add_environment_args(parser)
    parser = add_render_args(parser)
    group = parser.add_argument_group("ACL Training Parameters")
    group.add_argument("--run_id", type=str, default="acl_run1", help="Run identifier.")
    group.add_argument("--total_stages", type=int, default=500, help="Total number of curriculum stages.")
    group.add_argument("--student_steps_per_stage", type=int, default=2048 * 8, help="Training steps per stage.")
    group.add_argument("--eval_episodes", type=int, default=10, help="Number of evaluation episodes.")
    group.add_argument("--mastery_threshold", type=float, default=150.0, help="Reward threshold to increase difficulty.")
    group.add_argument("--difficulty_increment", type=float, default=0.01, help="Difficulty increase step.")
    group.add_argument("--render", action="store_true", help="Enable rendering during training and evaluation.")
    group.add_argument("--n_envs", type=int, default=8, help="Number of parallel environments for training.")
    parser.add_argument('--horizon', type=int, default=3000, help="Max steps per episode.")
    return parser

def sample_task_params(difficulty_ratio: float) -> dict:
    low_bounds = EASY_PARAMS
    high_bounds = np.array([lim[1] if p > 0 else lim[0] for p, lim in zip(EASY_PARAMS, PARAM_SPACE_LIMS)])
    current_max = low_bounds + difficulty_ratio * (high_bounds - low_bounds)
    sampled = np.random.uniform(low=low_bounds, high=current_max)
    return {"input_vector": sampled[:3], "water_level": sampled[3]}

def evaluate_student(student_model: PPO, env_id: str, body_type: str, difficulty_ratio: float, num_episodes: int, args: argparse.Namespace, render_mode: str = None) -> float:
    total_rewards = 0.0
    eval_env = None
    try:
        initial_task = sample_task_params(difficulty_ratio)
        eval_env = build_and_setup_env(env_id, body_type, initial_task, render_mode=render_mode, args=args)
        if render_mode == "human":
            setup_render_window(eval_env, args)
        for i in range(num_episodes):
            print(f"  Evaluating Episode {i + 1}/{num_episodes}...")
            if i > 0:
                task = sample_task_params(difficulty_ratio)
                eval_env.unwrapped.set_environment(**task)
            obs, _ = eval_env.reset()
            terminated, truncated = False, False
            episode_reward = 0.0
            while not (terminated or truncated):
                if render_mode == "human":
                    eval_env.render()
                    if eval_env.unwrapped.viewer and eval_env.unwrapped.viewer.window.has_exit:
                        print("Window closed. Evaluation stopped.")
                        return -np.inf
                action, _ = student_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                episode_reward += reward
            total_rewards += episode_reward
    finally:
        if eval_env:
            eval_env.close()
    if 'eval_env' in locals() and render_mode == "human" and (not eval_env.unwrapped.viewer or not eval_env.unwrapped.viewer.window or eval_env.unwrapped.viewer.window.has_exit):
        return -np.inf
    return total_rewards / num_episodes

def main(args: argparse.Namespace) -> str:
    if args.env != "parkour":
        print(f"ERROR: ACL training currently supports only 'parkour', got '{args.env}'.")
        sys.exit(1)

    output_base = f"output/acl/{args.run_id}"
    model_dir = os.path.join(output_base, "models")
    log_dir = os.path.join(output_base, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'=' * 60}\nStarting ACL training for env '{args.env}' with body '{args.body}'")
    print(f"Output directory: {output_base}\n{'=' * 60}")

    render_mode = "human" if args.render else None
    
    print("--- Initializing Student (PPO) ---")
    initial_params = sample_task_params(0.0)
    
    initial_n_envs = 1 if args.render else args.n_envs
    
    initial_vec_env = make_vec_env(
        lambda: build_and_setup_env(args.env, args.body, initial_params, args=args),
        n_envs=initial_n_envs,
        seed=getattr(args, 'seed', None)
    )

    ppo_kwargs = getattr(args, 'ppo_config', {})
    
    # START FIX: Xử lý các giá trị schedule trước khi truyền vào PPO
    if 'learning_rate_schedule' in ppo_kwargs:
        ppo_kwargs['learning_rate'] = parse_schedule(ppo_kwargs.pop('learning_rate_schedule'))
    if 'clip_range_schedule' in ppo_kwargs:
        ppo_kwargs['clip_range'] = parse_schedule(ppo_kwargs.pop('clip_range_schedule'))
    # END FIX

    print("Using PPO hyperparameters for student:", ppo_kwargs)
    
    student = PPO("MlpPolicy", initial_vec_env, verbose=0, tensorboard_log=os.path.join(log_dir, "student"), **ppo_kwargs)
    
    difficulty_ratio = 0.0
    print(f"\n--- Starting ACL training over {args.total_stages} stages ---")

    try:
        for stage in range(1, args.total_stages + 1):
            start_time = time.time()
            task_params = sample_task_params(difficulty_ratio)
            print(f"\n--- Stage {stage}/{args.total_stages} | Difficulty: {difficulty_ratio:.3f} ---")
            print(f"Task parameters: {task_params}")

            env_fn = lambda: build_and_setup_env(args.env, args.body, task_params, render_mode=render_mode, args=args)
            n_envs = 1 if args.render else args.n_envs
            vec_env = make_vec_env(env_fn, n_envs=n_envs, seed=getattr(args, 'seed', None))
            if render_mode == "human":
                render_env = vec_env.envs[0]
                setup_render_window(render_env, args)

            student.set_env(vec_env)
            student.learn(total_timesteps=args.student_steps_per_stage, reset_num_timesteps=False, progress_bar=True)
            
            print(f"Evaluating agent over {args.eval_episodes} episodes...")
            avg_reward = evaluate_student(student, args.env, args.body, difficulty_ratio, args.eval_episodes, args, render_mode=render_mode)
            
            print(f"Average reward = {avg_reward:.2f}")
            if avg_reward > args.mastery_threshold:
                new_diff = min(1.0, difficulty_ratio + args.difficulty_increment)
                if new_diff > difficulty_ratio:
                    difficulty_ratio = new_diff
                    print(f"Mastery achieved! Increasing difficulty to {difficulty_ratio:.3f}")
            else:
                print(f"Below threshold ({args.mastery_threshold:.2f}). Keeping difficulty unchanged.")

            if stage % 25 == 0:
                model_path = os.path.join(model_dir, f"student_{args.run_id}_stage_{stage}.zip")
                student.save(model_path)
                print(f"Model saved: {model_path}")

            print(f"Stage time: {time.time() - start_time:.2f}s")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        if student and student.get_env() is not None:
            student.get_env().close()

    final_path = os.path.join(model_dir, "student_final.zip")
    student.save(final_path)

    print(f"\n{'=' * 60}\nACL training complete.")
    print(f"Final Student model saved at: {final_path}")
    return final_path