import os
import sys
import time
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.base_class import BaseAlgorithm
from typing import Callable, Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import TeachMyAgent.environments
from utils.env_utils import build_and_setup_env
from utils.shared_args import add_common_args, add_environment_args, add_render_args

PARAM_SPACE_MAP = {
    "input_vector_0": [-1.0, 1.0],
    "input_vector_1": [-1.0, 1.0],
    "input_vector_2": [-1.0, 1.0],
    "water_level": [0.0, 0.8],
    "creepers_height": [0.0, 10.0],
    "creepers_spacing": [2.0, 0.05]
}

def add_acl_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = add_common_args(parser); parser = add_environment_args(parser); parser = add_render_args(parser)
    group = parser.add_argument_group("ACL Training Parameters")
    group.add_argument("--run_id", type=str, default="acl_run1")
    group.add_argument("--total_stages", type=int, default=50)
    group.add_argument("--student_steps_per_stage", type=int, default=2048 * 8)
    group.add_argument("--eval_episodes", type=int, default=10)
    group.add_argument("--mastery_threshold", type=float, default=150.0)
    group.add_argument("--render", action='store_true')
    group.add_argument("--n_envs", type=int, default=8)
    parser.add_argument('--horizon', type=int, default=3000)
    # XÓA BỎ --pretrained_model_path
    return parser

def sample_task_params(difficulty_vector: np.ndarray, dim_names: list) -> dict:
    params = {}
    current_params = {}

    for i, dim_name in enumerate(dim_names):
        min_val, max_val = PARAM_SPACE_MAP[dim_name]
        current_max = min_val + difficulty_vector[i] * (max_val - min_val)
        if min_val > max_val:
            sampled_val = np.random.uniform(low=current_max, high=min_val)
        else:
            sampled_val = np.random.uniform(low=min_val, high=current_max)
        current_params[dim_name] = sampled_val

    input_vec = [
        current_params.get("input_vector_0", 0.0),
        current_params.get("input_vector_1", 0.0),
        current_params.get("input_vector_2", 0.0)
    ]
    params["input_vector"] = np.array(input_vec)
    if "water_level" in current_params: params["water_level"] = current_params["water_level"]
    if "creepers_height" in current_params: params["creepers_height"] = current_params["creepers_height"]
    if "creepers_spacing" in current_params: params["creepers_spacing"] = current_params["creepers_spacing"]

    if params.get("creepers_height", 0.0) > 0:
        params["creepers_width"] = 0.5
    else:
        params["creepers_width"] = None

    return params

def evaluate_student(student_model: BaseAlgorithm, env_id: str, body_type: str,
                     difficulty_vector: np.ndarray, dim_names: list,
                     num_episodes: int, args: argparse.Namespace, render_mode: str = None, stats_path: str = None) -> float:
    total_rewards = 0.0
    vec_eval_env = None
    try:
        def make_eval_env():
            task_params = sample_task_params(difficulty_vector, dim_names)
            return build_and_setup_env(env_id, body_type, task_params, render_mode=render_mode, args=args)

        vec_eval_env = make_vec_env(make_eval_env, n_envs=1)
        if stats_path and os.path.exists(stats_path):
            vec_eval_env = VecNormalize.load(stats_path, vec_eval_env)
            vec_eval_env.training = False
            vec_eval_env.norm_reward = False

        for _ in range(num_episodes):
            obs = vec_eval_env.reset()
            done = False
            while not done:
                action, _ = student_model.predict(obs, deterministic=True)
                obs, _, done_vec, info = vec_eval_env.step(action)
                done = done_vec[0]
                if done:
                    total_rewards += info[0].get('episode', {}).get('r', 0)
    finally:
        if vec_eval_env: vec_eval_env.close()
    return total_rewards / num_episodes

def main(args: argparse.Namespace) -> str:
    if args.env != "parkour":
        print(f"ERROR: ACL training currently supports only 'parkour', got '{args.env}'."); sys.exit(1)

    output_base = f"output/acl/{args.run_id}"
    os.makedirs(output_base, exist_ok=True)

    print(f"\n{'=' * 60}\nStarting Multi-Dimensional ACL training for '{args.env}'\n{'=' * 60}")

    dim_names = getattr(args, 'difficulty_dims', [])
    initial_difficulty = np.array(getattr(args, 'initial_difficulty', [0.0]*len(dim_names)))
    mastery_order = getattr(args, 'mastery_order', dim_names)
    increments = getattr(args, 'difficulty_increments', {name: 0.1 for name in dim_names})

    difficulty_vector = initial_difficulty
    current_focus_dim_idx = 0

    student, final_path = None, os.path.join(output_base, "student_final.zip")

    try:
        for stage in range(1, args.total_stages + 1):
            print(f"\n--- Stage {stage}/{args.total_stages} | Focus: '{mastery_order[current_focus_dim_idx]}' ---")
            print(f"Current Difficulty Vector: {np.round(difficulty_vector, 3)}")

            def make_env_with_replay():
                replay_difficulty = np.random.rand(len(dim_names)) * difficulty_vector
                task_params = sample_task_params(replay_difficulty, dim_names)
                return build_and_setup_env(args.env, args.body, task_params, args=args, **getattr(args, 'reward_shaping', {}))

            student, _ = train_student_stage(args, stage, student, make_env_with_replay, output_base)

            avg_reward = evaluate_student(
                student, args.env, args.body, difficulty_vector, dim_names,
                args.eval_episodes, args, render_mode="human" if args.render else None,
                stats_path=os.path.join(output_base, "vecnormalize.pkl"))

            print(f"Evaluation Avg Reward = {avg_reward:.2f}")

            if avg_reward > args.mastery_threshold:
                print(f"Mastery achieved for focus '{mastery_order[current_focus_dim_idx]}'!")

                focused_dim_name = mastery_order[current_focus_dim_idx]
                dim_idx_in_vec = dim_names.index(focused_dim_name)
                increment = increments[focused_dim_name]
                difficulty_vector[dim_idx_in_vec] = min(1.0, difficulty_vector[dim_idx_in_vec] + increment)

                if difficulty_vector[dim_idx_in_vec] >= 1.0:
                    print(f"Dimension '{focused_dim_name}' maxed out.")
                    if current_focus_dim_idx < len(mastery_order) - 1:
                        current_focus_dim_idx += 1
                        print(f"Switching focus to '{mastery_order[current_focus_dim_idx]}'.")
                    else:
                        print("All curriculum dimensions mastered!")
            else:
                print(f"Below threshold ({args.mastery_threshold:.2f}). Keeping difficulty unchanged.")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        if student:
            stats_path = os.path.join(output_base, "vecnormalize.pkl")
            student.get_env().save(stats_path)
            student.save(final_path)
            student.get_env().close()

    print(f"\nACL training complete. Final model saved at: {final_path}")
    return final_path

def train_student_stage(args, stage, student_model, env_fn, output_base):
    model_dir = os.path.join(output_base, "models"); os.makedirs(model_dir, exist_ok=True)
    log_dir = os.path.join(output_base, "logs"); os.makedirs(log_dir, exist_ok=True)
    stats_path = os.path.join(output_base, "vecnormalize.pkl")

    if student_model:
        student_model.get_env().save(stats_path)
        student_model.get_env().close()

    n_envs = 1 if args.render else args.n_envs
    vec_env = make_vec_env(env_fn, n_envs=n_envs, seed=getattr(args, 'seed', None))
    vec_env = VecNormalize.load(stats_path, vec_env) if os.path.exists(stats_path) else VecNormalize(vec_env)

    agent_kwargs = getattr(args, 'ppo_config', {})
    AGENT_CLASS = SAC

    # START CHANGE: Đơn giản hóa logic, không còn kiểm tra pretrained model
    if student_model:
        print("Continuing training from previous ACL stage model.")
        student = student_model
        student.set_env(vec_env)
        student.tensorboard_log = log_dir
    else:
        print("Starting ACL from scratch with a new SAC model.")
        student = AGENT_CLASS("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir, device="auto", **agent_kwargs)
    # END CHANGE

    student.learn(total_timesteps=args.student_steps_per_stage, reset_num_timesteps=False, progress_bar=True)

    if stage % 10 == 0 or stage == args.total_stages:
        save_path = os.path.join(model_dir, f"student_stage_{stage}.zip")
        student.save(save_path)
        print(f"Model checkpoint saved: {save_path}")

    return student, stats_path