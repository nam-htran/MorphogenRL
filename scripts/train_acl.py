# scripts/train_acl.py
import os
import sys
import time
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from typing import Callable, Union

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
    group.add_argument("--render", action='store_true', help="Enable rendering during training and evaluation.")
    group.add_argument("--n_envs", type=int, default=8, help="Number of parallel environments for training.")
    parser.add_argument('--horizon', type=int, default=3000, help="Max steps per episode.")
    return parser

def sample_task_params(difficulty_ratio: float) -> dict:
    """
    Generates environment parameters based on a two-stage curriculum.
    """
    min_spacing = 0.05
    max_spacing = 2.0
    max_height = 10.0
    min_height = 1.5

    if difficulty_ratio < 0.5:
        stage_1_progress = difficulty_ratio * 2
        current_spacing = min_spacing + stage_1_progress * (max_spacing - min_spacing)
        current_height = max_height
    else:
        stage_2_progress = (difficulty_ratio - 0.5) * 2
        current_spacing = max_spacing
        current_height = max_height - stage_2_progress * (max_height - min_height)

    low_bounds = EASY_PARAMS
    high_bounds = np.array([lim[1] if p > 0 else lim[0] for p, lim in zip(EASY_PARAMS, PARAM_SPACE_LIMS)])
    current_max = low_bounds + difficulty_ratio * (high_bounds - low_bounds)
    sampled_terrain_params = np.random.uniform(low=low_bounds, high=current_max)

    return {
        "input_vector": sampled_terrain_params[:3],
        "water_level": sampled_terrain_params[3],
        "creepers_spacing": current_spacing,
        "creepers_height": current_height,
        "creepers_width": 0.5
    }

def evaluate_student(student_model: PPO, env_id: str, body_type: str, difficulty_ratio: float, num_episodes: int, args: argparse.Namespace, render_mode: str = None, stats_path: str = None) -> float:
    total_rewards = 0.0
    vec_eval_env = None
    try:
        def make_eval_env():
            task = sample_task_params(difficulty_ratio)
            return build_and_setup_env(env_id, body_type, task, render_mode=render_mode, args=args)

        vec_eval_env = make_vec_env(make_eval_env, n_envs=1)
        if stats_path and os.path.exists(stats_path):
            vec_eval_env = VecNormalize.load(stats_path, vec_eval_env)
            vec_eval_env.training = False
            vec_eval_env.norm_reward = False
        
        for i in range(num_episodes):
            print(f"  Evaluating Episode {i + 1}/{num_episodes}...")
            # START FIX: The evaluation environment is already recreated for each episode via make_vec_env,
            # so we don't need to call set_environment again. This makes it cleaner.
            # END FIX
            obs = vec_eval_env.reset()
            done = False
            episode_reward = 0.0
            while not done:
                if render_mode == "human":
                    vec_eval_env.render()
                action, _ = student_model.predict(obs, deterministic=True)
                obs, reward, done_vec, info = vec_eval_env.step(action)
                done = done_vec[0]
                if done: # Use the final info dict to get the true episode reward
                    episode_reward = info[0].get('episode', {}).get('r', 0)
            total_rewards += episode_reward
    finally:
        if vec_eval_env:
            vec_eval_env.close()

    return total_rewards / num_episodes

def main(args: argparse.Namespace) -> str:
    if args.env != "parkour":
        print(f"ERROR: ACL training currently supports only 'parkour', got '{args.env}'.")
        sys.exit(1)

    output_base = f"output/acl/{args.run_id}"
    model_dir = os.path.join(output_base, "models")
    log_dir = os.path.join(output_base, "logs")
    stats_path = os.path.join(output_base, "vecnormalize.pkl")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'=' * 60}\nStarting ACL training for env '{args.env}' with body '{args.body}'")
    print(f"Output directory: {output_base}\n{'=' * 60}")

    render_mode = "human" if args.render else None
    
    # --- PPO Hyperparameters ---
    ppo_kwargs = getattr(args, 'ppo_config', {})
    if 'learning_rate_schedule' in ppo_kwargs:
        ppo_kwargs['learning_rate'] = parse_schedule(ppo_kwargs.pop('learning_rate_schedule'))
    if 'clip_range_schedule' in ppo_kwargs:
        ppo_kwargs['clip_range'] = parse_schedule(ppo_kwargs.pop('clip_range_schedule'))
    print("Using PPO hyperparameters for student:", ppo_kwargs)
    
    # --- ACL Main Loop ---
    difficulty_ratio = 0.0
    student = None
    final_path = os.path.join(model_dir, "student_final.zip")

    try:
        for stage in range(1, args.total_stages + 1):
            start_time = time.time()
            task_params = sample_task_params(difficulty_ratio)
            print(f"\n--- Stage {stage}/{args.total_stages} | Difficulty: {difficulty_ratio:.3f} ---")
            print(f"Task parameters: {task_params}")

            # START FIX: Implement Clean Reset Logic
            # 1. Save current model and environment stats (if they exist)
            if student:
                temp_model_path = os.path.join(model_dir, "student_temp.zip")
                student.save(temp_model_path)
                student.get_env().save(stats_path)
                # Close the old environment completely
                student.get_env().close()
                del student # Free up memory
            else:
                temp_model_path = None

            # 2. Create a brand new vectorized environment with the new task parameters
            n_envs = 1 if args.render else args.n_envs
            vec_env = make_vec_env(
                lambda: build_and_setup_env(args.env, args.body, task_params, args=args),
                n_envs=n_envs,
                seed=getattr(args, 'seed', None)
            )
            
            # 3. Load old normalization stats if they exist, otherwise create new ones
            if os.path.exists(stats_path):
                print(f"Loading VecNormalize stats from: {stats_path}")
                vec_env = VecNormalize.load(stats_path, vec_env)
            else:
                print("Creating new VecNormalize stats.")
                vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

            # 4. Create a new PPO model and load the weights from the previous stage
            if temp_model_path and os.path.exists(temp_model_path):
                print(f"Loading model weights from previous stage: {temp_model_path}")
                student = PPO.load(temp_model_path, env=vec_env, tensorboard_log=os.path.join(log_dir, "student"), device="auto")
                # Reset timesteps to ensure tensorboard logs correctly for the new stage
                student.set_logger(student.logger) 
            else:
                print("Initializing new PPO model for the first stage.")
                student = PPO("MlpPolicy", vec_env, verbose=0, tensorboard_log=os.path.join(log_dir, "student"), device="auto", **ppo_kwargs)
            # END FIX

            student.learn(total_timesteps=args.student_steps_per_stage, reset_num_timesteps=False, progress_bar=True)
            
            # Save the normalization stats from the current training environment
            student.get_env().save(stats_path)
            
            print(f"Evaluating agent over {args.eval_episodes} episodes...")
            # During evaluation, we need a separate, non-training environment
            eval_model_path = os.path.join(model_dir, "student_eval_temp.zip")
            student.save(eval_model_path)
            eval_model = PPO.load(eval_model_path, device="auto")
            
            avg_reward = evaluate_student(eval_model, args.env, args.body, difficulty_ratio, args.eval_episodes, args, render_mode=render_mode, stats_path=stats_path)
            del eval_model # Clean up
            
            print(f"Average reward = {avg_reward:.2f}")
            if avg_reward > args.mastery_threshold:
                new_diff = min(1.0, difficulty_ratio + args.difficulty_increment)
                if new_diff > difficulty_ratio:
                    difficulty_ratio = new_diff
                    print(f"Mastery achieved! Increasing difficulty to {difficulty_ratio:.3f}")
            else:
                print(f"Below threshold ({args.mastery_threshold:.2f}). Keeping difficulty unchanged.")

            if stage % 25 == 0 or stage == args.total_stages:
                model_path = os.path.join(model_dir, f"student_{args.run_id}_stage_{stage}.zip")
                student.save(model_path)
                print(f"Model saved: {model_path}")

            print(f"Stage time: {time.time() - start_time:.2f}s")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        if student and student.get_env():
            student.get_env().save(stats_path)
            print(f"Final VecNormalize stats saved to: {stats_path}")
            student.save(final_path)
            student.get_env().close()

    print(f"\n{'=' * 60}\nACL training complete.")
    print(f"Final Student model saved at: {final_path}")
    return final_path