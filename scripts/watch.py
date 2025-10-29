# scripts/watch.py
import argparse
import time
import os
import sys
import numpy as np
import gymnasium as gym
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
import TeachMyAgent.environments
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.env_utils import build_and_setup_env, collect_env_params, setup_render_window
from utils.shared_args import add_common_args, add_environment_args, add_render_args
from scripts.train_marl import get_env_creator

class UserInterrupt(Exception):
    pass

def add_watch_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Adds arguments for watching both SB3 and RLlib models."""
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to the trained model (.zip for SB3, checkpoint directory for RLlib).")

    watch_group = parser.add_argument_group('Playback Parameters')
    watch_group.add_argument('--framework', type=str, default='auto', choices=['auto', 'sb3', 'rllib'],
                             help="Model's framework. 'auto' will attempt to autodetect.")
    watch_group.add_argument('--num-episodes', type=int, default=10, help="Number of episodes to watch.")
    watch_group.add_argument('--timeout', type=int, default=5000, help="Maximum number of steps per episode.")
    watch_group.add_argument('--fast-forward', '-ff', action='store_true', help="Skip the delay between frames.")

    parser = add_common_args(parser)
    parser = add_environment_args(parser)
    parser = add_render_args(parser)

    rllib_group = parser.add_argument_group('RLlib/MARL Parameters')
    rllib_group.add_argument('--mode', type=str, default='interactive', choices=['cooperative', 'interactive'],
                             help="[RLlib] Environment mode (cooperative or interactive).")
    rllib_group.add_argument('--n-agents', type=int, default=2, help="[RLlib] Number of agents.")

    return parser

def _watch_sb3(args):
    """Handles watching a Stable Baselines 3 model."""
    print("--- Stable Baselines 3 Watch Mode (PPO/ACL) ---")
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model file not found at '{args.model_path}'")
        sys.exit(1)

    model_dir = os.path.dirname(args.model_path)
    stats_path = os.path.join(os.path.dirname(model_dir), "vecnormalize.pkl")
    if not os.path.exists(stats_path):
        stats_path = os.path.join(model_dir, "vecnormalize.pkl")
        if not os.path.exists(stats_path):
            print(f"ERROR: VecNormalize stats file not found.")
            sys.exit(1)

    print(f"Found VecNormalize stats at: {stats_path}")

    venv = DummyVecEnv([lambda: build_and_setup_env(
        args.env, args.body,
        collect_env_params(args.env, args),
        render_mode="human", args=args
    )])
    venv = VecNormalize.load(stats_path, venv)
    venv.training = False
    venv.norm_reward = False
    print("Successfully loaded and applied VecNormalize stats.")

    setup_render_window(venv.envs[0], args)

    print("Loading model...")
    model = PPO.load(args.model_path, env=venv)
    print("Model loaded successfully.")

    for i in range(args.num_episodes):
        obs = venv.reset()
        done = False
        ep_len = 0
        total_reward = 0.0
        print(f"\n--- Starting episode {i + 1}/{args.num_episodes} ---")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_vec, infos = venv.step(action)
            venv.render()

            viewer = venv.envs[0].unwrapped.viewer
            if viewer and viewer.window and viewer.window.has_exit:
                raise UserInterrupt("Render window was closed.")

            total_reward += infos[0].get('reward', reward[0])
            is_done = done_vec[0]
            ep_len += 1

            if not args.fast_forward:
                time.sleep(1.0 / 60.0)

            if not is_done and ep_len >= args.timeout:
                 print(f"  -> Timeout reached ({args.timeout} steps).")
                 is_done = True

            done = is_done

        final_reward = infos[0].get('episode', {}).get('r', total_reward)
        print(f"Episode finished. Reward: {final_reward:.2f}, Length: {ep_len}")

    venv.close()

def _watch_rllib(args):
    """Handles watching a Ray RLlib model."""
    print("--- Ray RLlib Watch Mode (MARL) ---")
    if not os.path.isdir(args.model_path):
        print(f"ERROR: Checkpoint path '{args.model_path}' is not a directory.")
        sys.exit(1)

    env_name = f"marl-{args.mode}-v0"
    env_creator = get_env_creator(args.mode)
    register_env(env_name, env_creator)
    print(f"Registered custom environment: '{env_name}'")

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    try:
        print("Loading agent from checkpoint...")
        algo = Algorithm.from_checkpoint(args.model_path)
        print("Agent loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        if ray.is_initialized():
            ray.shutdown()
        sys.exit(1)

    env_config = {
        "n_agents": args.n_agents,
        "agent_body_type": args.body,
        "horizon": args.timeout,
        "render_mode": "human"
    }

    env = env_creator(env_config)
    setup_render_window(env.unwrapped, args)

    for i in range(args.num_episodes):
        print(f"\n--- Starting episode {i + 1}/{args.num_episodes} ---")

        obs, info = env.reset()
        terminated = {"__all__": False}
        truncated = {"__all__": False}
        ep_len = 0
        total_rewards = {agent_id: 0 for agent_id in env.unwrapped.possible_agents}
        
        if not hasattr(algo.config, "policy_mapping_fn"):
            raise AttributeError("Could not find 'policy_mapping_fn' in the algorithm's config. "
                                 "Please check train_marl.py.")
        policy_map_fn = algo.config.policy_mapping_fn

        while not terminated["__all__"] and not truncated["__all__"]:
            env.render()
            
            viewer = env.unwrapped.viewer
            if viewer and viewer.window and viewer.window.has_exit:
                raise UserInterrupt("Render window was closed.")
            
            actions = {}
            active_agents = obs.keys()

            for agent_id in active_agents:
                policy_id = policy_map_fn(agent_id, episode=None)
                module = algo.get_module(policy_id)
                obs_tensor = torch.from_numpy(obs[agent_id]).unsqueeze(0)
                with torch.no_grad():
                    action_dist_inputs = module.forward_inference({"obs": obs_tensor})['action_dist_inputs']
                mean, _ = torch.chunk(action_dist_inputs, 2, dim=-1)
                actions[agent_id] = mean.squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(actions)

            for agent_id, r in reward.items():
                if agent_id in total_rewards:
                    total_rewards[agent_id] += r

            ep_len += 1
            if not args.fast_forward:
                time.sleep(1.0 / 60.0)

        print(f"Episode finished. Rewards: {total_rewards}, Length: {ep_len}")

    env.close()

def main(args: argparse.Namespace):
    """Unified main function, autodetects the framework and calls the corresponding handler."""
    print(f"--- Starting model playback ---\nPath: {args.model_path}\n--------------------")

    framework = args.framework
    if framework == 'auto':
        if os.path.isdir(args.model_path):
            framework = 'rllib'
        elif os.path.isfile(args.model_path) and args.model_path.lower().endswith('.zip'):
            framework = 'sb3'
        else:
            print(f"ERROR: Could not autodetect framework from path '{args.model_path}'.")
            sys.exit(1)

    try:
        if framework == 'sb3':
            _watch_sb3(args)
        elif framework == 'rllib':
            _watch_rllib(args)
    except (KeyboardInterrupt, UserInterrupt) as e:
        print(f"\nPlayback stopped by user: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if __name__ == "__main__" and ray.is_initialized():
            ray.shutdown()
        print("Environment closed and resources cleaned up.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Watch a trained agent (supports SB3 and RLlib).")
    parser = add_watch_args(parser)
    args = parser.parse_args()
    main(args)