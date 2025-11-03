# scripts/train_marl.py
import argparse
import os
import sys
import time
import ray
from collections.abc import Mapping
from typing import Optional
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env
from gymnasium.wrappers import TransformObservation
import numpy as np
import gymnasium as gym
import TeachMyAgent.environments


from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    LEARNER_RESULTS,
    ALL_MODULES,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import TeachMyAgent.environments
from TeachMyAgent.environments.envs.multi_agent_parametric_parkour import MultiAgentParkour
from TeachMyAgent.environments.envs.interactive_multi_agent_parkour import InteractiveMultiAgentParkour

def add_marl_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--mode", type=str, default="interactive", choices=["cooperative", "interactive"], help="Environment mode.")
    parser.add_argument("--n-agents", type=int, default=2, help="Number of agents.")
    parser.add_argument("--body", type=str, default="classic_bipedal", help="Agent body type.")
    parser.add_argument("--iterations", type=int, default=100, help="Number of training iterations.")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of Ray rollout workers.")
    parser.add_argument("--num-gpus", type=float, default=0, help="Number of GPUs to use for training.")
    parser.add_argument("--check-env", action="store_true", help="Run a random environment check.")
    parser.add_argument("--width", type=int, help="Render window width.")
    parser.add_argument("--height", type=int, help="Render window height.")
    parser.add_argument("--fullscreen", action="store_true", help="Render in fullscreen.")
    parser.add_argument("--run_id", type=str, default="marl_run1", help="Identifier for the training run.")
    parser.add_argument("--use-tune", action="store_true", help="Use Ray Tune for hyperparameter tuning.")
    parser.add_argument('--horizon', type=int, default=3000, help="Max steps per episode.")
    parser.add_argument("--reward-type", type=str, default="individual", choices=["individual", "shared"], help="MARL reward structure.")
    parser.add_argument("--shared-policy", action="store_true", help="Use a single shared policy for all agents.")
    parser.add_argument("--use-cc", action="store_true", help="Use a Centralized Critic.")
    return parser

def deep_update(original, new_dict):
    for key, value in new_dict.items():
        if key in original and isinstance(original[key], Mapping) and isinstance(value, Mapping):
            deep_update(original[key], value)
        else:
            original[key] = value
    return original

def _parse_search_space(search_space_config):
    if not search_space_config: return {}
    TUNE_SAMPLERS = {"grid_search": tune.grid_search, "choice": tune.choice, "uniform": tune.uniform, "loguniform": tune.loguniform}
    def recursive_parse(config_dict):
        parsed_dict = {}
        for key, value in config_dict.items():
            if isinstance(value, str):
                is_sampler = False
                for sampler_name, sampler_func in TUNE_SAMPLERS.items():
                    if value.startswith(f"tune.{sampler_name}"):
                        args_str = value[len(f"tune.{sampler_name}"):]
                        try:
                            args_val = eval(args_str)
                            parsed_dict[key] = sampler_func(*args_val) if isinstance(args_val, tuple) else sampler_func(args_val)
                            is_sampler = True
                            break
                        except Exception as e:
                            print(f"ERROR: Could not parse tune args '{args_str}'. Error: {e}"); sys.exit(1)
                if not is_sampler: parsed_dict[key] = value
            elif isinstance(value, dict): parsed_dict[key] = recursive_parse(value)
            else: parsed_dict[key] = value
        return parsed_dict
    return recursive_parse(search_space_config)

def get_env_creator(mode):
    if mode == "cooperative": base_env_creator_func = lambda config: MultiAgentParkour(config=config)
    elif mode == "interactive": base_env_creator_func = lambda config: InteractiveMultiAgentParkour(config=config)
    else: raise ValueError(f"Invalid mode: '{mode}'.")
    
    # START FIX: Wrap the environment to clip observations and prevent NaN/inf values from crashing workers.
    def wrapped_env_creator(config):
        base_env = base_env_creator_func(config)
        
        # This function will be applied to the observation from the environment at each step.
        def clip_multi_agent_obs(obs_dict):
            return {agent_id: np.clip(agent_obs, -10.0, 10.0) for agent_id, agent_obs in obs_dict.items()}

        # We must also update the observation space to reflect the clipped values.
        original_obs_space = base_env.observation_space
        clipped_sub_spaces = {}
        if isinstance(original_obs_space, gym.spaces.Dict):
             for agent_id, sub_space in original_obs_space.items():
                clipped_sub_spaces[agent_id] = gym.spaces.Box(low=-10.0, high=10.0, shape=sub_space.shape, dtype=sub_space.dtype)
        
        clipped_observation_space = gym.spaces.Dict(clipped_sub_spaces)
        
        # Return the original environment wrapped with our observation transformation.
        return TransformObservation(base_env, clip_multi_agent_obs, clipped_observation_space)
    # END FIX
    return wrapped_env_creator

def main(args: argparse.Namespace) -> Optional[str]:
    if hasattr(args, 'check_env') and args.check_env: return

    if not hasattr(args, 'horizon'): args.horizon = 3000

    env_name = f"marl-{args.mode}-v0"
    register_env(env_name, get_env_creator(args.mode))
    env_config = {"n_agents": args.n_agents, "agent_body_type": args.body, "horizon": args.horizon, "reward_type": args.reward_type}
    
    with get_env_creator(args.mode)(env_config) as temp_env:
        obs_space, act_space = temp_env.observation_space["agent_0"], temp_env.action_space["agent_0"]

    ppo_config_dict = getattr(args, 'ppo_config', {}); training_config = ppo_config_dict.get('training', {})
    model_config = ppo_config_dict.get('model', {}); env_runners_config = ppo_config_dict.get('env_runners', {})

    config = (PPOConfig().environment(env=env_name, env_config=env_config, normalize_actions=True)
        .env_runners(num_env_runners=args.num_workers, **env_runners_config).framework("torch").training(**training_config)
        .rl_module(model_config=model_config).resources(num_gpus=args.num_gpus))

    if hasattr(args, 'shared_policy') and args.shared_policy:
        print("Using Parameter Sharing.")
        config.multi_agent(policies={"shared_policy": PolicySpec(observation_space=obs_space, action_space=act_space)},
                           policy_mapping_fn=lambda agent_id, *a, **kw: "shared_policy")
    else:
        print("Using Independent Learning.")
        config.multi_agent(policies={f"agent_{i}": PolicySpec(observation_space=obs_space, action_space=act_space) for i in range(args.n_agents)},
                           policy_mapping_fn=lambda agent_id, *a, **kw: agent_id)

    if hasattr(args, 'use_cc') and args.use_cc:
        print("Activating Centralized Critic.")
        config.update_from_dict({"use_centralized_value_function": True})

    use_tune = getattr(args, 'use_tune', False)

    if use_tune:
        print("Starting MARL training with Ray Tune...")
        tune_config = getattr(args, 'tune_config', {}); search_space_config = tune_config.get('search_space', {})
        run_config_dict = tune_config.get('run_config', {})
        if 'stop' not in run_config_dict: raise ValueError("`stop` criteria must be defined in `tune_config.run_config` when `use_tune` is true.")
        
        param_space = config.to_dict(); deep_update(param_space, _parse_search_space(search_space_config))
        
        checkpoint_config_dict = run_config_dict.get('checkpoint_config', {}); checkpoint_config_dict.setdefault('num_to_keep', 1)
        checkpoint_config_dict.setdefault('checkpoint_score_attribute', f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}"); checkpoint_config_dict.setdefault('checkpoint_score_order', 'max')
        
        storage_path = os.path.abspath(os.path.join("output", "marl")); run_name = run_config_dict.get('name', args.run_id)
        print(f"Checkpoints will be saved to: {os.path.join(storage_path, run_name)}")

        run_config = air.RunConfig(name=run_name, stop=run_config_dict['stop'], verbose=run_config_dict.get('verbose', 2),
                                   checkpoint_config=air.CheckpointConfig(**checkpoint_config_dict), storage_path=storage_path)

        tuner = tune.Tuner("PPO", param_space=param_space, run_config=run_config, tune_config=tune.TuneConfig(trial_dirname_creator=lambda t: t.trial_id))
        results = tuner.fit()
        best_result = results.get_best_result(metric=f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}", mode="max")
        best_checkpoint_path = best_result.checkpoint.path if best_result and best_result.checkpoint else None
        
        print("\n" + "="*60 + "\nRay Tune finished!")
        if best_result:
            print(f"Best trial final reward: {best_result.metrics.get(ENV_RUNNER_RESULTS, {}).get(EPISODE_RETURN_MEAN, 'N/A')}")
            print(f"Best trial config: {best_result.config}\nBest checkpoint saved at: {best_checkpoint_path}")
        else: print("No trials completed successfully.")
        print("="*60)
        return best_checkpoint_path
    else:
        print(f"Starting a single MARL training run for {args.iterations} iterations...")
        algo = config.build()
        output_dir = os.path.abspath(os.path.join("output", "marl", args.run_id)); os.makedirs(output_dir, exist_ok=True)
        print(f"Checkpoints will be saved to: {output_dir}")
        print(f"TensorBoard logs will be saved under: {algo.logdir}")

        best_reward, checkpoint_path_to_return = -float('inf'), None
        for i in range(args.iterations):
            result = algo.train()
            env_runner_results, learner_results = result.get(ENV_RUNNER_RESULTS, {}), result.get(LEARNER_RESULTS, {})
            episode_reward_mean = env_runner_results.get(EPISODE_RETURN_MEAN, float("nan"))
            
            if episode_reward_mean > best_reward:
                best_reward = episode_reward_mean
                checkpoint_result = algo.save(checkpoint_dir=output_dir)
                checkpoint_path_to_return = checkpoint_result.checkpoint.path
                print(f"\nNew best average reward: {best_reward:.2f}. Checkpoint saved at: {checkpoint_path_to_return}")
            
            per_policy_rewards = env_runner_results.get("module_episode_returns_mean", {})
            total_loss_per_module = {mid: stats.get("total_loss", float("nan")) for mid, stats in learner_results.items() if mid != ALL_MODULES}
            
            print(f"Iter {i + 1}/{args.iterations}: Mean Reward: {episode_reward_mean:.2f}, Policy Rewards: {per_policy_rewards}, Module Loss: {total_loss_per_module}")
        
        print(f"\nTraining complete. Best checkpoint saved at: {checkpoint_path_to_return}")
        return checkpoint_path_to_return