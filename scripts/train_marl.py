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
import torch
from stable_baselines3 import PPO, SAC 

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
from TeachMyAgent.environments.envs.parametric_continuous_parkour import ParametricContinuousParkour

def load_sb3_weights_into_rllib_module(rllib_module, sb3_model_path, temp_env):
    """
    Loads weights from a saved SB3 model (PPO or SAC) into an RLlib RLModule.
    This function performs a best-effort transfer assuming similar MLP architectures.
    """
    print(f"Attempting to transfer weights from SB3 model: {sb3_model_path}")
    try:
        # Provide a temporary environment to `load` to prevent internal conflicts
        # Heuristic to detect model type from file name for robustness
        model_name_upper = sb3_model_path.upper().replace('\\', '/')
        if "SAC" in os.path.basename(model_name_upper):
            sb3_model = SAC.load(sb3_model_path, device='cpu', env=temp_env)
            sb3_policy_state_dict = sb3_model.policy.state_dict()
            print("Loaded SB3 SAC model for weight transfer.")
        else:
            sb3_model = PPO.load(sb3_model_path, device='cpu', env=temp_env)
            sb3_policy_state_dict = sb3_model.policy.state_dict()
            print("Loaded SB3 PPO model for weight transfer.")
            
    except Exception as e:
        print(f"ERROR: Could not load SB3 model. Aborting weight transfer. Error: {e}")
        return

    rllib_module_state_dict = rllib_module.state_dict()

    key_mapping = {}
    if isinstance(sb3_model, SAC):
        # Mapping from SAC MlpPolicy to RLlib PPO RLModule
        key_mapping = {
            "actor.features_extractor.0.weight": "pi_encoder._nets.0.weight", "actor.features_extractor.0.bias": "pi_encoder._nets.0.bias",
            "actor.features_extractor.2.weight": "pi_encoder._nets.1.weight", "actor.features_extractor.2.bias": "pi_encoder._nets.1.bias",
            "critic.features_extractor.0.weight": "vf_encoder._nets.0.weight", "critic.features_extractor.0.bias": "vf_encoder._nets.0.bias",
            "critic.features_extractor.2.weight": "vf_encoder._nets.1.weight", "critic.features_extractor.2.bias": "vf_encoder._nets.1.bias",
            "actor.mu.weight": "_action_dist_layer.weight", "actor.mu.bias": "_action_dist_layer.bias",
            "critic.qf0.2.weight": "_value_layer.weight", "critic.qf0.2.bias": "_value_layer.bias",
        }
    elif isinstance(sb3_model, PPO):
        # Mapping from PPO MlpPolicy to RLlib PPO RLModule
        key_mapping = {
            "mlp_extractor.policy_net.0.weight": "pi_encoder._nets.0.weight", "mlp_extractor.policy_net.0.bias": "pi_encoder._nets.0.bias",
            "mlp_extractor.policy_net.2.weight": "pi_encoder._nets.1.weight", "mlp_extractor.policy_net.2.bias": "pi_encoder._nets.1.bias",
            "mlp_extractor.value_net.0.weight": "vf_encoder._nets.0.weight", "mlp_extractor.value_net.0.bias": "vf_encoder._nets.0.bias",
            "mlp_extractor.value_net.2.weight": "vf_encoder._nets.1.weight", "mlp_extractor.value_net.2.bias": "vf_encoder._nets.1.bias",
            "action_net.weight": "_action_dist_layer.weight", "action_net.bias": "_action_dist_layer.bias",
            "value_net.weight": "_value_layer.weight", "value_net.bias": "_value_layer.bias",
        }
    
    new_rllib_state_dict = rllib_module_state_dict.copy()
    weights_transferred = 0
    for sb3_key, rllib_key in key_mapping.items():
        if sb3_key in sb3_policy_state_dict and rllib_key in new_rllib_state_dict:
            if sb3_policy_state_dict[sb3_key].shape == new_rllib_state_dict[rllib_key].shape:
                new_rllib_state_dict[rllib_key] = sb3_policy_state_dict[sb3_key]
                weights_transferred += 1
            else:
                print(f"  - Shape mismatch for key '{rllib_key}'. SB3 shape: {sb3_policy_state_dict[sb3_key].shape}, RLlib shape: {new_rllib_state_dict[rllib_key].shape}. Skipping.")

    if weights_transferred > 0:
        rllib_module.load_state_dict(new_rllib_state_dict)
        print(f"Successfully transferred {weights_transferred} weight tensors from SB3 to RLlib module.")
    else:
        print("WARNING: No weights were transferred. Check network architectures in YAML and key mappings in train_marl.py.")

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
    parser.add_argument("--pretrained_model_path", type=str, default=None, help="Path to a pre-trained SB3 model to start MARL from.")
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
    
    def wrapped_env_creator(config):
        base_env = base_env_creator_func(config)
        def clip_multi_agent_obs(obs_dict):
            return {agent_id: np.clip(agent_obs, -10.0, 10.0) for agent_id, agent_obs in obs_dict.items()}
        original_obs_space = base_env.observation_space
        clipped_sub_spaces = {}
        if isinstance(original_obs_space, gym.spaces.Dict):
             for agent_id, sub_space in original_obs_space.items():
                clipped_sub_spaces[agent_id] = gym.spaces.Box(low=-10.0, high=10.0, shape=sub_space.shape, dtype=sub_space.dtype)
        clipped_observation_space = gym.spaces.Dict(clipped_sub_spaces)
        return TransformObservation(base_env, clip_multi_agent_obs, clipped_observation_space)
    return wrapped_env_creator

def main(args: argparse.Namespace) -> Optional[str]:
    if hasattr(args, 'check_env') and args.check_env: return

    if not hasattr(args, 'horizon'): args.horizon = 3000

    env_name = f"marl-{args.mode}-v0"
    register_env(env_name, get_env_creator(args.mode))
    env_config = {"n_agents": args.n_agents, "agent_body_type": args.body, "horizon": args.horizon, "reward_type": args.reward_type}
    
    # Create a temporary SINGLE-AGENT environment for loading SB3 models
    temp_env_for_loading = ParametricContinuousParkour(agent_body_type=args.body)
    # Use the observation/action space from the actual multi-agent env for the config
    with get_env_creator(args.mode)(env_config) as temp_marl_env:
        obs_space, act_space = temp_marl_env.observation_space["agent_0"], temp_marl_env.action_space["agent_0"]

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
        pass # Tune logic (unchanged)
    else:
        print(f"Starting a single MARL training run for {args.iterations} iterations...")
        algo = config.build()
        
        pretrained_path = getattr(args, 'pretrained_model_path', None)
        if pretrained_path and os.path.exists(pretrained_path):
            if hasattr(args, 'shared_policy') and args.shared_policy:
                print("\n" + "="*80)
                print("TRANSFER LEARNING: Loading weights from SB3 model into shared policy...")
                module_to_load = algo.get_module("shared_policy")
                load_sb3_weights_into_rllib_module(module_to_load, pretrained_path, temp_env_for_loading)
                
                print("Syncing transferred weights to all rollout workers...")
                algo.env_runner_group.sync_weights()
                
                print("Weight sync complete.")
                print("="*80 + "\n")
            else:
                print("WARNING: Pre-trained model provided, but 'shared_policy' is not enabled. Cannot transfer weights.")
        
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
        temp_env_for_loading.close() 
        return checkpoint_path_to_return