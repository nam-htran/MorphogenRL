# scripts/train_marl.py
import argparse
import os
import sys
from collections.abc import Mapping
from typing import Optional
import ray
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env
from gymnasium.wrappers import TransformObservation
import numpy as np
import gymnasium as gym

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import TeachMyAgent.environments
from TeachMyAgent.environments.envs.multi_agent_parametric_parkour import MultiAgentParkour
from TeachMyAgent.environments.envs.interactive_multi_agent_parkour import InteractiveMultiAgentParkour

# START FIX: Add the missing print_header function
def print_header(title: str):
    width = 80
    print("\n\n" + "#"*width + f"\n##{title:^76}##\n" + "#"*width)
# END FIX

def add_marl_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--mode", type=str, default="interactive", choices=["cooperative", "interactive"], help="Environment mode.")
    parser.add_argument("--n-agents", type=int, default=2, help="Number of agents.")
    parser.add_argument("--body", type=str, default="classic_bipedal", help="Agent body type.")
    parser.add_argument("--iterations", type=int, default=100, help="Number of training iterations.")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of Ray rollout workers.")
    parser.add_argument("--num-gpus", type=float, default=0, help="Number of GPUs to use for training.")
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
    env_name = f"marl-{args.mode}-v0"
    register_env(env_name, get_env_creator(args.mode))
    env_config = {"n_agents": args.n_agents, "agent_body_type": args.body, "horizon": args.horizon, "reward_type": args.reward_type}
    
    temp_env = get_env_creator(args.mode)(env_config)
    obs_space, act_space = temp_env.observation_space["agent_0"], temp_env.action_space["agent_0"]
    temp_env.close()

    ppo_config_dict = getattr(args, 'ppo_config', {}); training_config = ppo_config_dict.get('training', {})
    model_config = ppo_config_dict.get('model', {}); env_runners_config = ppo_config_dict.get('env_runners', {})

    config = (PPOConfig().environment(env=env_name, env_config=env_config, normalize_actions=True)
        .env_runners(num_env_runners=args.num_workers, **env_runners_config).framework("torch").training(**training_config)
        .rl_module(model_config=model_config).resources(num_gpus=args.num_gpus))

    if hasattr(args, 'shared_policy') and args.shared_policy:
        config.multi_agent(policies={"shared_policy": PolicySpec(observation_space=obs_space, action_space=act_space)},
                           policy_mapping_fn=lambda agent_id, *a, **kw: "shared_policy")
    else:
        config.multi_agent(policies={f"agent_{i}": PolicySpec(observation_space=obs_space, action_space=act_space) for i in range(args.n_agents)},
                           policy_mapping_fn=lambda agent_id, *a, **kw: agent_id)

    if hasattr(args, 'use_cc') and args.use_cc:
        config.update_from_dict({"use_centralized_value_function": True})

    use_tune = getattr(args, 'use_tune', False)
    output_dir = os.path.abspath(os.path.join("output", "marl", args.run_id))

    if use_tune:
        print_header("STARTING RAY TUNE HYPERPARAMETER SEARCH")
        tune_config = getattr(args, 'tune_config', {})
        search_space = _parse_search_space(tune_config.get('search_space', {}))
        
        # Merge search space into the main config
        config_dict = config.to_dict()
        final_search_space = deep_update(config_dict, search_space)

        # Configure stopper
        stopper_config = tune_config.get('stopper', {})
        stopper = ray.tune.stopper.ExperimentPlateauStopper(
            metric=stopper_config.get("metric", "episode_reward_mean"),
            std=stopper_config.get("std", 0.01),
            patience=stopper_config.get("patience", 10),
            top=stopper_config.get("top", 10),
            mode=stopper_config.get("mode", "max")
        )

        run_config = air.RunConfig(
            name=args.run_id,
            stop=stopper,
            local_dir=os.path.dirname(output_dir),
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=3,
                checkpoint_score_attribute="episode_reward_mean",
                checkpoint_score_order="max"
            ),
        )

        tuner = tune.Tuner(
            "PPO",
            param_space=final_search_space,
            run_config=run_config,
            tune_config=tune.TuneConfig(
                num_samples=tune_config.get("num_samples", 10),
                metric="episode_reward_mean",
                mode="max",
            ),
        )
        
        results = tuner.fit()
        best_result = results.get_best_result()
        best_checkpoint_path = best_result.checkpoint.path
        print(f"\nTune complete. Best checkpoint saved at: {best_checkpoint_path}")
        print(f"Best hyperparameters found: {best_result.config}")
        return best_checkpoint_path

    else:
        print_header(f"STARTING SINGLE MARL RUN ({args.iterations} ITERATIONS)")
        algo = config.build()
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"Checkpoints will be saved to: {output_dir}")
        print(f"TensorBoard logs: {algo.logdir}")

        best_reward, best_checkpoint_path = -float('inf'), None
        for i in range(args.iterations):
            result = algo.train()
            reward_mean = result.get("episode_reward_mean", float('nan'))
            
            if reward_mean > best_reward:
                best_reward = reward_mean
                checkpoint_result = algo.save(checkpoint_dir=output_dir)
                best_checkpoint_path = checkpoint_result.checkpoint.path
                print(f"\nNew best avg reward: {best_reward:.2f}. Checkpoint: {best_checkpoint_path}")
            
            print(f"Iter {i+1}/{args.iterations}: Mean Reward: {reward_mean:.2f}")
        
        print(f"\nTraining complete. Best checkpoint at: {best_checkpoint_path}")
        return best_checkpoint_path