import argparse
import os
import sys
import time
import ray
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import TeachMyAgent.environments
from TeachMyAgent.environments.envs.multi_agent_parametric_parkour import MultiAgentParkour
from TeachMyAgent.environments.envs.interactive_multi_agent_parkour import InteractiveMultiAgentParkour
from utils.env_utils import get_screen_resolution, setup_render_window as setup_render_window_util


def add_marl_args(parser):
    parser.add_argument("--mode", type=str, default="cooperative", choices=["cooperative", "interactive"], help="Environment mode.")
    parser.add_argument("--n-agents", type=int, default=2, help="Number of agents.")
    parser.add_argument("--body", type=str, default="classic_bipedal", help="Agent body type.")
    parser.add_argument("--iterations", type=int, default=10, help="Number of training iterations.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers.")
    parser.add_argument("--check-env", action="store_true", help="Run environment check only.")
    parser.add_argument("--width", type=int, help="Render window width.")
    parser.add_argument("--height", type=int, help="Render window height.")
    parser.add_argument("--fullscreen", action="store_true", help="Run in fullscreen render mode.")
    parser.add_argument("--run_id", type=str, default="marl_run1", help="Run identifier.")
    return parser


def get_env_creator(mode):
    if mode == "cooperative":
        return lambda config: MultiAgentParkour(config=config)
    elif mode == "interactive":
        return lambda config: InteractiveMultiAgentParkour(config=config)
    else:
        raise ValueError(f"Invalid mode: '{mode}'. Use 'cooperative' or 'interactive'.")


def check_environment_mode(args):
    print("\n" + "=" * 60)
    print(f" ENVIRONMENT CHECK MODE ({args.mode.upper()})")
    print(f"  - Body: {args.body}, Agents: {args.n_agents}")
    print("=" * 60 + "\n")

    env = None
    try:
        env_creator_func = get_env_creator(args.mode)
        env_config = {"n_agents": args.n_agents, "agent_body_type": args.body, "render_mode": "human"}

        print("1. Initializing environment...")
        env = env_creator_func(env_config)
        print("   -> Success.\n")

        print("2. Setting up render window...")
        setup_render_window_util(env, args)

        print("\n3. Resetting environment...")
        obs, _ = env.reset()
        print("   -> Success.\n")

        steps_to_run = 300
        print(f"4. Running {steps_to_run} random steps...")
        for step in range(steps_to_run):
            env.render()
            viewer = getattr(env, "viewer", None) or getattr(env.envs[0], "viewer", None)
            if viewer and viewer.window.has_exit:
                raise KeyboardInterrupt("Window closed by user.")

            random_actions = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(random_actions)
            time.sleep(1 / 60)

            if terminated["__all__"] or truncated["__all__"]:
                print(f"   Episode ended early after {step + 1} steps. Resetting...")
                obs, _ = env.reset()

        print("\n✅ Environment random-step test passed.")
        print("Keep the render window open. Close it to exit.")
        while True:
            env.render()
            viewer = getattr(env, "viewer", None) or getattr(env.envs[0], "viewer", None)
            if viewer and viewer.window.has_exit:
                break
            time.sleep(1 / 60)

    except (Exception, KeyboardInterrupt) as e:
        if not isinstance(e, KeyboardInterrupt):
            print("\n❌ Environment check failed:", e)
            import traceback
            traceback.print_exc()
    finally:
        if env:
            env.close()
            print("Environment closed.")


def main(args):
    if args.check_env:
        check_environment_mode(args)
        return

    ray.init(ignore_reinit_error=True)

    env_name = f"marl-{args.mode}-v0"
    register_env(env_name, get_env_creator(args.mode))

    env_config = {"n_agents": args.n_agents, "agent_body_type": args.body}
    print(f"Retrieving sample spaces for '{env_name}'...")
    temp_env = get_env_creator(args.mode)(env_config)
    obs_space = temp_env.observation_space["agent_0"]
    act_space = temp_env.action_space["agent_0"]
    temp_env.close()
    print("...Spaces retrieved.\n")

    config = (
        PPOConfig()
        .environment(
            env=env_name,
            env_config=env_config,
            normalize_actions=True
        )
        .env_runners(num_env_runners=args.num_workers, rollout_fragment_length="auto")
        .framework("torch")
        .training(
            lr=1e-5,
            grad_clip=0.5,
            grad_clip_by="norm",
            vf_clip_param=100.0,
            entropy_coeff=0.01
        )
        .multi_agent(
            policies={f"agent_{i}": PolicySpec(observation_space=obs_space, action_space=act_space) for i in range(args.n_agents)},
            policy_mapping_fn=lambda agent_id, *a, **kw: agent_id,
        )
        .resources(num_gpus=0)
    )

    algo = config.build()
    print(f"🚀 Training {args.n_agents} agents ({args.body}) for {args.iterations} iterations...\n")

    for i in range(args.iterations):
        result = algo.train()
        reward_mean = result.get("env_runners", {}).get("episode_reward_mean", float("nan"))
        print(f"Iteration {i + 1}/{args.iterations}: reward_mean={reward_mean:.4f}")

    checkpoint_dir = algo.save().checkpoint.path
    print(f"\n✅ Checkpoint saved at: {checkpoint_dir}")

    ray.shutdown()
    return checkpoint_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_marl_args(parser)
    args = parser.parse_args()
    main(args)
