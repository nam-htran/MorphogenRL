# scripts/run_single_test.py
import argparse
import os
import sys
import time
from types import SimpleNamespace

# Add root directory to import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import TeachMyAgent.environments
from utils.env_utils import build_and_setup_env, setup_render_window
from TeachMyAgent.environments.envs.multi_agent_parametric_parkour import MultiAgentParkour
from TeachMyAgent.environments.envs.interactive_multi_agent_parkour import InteractiveMultiAgentParkour


def check_single_agent_env(env_key, body_name, steps, delay):
    """Single-agent test."""
    env = None
    try:
        fake_args = SimpleNamespace(fullscreen=False, width=None, height=None)
        env = build_and_setup_env(env_key, body_name, {}, render_mode="human")
        setup_render_window(env, fake_args)
        env.reset()
        for _ in range(steps):
            env.render()
            if env.unwrapped.viewer and env.unwrapped.viewer.window and env.unwrapped.viewer.window.has_exit:
                break
            action = env.action_space.sample()
            env.step(action)
            time.sleep(delay)
        return True
    finally:
        if env:
            env.close()


def check_multi_agent_env(mode, body_name, steps, delay):
    """Multi-agent test."""
    env = None
    try:
        fake_args = SimpleNamespace(fullscreen=False, width=None, height=None, n_agents=2)
        env_config = {
            "n_agents": fake_args.n_agents,
            "agent_body_type": body_name,
            "render_mode": "human"
        }

        if mode == 'cooperative':
            env = MultiAgentParkour(config=env_config)
            if env.envs:
                setup_render_window(env.envs[0], fake_args)
        else:  # interactive
            env = InteractiveMultiAgentParkour(config=env_config)
            setup_render_window(env, fake_args)
        
        env.reset()
        for _ in range(steps):
            env.render()
            viewer = env.viewer if hasattr(env, 'viewer') and env.viewer else (
                env.envs[0].viewer if hasattr(env, 'envs') and env.envs else None
            )
            if viewer and viewer.window and viewer.window.has_exit:
                break
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            if terminated.get("__all__", False) or truncated.get("__all__", False):
                env.reset()
            time.sleep(delay)
        return True
    finally:
        if env:
            env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a single environment test.")
    parser.add_argument('--env', type=str, required=True, help="Environment type.")
    parser.add_argument('--body', type=str, required=True, help="Body type.")
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--delay', type=float, default=0.02)
    args = parser.parse_args()

    try:
        success = False
        if args.env in ['stump', 'parkour']:
            success = check_single_agent_env(args.env, args.body, args.steps, args.delay)
        elif args.env == 'marl-cooperative':
            success = check_multi_agent_env('cooperative', args.body, args.steps, args.delay)
        elif args.env == 'marl-interactive':
            success = check_multi_agent_env('interactive', args.body, args.steps, args.delay)

        sys.exit(0 if success else 1)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
