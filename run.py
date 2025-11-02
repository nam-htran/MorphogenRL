# run.py
import argparse
import os
import sys
import yaml
from types import SimpleNamespace
import ray
from typing import Any, Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts import demo as demo_script
from scripts import train_ppo, train_acl, train_marl, watch
from scripts import test_suite as test_suite_script
from scripts import evaluate as evaluate_script
from preprocessing import convert_weight
from scripts import check_all as check_all_script
from utils.seeding import set_seed

def ray_init_and_run(func, args):
    """Initializes Ray and runs a function that requires it."""
    if not ray.is_initialized():
        print("Initializing Ray...")
        ray.init(ignore_reinit_error=True)
    
    try:
        func(args)
    finally:
        if ray.is_initialized():
            ray.shutdown()
            print("Ray has been shut down.")

def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        print(f"ERROR: Config not found at '{config_path}'")
        sys.exit(1)
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def print_header(title: str):
    width = 80
    print("\n\n" + "#"*width + f"\n##{title:^76}##\n" + "#"*width)

def run_task(func: Callable, args_namespace: SimpleNamespace, task_name: str) -> Any:
    print_header(f"START TASK: {task_name.upper()}")
    try:
        is_no_args_func = 'args_namespace' not in func.__code__.co_varnames and func.__code__.co_argcount == 0
        result = func() if is_no_args_func else func(args_namespace)
        print_header(f"FINISHED TASK: {task_name.upper()}")
        return result
    except KeyboardInterrupt:
        print(f"\nINFO: Task '{task_name}' stopped by user.")
        sys.exit(0)
    except Exception as e:
        if e.__class__.__name__ == 'UserInterrupt':
            print(f"\nINFO: Task '{task_name}' stopped by user.")
            sys.exit(0)
        print(f"\nERROR: Task '{task_name}' failed.")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def run_pipeline(args: SimpleNamespace):
    config = load_config(args.config)
    watch_config = config.get('WATCH', {})
    
    default_params = {
        'render': False, 'width': args.width, 'height': args.height, 'fullscreen': args.fullscreen,
        'horizon': 3000, 'roughness': None, 'stump_height': None,
        'stump_width': None, 'obstacle_spacing': None, 'input_vector': None,
        'water_level': None, 'creepers_width': None, 'creepers_height': None,
        'creepers_spacing': None, 'check_env': False, 'shared_policy': False,
        'use_cc': False, 'seed': args.seed
    }
    
    for stage_name in ['PPO', 'ACL', 'MARL']:
        stage_config = config.get(stage_name, {})
        if not stage_config.get('enabled', False):
            continue
        
        final_config = {**default_params, **stage_config}
        stage_args = SimpleNamespace(**final_config)

        checkpoint_path = None
        if stage_name == 'PPO':
            checkpoint_path = run_task(train_ppo.main, stage_args, "PPO Training")
        elif stage_name == 'ACL':
            checkpoint_path = run_task(train_acl.main, stage_args, "ACL Training")
        elif stage_name == 'MARL':
            checkpoint_path = run_task(train_marl.main, stage_args, f"MARL Training ({stage_args.mode})")

        if watch_config.get('enabled', False) and checkpoint_path:
            if stage_name == 'MARL':
                print_header("WATCH MARL (INFO)")
                print("RLlib MARL checkpoints are complex and not directly supported by 'watch.py'.")
                print(f"Best MARL checkpoint from tuning saved at: {checkpoint_path}")
            else:
                watch_args = SimpleNamespace(
                    model_path=checkpoint_path,
                    framework='sb3', num_episodes=5,
                    env=stage_args.env, body=stage_args.body,
                    width=args.width, height=args.height, fullscreen=args.fullscreen,
                    **watch_config, **vars(stage_args)
                )
                run_task(watch.main, watch_args, f"Watch {stage_name}")

    print_header("PIPELINE COMPLETE")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Main entry point for TeachMyAgent.", formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', required=True, help="Available commands")

    ray_commands = ['pipeline', 'train', 'test_suite']

    pipeline_parser = subparsers.add_parser('pipeline', help="Run the full training pipeline from a YAML config file.")
    pipeline_parser.add_argument('--config', type=str, default='configs/main_pipeline.yaml', help="Path to the main pipeline configuration file.")
    pipeline_parser.add_argument('--width', type=int, help="Override window width for watching.")
    pipeline_parser.add_argument('--height', type=int, help="Override window height for watching.")
    pipeline_parser.add_argument('--fullscreen', action='store_true', help="Override fullscreen for watching.")
    pipeline_parser.add_argument('--seed', type=int, default=None, help="Global seed for reproducibility. Overrides YAML config.")
    pipeline_parser.set_defaults(func=run_pipeline)

    train_parser = subparsers.add_parser('train', help="Train a specific algorithm.")
    train_parser.add_argument('--seed', type=int, default=None, help="Global seed for reproducibility.")
    train_subparsers = train_parser.add_subparsers(dest='algorithm', required=True)
    
    ppo_parser = train_subparsers.add_parser('ppo', help="Train PPO (single-agent)."); train_ppo.add_ppo_args(ppo_parser)
    ppo_parser.set_defaults(func=lambda args: run_task(train_ppo.main, args, "PPO Training"))
    
    acl_parser = train_subparsers.add_parser('acl', help="Train with Automatic Curriculum Learning."); train_acl.add_acl_args(acl_parser)
    acl_parser.set_defaults(func=lambda args: run_task(train_acl.main, args, "ACL Training"))
    
    marl_parser = train_subparsers.add_parser('marl', help="Train MARL (multi-agent)."); train_marl.add_marl_args(marl_parser)
    marl_parser.set_defaults(func=lambda args: run_task(train_marl.main, args, "MARL Training"))

    eval_parser = subparsers.add_parser('evaluate', help="Evaluate a trained single-agent model.")
    evaluate_script.add_evaluation_args(eval_parser)
    eval_parser.add_argument('--seed', type=int, default=None, help="Global seed for reproducibility.")
    eval_parser.set_defaults(func=lambda args: run_task(evaluate_script.evaluate_agent, args, "Evaluate Model"))

    watch_parser = subparsers.add_parser('watch', help="Watch a trained single-agent model."); watch.add_watch_args(watch_parser)
    watch_parser.add_argument('--seed', type=int, default=None, help="Global seed for reproducibility.")
    watch_parser.set_defaults(func=lambda args: run_task(watch.main, args, "Watch Model"))
    
    demo_parser = subparsers.add_parser('demo', help="Run a random agent in an environment."); demo_script.add_demo_args(demo_parser)
    demo_parser.add_argument('--seed', type=int, default=None, help="Global seed for reproducibility.")
    demo_parser.set_defaults(func=lambda args: run_task(demo_script.main, args, "Environment Demo"))
    
    check_parser = subparsers.add_parser('check_envs', help="Run comprehensive checks on all environments and bodies."); check_all_script.add_check_all_args(check_parser)
    check_parser.set_defaults(func=lambda args: run_task(check_all_script.main, args, "Environment Check"))
    
    test_parser = subparsers.add_parser('test_suite', help="Run the quick, integrated project test suite.")
    test_parser.set_defaults(func=test_suite_script.main)
    
    convert_parser = subparsers.add_parser('convert_weights', help="Convert legacy TF1 weights to PyTorch.")
    convert_parser.set_defaults(func=convert_weight.convert_tf1_to_pytorch)

    args = parser.parse_args()
    
    # START CHANGE: Smart seed handling
    # Priority: Command line > YAML config > No seed
    seed = args.seed
    if args.command == 'pipeline' and seed is None:
        try:
            config = load_config(args.config)
            for stage_name in ['PPO', 'ACL', 'MARL']:
                stage_config = config.get(stage_name, {})
                if stage_config.get('enabled', False):
                    seed_from_config = stage_config.get('seed')
                    if seed_from_config is not None:
                        print(f"INFO: Using seed '{seed_from_config}' from '{stage_name}' configuration in '{args.config}'.")
                        seed = seed_from_config
                        break # Use the first one found
        except Exception as e:
            print(f"WARNING: Could not load config to check for seed. Reason: {e}")
    
    if seed is not None:
        print(f"INFO: Setting global seed to {seed}.")
        set_seed(seed)
        # Ensure the args object carries the definitive seed value
        args.seed = seed
    # END CHANGE
    
    try:
        if args.command in ray_commands:
            ray_init_and_run(args.func, args)
        else:
            if hasattr(args, 'func'):
                is_no_args_func = 'args' not in args.func.__code__.co_varnames and args.func.__code__.co_argcount == 0
                if is_no_args_func:
                     run_task(args.func, None, args.command.upper())
                else:
                     run_task(args.func, args, args.command.upper())
            else:
                parser.print_help()
    finally:
        if ray.is_initialized():
            ray.shutdown()
            print("Final Ray shutdown completed.")