
import argparse
import os
import sys
import yaml
from types import SimpleNamespace
import ray
from typing import Any, Callable, Mapping

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts import demo as demo_script
from scripts import train_ppo, train_acl, train_marl, watch
from scripts import test_suite as test_suite_script
from scripts import evaluate as evaluate_script
from preprocessing import convert_weight
from scripts import check_all as check_all_script
from utils.seeding import set_seed

def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        print(f"ERROR: Config not found at '{config_path}'")
        sys.exit(1)
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def deep_update(original, new_dict):
    for key, value in new_dict.items():
        if key in original and isinstance(original[key], Mapping) and isinstance(value, Mapping):
            deep_update(original[key], value)
        else:
            original[key] = value
    return original

def load_and_merge_configs(base_config_path: str) -> dict:
    print(f"Loading base pipeline config from: {base_config_path}")
    base_config = load_config(base_config_path)
    
    final_config = base_config.copy()
    
    for stage_name, stage_info in base_config.get('STAGES', {}).items():
        if not stage_info.get('enabled', False):
            final_config[stage_name] = {'enabled': False}
            continue

        print(f"Loading configs for stage: {stage_name}")
        
        algo_config_path = stage_info['config_path']
        algo_config = load_config(algo_config_path)
        
        env_config = {}
        if 'env_config_path' in stage_info:
            env_config_path = stage_info['env_config_path']
            env_config = load_config(env_config_path)

        merged_stage_config = deep_update(env_config, algo_config)
        merged_stage_config['enabled'] = True
        final_config[stage_name] = merged_stage_config
        
    return final_config

def ray_init_and_run(func, args):
    if not ray.is_initialized():
        print("Initializing Ray...")
        ray.init(ignore_reinit_error=True)
    
    try:
        return func(args)
    finally:
        if ray.is_initialized():
            ray.shutdown()
            print("Ray has been shut down.")

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
    config = load_and_merge_configs(args.config)
    watch_config = config.get('WATCH', {})
    
    pipeline_settings = config.get('PIPELINE_SETTINGS', {})
    transfer_learning_enabled = pipeline_settings.get('transfer_learning', False)
    pretrained_model_path = None
    if transfer_learning_enabled:
        print_header("TRANSFER LEARNING IS ENABLED IN PIPELINE")
    
    default_params = {
        'render': False, 'width': args.width, 'height': args.height, 'fullscreen': args.fullscreen,
        'check_env': False, 'seed': args.seed,
        'pretrained_model_path': None
    }
    
    for stage_name in ['PPO', 'ACL', 'MARL']:
        stage_config = config.get(stage_name, {})
        if not stage_config.get('enabled', False):
            continue
        
        final_config = {**default_params, **stage_config, **stage_config.get('env_params', {})}
        
        if transfer_learning_enabled and pretrained_model_path:
            if stage_name != 'MARL':
                final_config['pretrained_model_path'] = pretrained_model_path
                print(f"INFO: Stage '{stage_name}' will start from pre-trained model: {pretrained_model_path}")
            else:
                print("INFO: MARL stage will be trained from scratch (transfer learning skipped for MARL).")
            
        stage_args = SimpleNamespace(**final_config)

        checkpoint_path = None
        if stage_name == 'PPO':
            checkpoint_path = run_task(train_ppo.main, stage_args, "PPO Training")
        elif stage_name == 'ACL':
            checkpoint_path = run_task(train_acl.main, stage_args, "ACL Training")
        elif stage_name == 'MARL':
            checkpoint_path = ray_init_and_run(train_marl.main, stage_args)

        if transfer_learning_enabled and checkpoint_path:
            pretrained_model_path = checkpoint_path
            print(f"INFO: Stage '{stage_name}' finished. Model for next stage is now: {pretrained_model_path}")

        if watch_config.get('enabled', False) and checkpoint_path:
            if stage_name == 'MARL':
                print_header("WATCH MARL (INFO)")
                print("To watch the MARL agent, run the following command:")
                print(f"python run.py watch --framework rllib --mode {stage_args.mode} --body {stage_args.body} --model_path {checkpoint_path}")
            else:
                framework = 'sb3'
                watch_args = SimpleNamespace(
                    model_path=checkpoint_path,
                    framework=framework,
                    width=args.width, height=args.height, fullscreen=args.fullscreen,
                    **watch_config, **vars(stage_args)
                )
                run_task(watch.main, watch_args, f"Watch {stage_name}")

    print_header("PIPELINE COMPLETE")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Main entry point for TeachMyAgent.", formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', required=True, help="Available commands")

    pipeline_parser = subparsers.add_parser('pipeline', help="Run the full training pipeline from a YAML config file.")
    pipeline_parser.add_argument('--config', type=str, default='configs/base_pipeline.yaml', help="Path to the base pipeline configuration file.")
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
    marl_parser.set_defaults(func=lambda args: ray_init_and_run(train_marl.main, args))
    
    eval_parser = subparsers.add_parser('evaluate', help="Evaluate a trained single-agent model.")
    evaluate_script.add_evaluation_args(eval_parser)
    eval_parser.add_argument('--seed', type=int, default=None, help="Global seed for reproducibility.")
    eval_parser.set_defaults(func=lambda args: run_task(evaluate_script.evaluate_agent, args, "Evaluate Model"))

    watch_parser = subparsers.add_parser('watch', help="Watch a trained model (SB3 & RLlib)."); watch.add_watch_args(watch_parser)
    watch_parser.add_argument('--seed', type=int, default=None, help="Global seed for reproducibility.")
    watch_parser.set_defaults(func=lambda args: run_task(watch.main, args, "Watch Model"))
    
    demo_parser = subparsers.add_parser('demo', help="Run a random agent in an environment."); demo_script.add_demo_args(demo_parser)
    demo_parser.add_argument('--seed', type=int, default=None, help="Global seed for reproducibility.")
    demo_parser.set_defaults(func=lambda args: run_task(demo_script.main, args, "Environment Demo"))
    
    check_parser = subparsers.add_parser('check_envs', help="Run comprehensive checks on all environments and bodies."); check_all_script.add_check_all_args(check_parser)
    check_parser.add_argument('--seed', type=int, default=None, help="Global seed for reproducibility.")
    check_parser.set_defaults(func=lambda args: run_task(check_all_script.main, args, "Environment Check"))
    
    # START CHANGE: Add --render-fullscreen argument for test_suite
    test_parser = subparsers.add_parser('test_suite', help="Run the quick, integrated project test suite.")
    test_parser.add_argument('--seed', type=int, default=None, help="Global seed for reproducibility.")
    test_parser.add_argument('--test-render', action='store_true', help="Also run a visual test for the rendering window.")
    test_parser.add_argument('--render-fullscreen', action='store_true', help="Use fullscreen for the rendering test.")
    test_parser.set_defaults(func=test_suite_script.main)
    # END CHANGE
    
    convert_parser = subparsers.add_parser('convert_weights', help="Convert legacy TF1 weights to PyTorch.")
    convert_parser.add_argument('--seed', type=int, default=None, help="Global seed for reproducibility.")
    convert_parser.set_defaults(func=convert_weight.convert_tf1_to_pytorch)
    
    args = parser.parse_args()
    
    seed = args.seed
    if args.command == 'pipeline' and seed is None:
        try:
            config = load_and_merge_configs(args.config)
            for stage_name in ['PPO', 'ACL', 'MARL']:
                stage_config = config.get(stage_name, {})
                if stage_config.get('enabled', False):
                    seed_from_config = stage_config.get('seed')
                    if seed_from_config is not None:
                        print(f"INFO: Using seed '{seed_from_config}' from '{stage_name}' configuration.")
                        seed = seed_from_config
                        break 
        except Exception as e:
            print(f"WARNING: Could not load config to check for seed. Reason: {e}")
    
    if seed is not None:
        print(f"INFO: Setting global seed to {seed}.")
        set_seed(seed)
        args.seed = seed
    
    try:
        if hasattr(args, 'func'):
             args.func(args)
        else:
            parser.print_help()
    finally:
        if ray.is_initialized():
            ray.shutdown()
            print("Final Ray shutdown completed.")