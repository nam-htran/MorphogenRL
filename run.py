# run.py
import argparse
import os
import sys
import yaml
from types import SimpleNamespace
import ray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts import demo as demo_script
from scripts import train_ppo, train_acl, train_marl, watch
from scripts import test_suite as test_suite_script
from preprocessing import convert_weight
from scripts import record as record_script
from scripts import check_all as check_all_script

def load_config(config_path):
    """Loads a YAML configuration file."""
    if not os.path.exists(config_path):
        print(f"ERROR: Config not found at '{config_path}'")
        sys.exit(1)
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def print_header(title):
    """Prints a standardized header to the console."""
    width = 80
    print("\n\n" + "#"*width + f"\n##{title:^76}##\n" + "#"*width)

def run_task(func, args_namespace, task_name):
    """Executes a given function (task) with specified arguments and handles exceptions."""
    print_header(f"START TASK: {task_name.upper()}")
    try:
        result = func(args_namespace) if args_namespace is not None else func()
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

def run_pipeline(args):
    """Runs the main training and evaluation pipeline from a configuration file."""
    config = load_config(args.config)
    watch_config = config.get('WATCH', {})
    
    for stage_name in ['PPO', 'ACL', 'MARL']:
        stage_config = config.get(stage_name, {})
        if not stage_config.get('enabled', False):
            continue
        
        cfg = stage_config
        stage_args = SimpleNamespace(**cfg)

        if not hasattr(stage_args, 'render'): stage_args.render = False
        if not hasattr(stage_args, 'width'): stage_args.width = None
        if not hasattr(stage_args, 'height'): stage_args.height = None
        if not hasattr(stage_args, 'fullscreen'): stage_args.fullscreen = False
        if not hasattr(stage_args, 'horizon'): stage_args.horizon = 3000

        if not hasattr(stage_args, 'roughness'): stage_args.roughness = None
        if not hasattr(stage_args, 'stump_height'): stage_args.stump_height = None
        if not hasattr(stage_args, 'stump_width'): stage_args.stump_width = None
        if not hasattr(stage_args, 'obstacle_spacing'): stage_args.obstacle_spacing = None
        if not hasattr(stage_args, 'input_vector'): stage_args.input_vector = None
        if not hasattr(stage_args, 'water_level'): stage_args.water_level = None
        if not hasattr(stage_args, 'creepers_width'): stage_args.creepers_width = None
        if not hasattr(stage_args, 'creepers_height'): stage_args.creepers_height = None
        if not hasattr(stage_args, 'creepers_spacing'): stage_args.creepers_spacing = None

        checkpoint_path = None
        if stage_name == 'PPO':
            checkpoint_path = run_task(train_ppo.main, stage_args, "PPO Training")
        elif stage_name == 'ACL':
            checkpoint_path = run_task(train_acl.main, stage_args, "ACL Training")
        elif stage_name == 'MARL':
            if not hasattr(stage_args, 'check_env'): stage_args.check_env = False
            if not hasattr(stage_args, 'shared_policy'): stage_args.shared_policy = False
            if not hasattr(stage_args, 'use_cc'): stage_args.use_cc = False
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
    temp_dir = "/tmp/ray_results"
    print(f"Initializing Ray with temporary directory: {temp_dir}")
    os.makedirs(temp_dir, exist_ok=True)
    
    ray.init(ignore_reinit_error=True, _temp_dir=temp_dir, object_store_memory=10**9)
    
    try:
        parser = argparse.ArgumentParser(description="Main entry point for TeachMyAgent.", formatter_class=argparse.RawTextHelpFormatter)
        subparsers = parser.add_subparsers(dest='command', required=True, help="Available commands")

        pipeline_parser = subparsers.add_parser('pipeline', help="Run the full training pipeline from a YAML config file.")
        pipeline_parser.add_argument('--config', type=str, default='configs/main_pipeline.yaml', help="Path to the main pipeline configuration file.")
        pipeline_parser.add_argument('--width', type=int, help="Override window width for watching.")
        pipeline_parser.add_argument('--height', type=int, help="Override window height for watching.")
        pipeline_parser.add_argument('--fullscreen', action='store_true', help="Override fullscreen for watching.")
        pipeline_parser.set_defaults(func=run_pipeline)

        train_parser = subparsers.add_parser('train', help="Train a specific algorithm.")
        train_subparsers = train_parser.add_subparsers(dest='algorithm', required=True)
        
        ppo_parser = train_subparsers.add_parser('ppo', help="Train PPO (single-agent).")
        train_ppo.add_ppo_args(ppo_parser)
        ppo_parser.set_defaults(func=lambda args: run_task(train_ppo.main, args, "PPO Training"))
        
        acl_parser = train_subparsers.add_parser('acl', help="Train with Automatic Curriculum Learning.")
        train_acl.add_acl_args(acl_parser)
        acl_parser.set_defaults(func=lambda args: run_task(train_acl.main, args, "ACL Training"))
        
        marl_parser = train_subparsers.add_parser('marl', help="Train MARL (multi-agent).")
        train_marl.add_marl_args(marl_parser)
        marl_parser.set_defaults(func=lambda args: run_task(train_marl.main, args, "MARL Training"))

        watch_parser = subparsers.add_parser('watch', help="Watch a trained single-agent model.")
        watch.add_watch_args(watch_parser)
        watch_parser.set_defaults(func=lambda args: run_task(watch.main, args, "Watch Model"))

        demo_parser = subparsers.add_parser('demo', help="Run a random agent in an environment.")
        demo_script.add_demo_args(demo_parser)
        demo_parser.set_defaults(func=lambda args: run_task(demo_script.main, args, "Environment Demo"))

        record_parser = subparsers.add_parser('record', help="Record an MP4 video of a trained single-agent model.")
        record_script.add_record_args(record_parser)
        record_parser.set_defaults(func=lambda args: run_task(record_script.main, args, "Record Video"))

        check_parser = subparsers.add_parser('check_envs', help="Run comprehensive checks on all environments and bodies.")
        check_all_script.add_check_all_args(check_parser)
        check_parser.set_defaults(func=lambda args: run_task(check_all_script.main, args, "Environment Check"))

        test_parser = subparsers.add_parser('test_suite', help="Run the quick, integrated project test suite.")
        test_parser.set_defaults(func=lambda args=None: run_task(test_suite_script.main, None, "Project Test Suite"))

        convert_parser = subparsers.add_parser('convert_weights', help="Convert legacy TF1 weights to PyTorch.")
        convert_parser.set_defaults(func=lambda args=None: run_task(convert_weight.convert_tf1_to_pytorch, None, "Convert Weights"))

        args = parser.parse_args()
        if hasattr(args, 'func'):
            args.func(args)
        else:
            parser.print_help()

    finally:
        if ray.is_initialized():
            ray.shutdown()
            print("Ray has been shut down.")