# run.py
import argparse
import os
import sys
import yaml
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts import demo as demo_script
from scripts import train_ppo, train_acl, train_marl, watch
from scripts import test_suite as test_suite_script
from preprocessing import convert_weight
from scripts import record as record_script  # new script

def load_config(config_path):
    if not os.path.exists(config_path):
        print(f"ERROR: Config not found at '{config_path}'")
        sys.exit(1)
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def print_header(title):
    width = 80
    print("\n\n" + "#"*width + f"\n##{title:^76}##\n" + "#"*width)

def run_task(func, args_namespace, task_name):
    print_header(f"START TASK: {task_name.upper()}")
    if args_namespace: print(f"Args: {vars(args_namespace)}")
    try:
        result = func(args_namespace) if args_namespace else func()
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
    config = load_config(args.config)
    global_cfg = {'watch': args.watch, 'WATCH': config.get('WATCH', {})}
    
    for stage_name in ['PPO', 'ACL', 'MARL']:
        if not config.get(stage_name, {}).get('enabled', False):
            continue
        cfg = config[stage_name]

        if stage_name == 'PPO':
            n_envs = 1 if args.render_stages else 4
            train_args = SimpleNamespace(
                run_id=cfg['run_id'], total_timesteps=cfg['total_timesteps'], body=cfg['body'], env=cfg.get('env', 'parkour'),
                save_freq=int(cfg['total_timesteps'] * cfg.get('save_freq_ratio', 1.0)), 
                n_envs=n_envs, render=args.render_stages,
                width=None, height=None, fullscreen=False, roughness=None, stump_height=None, stump_width=None, 
                obstacle_spacing=None, input_vector=None, water_level=None, creepers_width=None, creepers_height=None, creepers_spacing=None
            )
            model_path = run_task(train_ppo.main, train_args, "PPO Training")
            
            if args.watch and model_path:
                watch_args = SimpleNamespace(
                    model_path=model_path, framework='sb3', n_agents=1, body=cfg['body'], env=cfg.get('env', 'parkour'),
                    num_episodes=global_cfg['WATCH'].get('num_episodes', 1), fullscreen=global_cfg['WATCH'].get('fullscreen', False),
                    width=None, height=None, roughness=None, stump_height=None, stump_width=None, obstacle_spacing=None,
                    input_vector=None, water_level=None, creepers_width=None, creepers_height=None, creepers_spacing=None,
                    auto_skip_stuck=True, fast_forward=True
                )
                run_task(watch.main, watch_args, f"Watch {stage_name}")

        elif stage_name == 'ACL':
            train_args = SimpleNamespace(
                run_id=cfg['run_id'], body=cfg['body'], env=cfg.get('env', 'parkour'),
                total_stages=cfg['total_stages'], student_steps_per_stage=cfg['student_steps_per_stage'],
                eval_episodes=cfg.get('eval_episodes', 5), mastery_threshold=cfg.get('mastery_threshold', 150.0),
                difficulty_increment=cfg.get('difficulty_increment', 0.01), render=args.render_stages, fullscreen=False,
                width=None, height=None, roughness=None, stump_height=None, stump_width=None, 
                obstacle_spacing=None, input_vector=None, water_level=None, creepers_width=None, creepers_height=None, creepers_spacing=None
            )
            model_path = run_task(train_acl.main, train_args, "ACL Training")
            
            if args.watch and model_path:
                watch_args = SimpleNamespace(
                    model_path=model_path, framework='sb3', n_agents=1, body=cfg['body'], env=cfg.get('env', 'parkour'),
                    num_episodes=global_cfg['WATCH'].get('num_episodes', 1), fullscreen=global_cfg['WATCH'].get('fullscreen', False),
                    width=None, height=None, roughness=None, stump_height=None, stump_width=None, obstacle_spacing=None,
                    input_vector=None, water_level=None, creepers_width=None, creepers_height=None, creepers_spacing=None,
                    auto_skip_stuck=True, fast_forward=True
                )
                run_task(watch.main, watch_args, f"Watch {stage_name}")

        elif stage_name == 'MARL':
            marl_args = SimpleNamespace(
                run_id=cfg.get('run_id'), mode=cfg['mode'], n_agents=cfg['n_agents'], 
                body=cfg['body'], iterations=cfg['iterations'], num_workers=2, check_env=False,
                width=None, height=None, fullscreen=False
            )
            checkpoint_path = run_task(train_marl.main, marl_args, f"MARL Training ({cfg['mode']})")
            
            if args.watch and checkpoint_path:
                print_header("WATCH MARL (NOT SUPPORTED)")
                print(f"Checkpoint at: {checkpoint_path}")

    print_header("PIPELINE COMPLETE")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Main entry point for TeachMyAgent.", formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', required=True)

    pipeline_parser = subparsers.add_parser('pipeline', help="Run training pipeline from config.")
    pipeline_parser.add_argument('--config', type=str, default='configs/main_pipeline.yaml')
    pipeline_parser.add_argument('--watch', action='store_true')
    pipeline_parser.add_argument('--render-stages', action='store_true')
    pipeline_parser.set_defaults(func=run_pipeline)

    test_parser = subparsers.add_parser('test', help="Run quick integration tests.")
    test_parser.set_defaults(func=lambda args=None: run_task(test_suite_script.main, None, "Test Suite"))

    train_parser = subparsers.add_parser('train', help="Train a specific model.")
    train_subparsers = train_parser.add_subparsers(dest='algorithm', required=True)
    ppo_parser = train_subparsers.add_parser('ppo', help="Train PPO.")
    train_ppo.add_ppo_args(ppo_parser)
    ppo_parser.set_defaults(func=lambda args: run_task(train_ppo.main, args, "PPO Training"))
    acl_parser = train_subparsers.add_parser('acl', help="Train ACL.")
    train_acl.add_acl_args(acl_parser)
    acl_parser.set_defaults(func=lambda args: run_task(train_acl.main, args, "ACL Training"))
    marl_parser = train_subparsers.add_parser('marl', help="Train MARL.")
    train_marl.add_marl_args(marl_parser)
    marl_parser.set_defaults(func=lambda args: run_task(train_marl.main, args, "MARL Training"))

    watch_parser = subparsers.add_parser('watch', help="Watch a trained model.")
    watch.add_watch_args(watch_parser)
    watch_parser.set_defaults(func=lambda args: run_task(watch.main, args, "Watch Model"))

    demo_parser = subparsers.add_parser('demo', help="Run random demo environment.")
    demo_script.add_demo_args(demo_parser)
    demo_parser.set_defaults(func=lambda args: run_task(demo_script.main, args, "Demo"))

    record_parser = subparsers.add_parser('record', help="Record MP4 of trained model.")  # new
    record_script.add_record_args(record_parser)
    record_parser.set_defaults(func=lambda args: run_task(record_script.main, args, "Record Video"))

    convert_parser = subparsers.add_parser('convert', help="Convert TF1 weights to PyTorch.")
    convert_parser.set_defaults(func=lambda args=None: run_task(convert_weight.convert_tf1_to_pytorch, None, "Convert Weights"))

    args = parser.parse_args()
    args.func(args)
