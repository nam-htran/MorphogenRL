# scripts/check_all.py
import argparse
import os
import sys
import subprocess
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.shared_args import available_bodies

def print_header(text, char='='):
    width = 80
    print("\n" + char * width)
    print(f"{text:^{width}}")
    print(char * width)

def print_success(text):
    print(f"✅ SUCCESS: {text}")

def print_failure(text):
    print(f"❌ FAILURE: {text}")

def add_check_all_args(parser):
    parser.add_argument('--env', type=str, default='all',
                        choices=['all', 'stump', 'parkour', 'marl-cooperative', 'marl-interactive'],
                        help="Environment type to test.")
    parser.add_argument('--body', type=str, default='all',
                        help="Body type to test. Use 'all' to test all available ones.")
    parser.add_argument('--steps', type=int, default=100,
                        help="Number of random steps to run in each test.")
    parser.add_argument('--delay', type=float, default=0.02,
                        help="Pause duration (seconds) between steps for visualization.")
    return parser

def main(args):
    all_bodies = available_bodies()
    bodies_to_test = all_bodies if args.body == 'all' else [args.body]

    if args.body != 'all' and args.body not in all_bodies:
        print(f"ERROR: Body '{args.body}' not found. Available bodies: {all_bodies}")
        sys.exit(1)

    all_envs = ['stump', 'parkour', 'marl-cooperative', 'marl-interactive']
    envs_to_test = all_envs if args.env == 'all' else [args.env]
        
    test_plan = [(env_key, body_name) for env_key in envs_to_test for body_name in bodies_to_test]
    
    failures = 0
    total_tests = len(test_plan)
    
    print_header(f"STARTING FULL TEST: {total_tests} TESTS")
    
    single_test_script_path = os.path.join(os.path.dirname(__file__), 'run_single_test.py')

    for i, (env_key, body_name) in enumerate(test_plan):
        test_name = f"Env: '{env_key}', Body: '{body_name}'"
        print(f"\n--- Test [{i+1}/{total_tests}]: {test_name} ---")

        command = [
            sys.executable,
            single_test_script_path,
            '--env', env_key,
            '--body', body_name,
            '--steps', str(args.steps),
            '--delay', str(args.delay)
        ]

        child_env = os.environ.copy()
        child_env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            encoding='utf-8',
            errors='replace',
            env=child_env
        )

        if result.returncode == 0:
            print_success(test_name)
            if result.stdout:
                print(result.stdout.strip())
        else:
            print_failure(test_name)
            failures += 1
            print("------- ERROR FROM CHILD PROCESS -------")
            print(result.stdout)
            print(result.stderr)
            print("---------------------------------------")
            
    print_header("TEST SUMMARY")
    if failures == 0:
        print(f"🎉 All {total_tests} tests passed successfully! 🎉")
    else:
        print(f"🔥 {failures} out of {total_tests} tests failed. Please check logs above.")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run full environment and agent tests.")
    parser = add_check_all_args(parser)
    args = parser.parse_args()
    main(args)
