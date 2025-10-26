# scripts/record.py
import argparse
import time
import os
import sys
import numpy as np
import cv2  # OpenCV for video recording

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.env_utils import build_and_setup_env, collect_env_params, setup_render_window
from utils.shared_args import add_common_args, add_environment_args, add_render_args
from stable_baselines3 import PPO

# Custom exception to handle user interruption
class UserInterrupt(Exception):
    pass

def add_record_args(parser):
    """Add CLI arguments for recording."""
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model (.zip).")
    
    parser = add_common_args(parser)
    parser = add_environment_args(parser)
    parser = add_render_args(parser)
    
    record_group = parser.add_argument_group('Recording Parameters')
    record_group.add_argument('-o', '--output', type=str, default='recordings/recording.mp4', help="Output MP4 file path.")
    record_group.add_argument('--num_episodes', type=int, default=1, help="Number of episodes to record.")
    record_group.add_argument('--timeout', type=int, default=2000, help="Maximum number of steps per episode.")
    record_group.add_argument('--fps', type=int, default=60, help="FPS of the output video.")
    return parser

def main(args):
    """Main function: load model, simulate, and record video."""
    print(f"--- Starting Video Recording ---\nModel: {args.model_path}\nOutput: {args.output}\n--------------------")

    if not os.path.exists(args.model_path):
        print(f"ERROR: Model not found at '{args.model_path}'")
        sys.exit(1)

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    env = None
    video_writer = None
    try:
        user_params = collect_env_params(args.env, args)
        env = build_and_setup_env(args.env, args.body, user_params, render_mode="rgb_array")
        setup_render_window(env, args)

        print("Loading model...")
        model = PPO.load(args.model_path)
        print("Model loaded successfully.")

        obs, info = env.reset()
        frame = env.render()
        height, width, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.output, fourcc, args.fps, (width, height))
        print(f"Initialized video file: {width}x{height} @ {args.fps} FPS.")
        
        print("\nRecording video... this may take a while and will not open a display window.")

        for i in range(args.num_episodes):
            if i > 0:
                obs, info = env.reset()

            done = False
            ep_len = 0
            print(f"--- Recording Episode {i+1}/{args.num_episodes} ---")
            
            # Write first frame
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(bgr_frame)

            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_len += 1
                
                frame = env.render()
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(bgr_frame)
                
                if ep_len >= args.timeout:
                    print(f"  -> Episode stopped after exceeding {args.timeout} steps (timeout).")
                    done = True

            print(f"Episode {i+1} finished after {ep_len} steps.")

    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C).")
        raise UserInterrupt
    finally:
        if env:
            env.close()
        if video_writer:
            video_writer.release()
            print(f"\n✅ Video saved successfully at: {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Record an MP4 video of a trained agent.")
    parser = add_record_args(parser)
    args = parser.parse_args()
    
    try:
        main(args)
    except UserInterrupt:
        sys.exit(0)