# scripts/calculate_map_length.py
import argparse
import sys
import numpy as np

def print_header(title):
    """Prints a standardized header."""
    width = 80
    print("\n" + "#"*width + f"\n##{title:^76}##\n" + "#"*width)

def add_calculate_length_args(parser):
    """Adds CLI arguments for the map length calculation."""
    parser.add_argument('--env', type=str, default='parkour', choices=['parkour', 'stump'],
                        help="Select the environment to calculate the length for.")
    return parser

def main(args):
    """
    Calculates the map length based on predefined constants in the environment code.
    This provides an instant estimate without running a simulation.
    """
    print_header(f"CALCULATING MAP LENGTH FOR '{args.env.upper()}'")

    # Import constants directly from the environment files
    if args.env == 'parkour':
        from TeachMyAgent.environments.envs.parametric_continuous_parkour import (
            SCALE, TERRAIN_STEP, TERRAIN_LENGTH, TERRAIN_END, INITIAL_TERRAIN_STARTPAD
        )
    elif args.env == 'stump':
        from TeachMyAgent.environments.envs.parametric_continuous_stump_tracks import (
            SCALE, TERRAIN_STEP, TERRAIN_LENGTH, TERRAIN_END, INITIAL_TERRAIN_STARTPAD
        )
    else:
        print(f"ERROR: Environment '{args.env}' not supported by this script.")
        sys.exit(1)

    # The condition to finish an episode is based on the agent's x-position
    finish_line_x = (TERRAIN_LENGTH + INITIAL_TERRAIN_STARTPAD - TERRAIN_END) * TERRAIN_STEP

    # Estimate required steps. This is a rough heuristic.
    # A bipedal walker's speed varies greatly. We can assume an average of ~0.2 units of distance per step.
    # So, steps = distance / speed_per_step
    estimated_steps = int(finish_line_x / 0.2)

    print("Environment Constants:")
    print(f"  - TERRAIN_LENGTH: {TERRAIN_LENGTH} (steps)")
    print(f"  - TERRAIN_STEP: {TERRAIN_STEP:.4f} (meters per step)")
    print(f"  - TERRAIN_STARTPAD: {INITIAL_TERRAIN_STARTPAD} (steps)")
    print(f"  - TERRAIN_END: {TERRAIN_END} (steps before actual end)")
    print("-" * 80)
    print(f"CALCULATION:")
    print(f"  - Finish Line X-Coordinate = ({TERRAIN_LENGTH} + {INITIAL_TERRAIN_STARTPAD} - {TERRAIN_END}) * {TERRAIN_STEP:.4f}")
    print(f"  - Finish Line at approx. x = {finish_line_x:.2f} meters")
    print("-" * 80)
    print("RECOMMENDATION:")
    print(f"An agent needs to travel approximately {finish_line_x:.2f} meters to finish the map.")
    print(f"A rough estimate of steps required would be around: {estimated_steps} steps.")
    print("\n=> Recommended 'horizon' value in your YAML file: {}".format(int(estimated_steps * 1.2)))
    print("(This includes a 20% buffer. Use 'check_length' command for a more accurate simulation-based value).")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Instantly calculate the length of a procedurally generated map.")
    parser = add_calculate_length_args(parser)
    args = parser.parse_args()
    main(args)