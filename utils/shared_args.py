# shared_args.py
import argparse

def available_bodies():
    """Return a list of available agent bodies."""
    from TeachMyAgent.environments.envs.bodies.BodiesEnum import BodiesEnum
    return [e.name for e in BodiesEnum]

def add_common_args(parser):
    """Add common arguments for environment and agent."""
    parser.add_argument('--env', type=str, default='parkour',
                        choices=['stump', 'parkour'],
                        help="Select environment.")
    parser.add_argument('--body', type=str, default='classic_bipedal',
                        help=f"Select body. Available: {available_bodies()}")
    return parser

def add_environment_args(parser):
    """Add all environment parameters to the ArgumentParser."""
    # Stump environment parameters
    stump_group = parser.add_argument_group('Stump Environment Parameters')
    stump_group.add_argument('--roughness', type=float, help="Terrain roughness.")
    stump_group.add_argument('--stump_height', type=float, nargs=2, metavar=('MEAN', 'STD'),
                             help="Mean and standard deviation of obstacle height.")
    stump_group.add_argument('--stump_width', type=float, nargs=2, metavar=('MEAN', 'STD'),
                             help="Mean and standard deviation of obstacle width.")
    stump_group.add_argument('--obstacle_spacing', type=float, help="Spacing between obstacles.")

    # Parkour environment parameters
    parkour_group = parser.add_argument_group('Parkour Environment Parameters')
    parkour_group.add_argument('--input_vector', type=float, nargs='+',
                               help="Input vector for CPPN terrain generation.")
    parkour_group.add_argument('--water_level', type=float, help="Water level (0.0–1.0).")
    parkour_group.add_argument('--creepers_width', type=float, help="Width of climbing objects.")
    parkour_group.add_argument('--creepers_height', type=float,
                               help="Height of climbing objects (0 = none).")
    parkour_group.add_argument('--creepers_spacing', type=float,
                               help="Spacing between climbing objects.")
    
    return parser

def add_render_args(parser):
    """Add rendering/display parameters to the ArgumentParser."""
    render_group = parser.add_argument_group('Rendering Parameters')
    render_group.add_argument('--width', type=int, help="Window width (manual override).")
    render_group.add_argument('--height', type=int, help="Window height (manual override).")
    render_group.add_argument('--fullscreen', action='store_true', help="Enable fullscreen mode.")
    return parser
