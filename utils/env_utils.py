# utils/env_utils.py
import gymnasium as gym
import numpy as np
import tkinter as tk
from TeachMyAgent.environments.envs.bodies.BodiesEnum import BodiesEnum
from TeachMyAgent.environments.envs.bodies.BodyTypesEnum import BodyTypesEnum

def get_screen_resolution():
    """Get screen resolution."""
    try:
        root = tk.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        return 1280, 720

def collect_env_params(env_key, args):
    """Collect environment parameters from command-line args into a dictionary."""
    params = {}
    if env_key == 'stump':
        if hasattr(args, 'roughness') and args.roughness is not None:
            params['roughness'] = args.roughness
        if hasattr(args, 'stump_height') and args.stump_height is not None:
            params['stump_height'] = tuple(args.stump_height)
        if hasattr(args, 'stump_width') and args.stump_width is not None:
            params['stump_width'] = tuple(args.stump_width)
        if hasattr(args, 'obstacle_spacing') and args.obstacle_spacing is not None:
            params['obstacle_spacing'] = args.obstacle_spacing
    
    elif env_key == 'parkour':
        if hasattr(args, 'input_vector') and args.input_vector is not None:
            params['input_vector'] = np.array(args.input_vector)
        if hasattr(args, 'water_level') and args.water_level is not None:
            params['water_level'] = args.water_level
        if hasattr(args, 'creepers_width') and args.creepers_width is not None:
            params['creepers_width'] = args.creepers_width
        if hasattr(args, 'creepers_height') and args.creepers_height is not None:
            params['creepers_height'] = args.creepers_height
        if hasattr(args, 'creepers_spacing') and args.creepers_spacing is not None:
            params['creepers_spacing'] = args.creepers_spacing
    
    return params

# START CHANGE: Add **kwargs to accept and forward extra arguments
def build_and_setup_env(env_key, body_name, user_params, render_mode=None, args=None, **kwargs):
    """Create and configure the environment with default and user parameters."""
    
    mapping = {
        'stump': ('parametric-continuous-stump-tracks-v0', 'walker_type'),
        'parkour': ('parametric-continuous-parkour-v0', 'agent_body_type'),
    }
    env_id, param_name = mapping.get(env_key, (None, None))
    if not env_id:
        raise ValueError(f"Invalid environment '{env_key}'.")

    env_kwargs = {param_name: body_name}
    
    # Merge any extra keyword arguments (like reward shaping params) into the env constructor arguments
    env_kwargs.update(kwargs)
    
    if render_mode:
        env_kwargs['render_mode'] = render_mode
        
    if args and hasattr(args, 'horizon') and args.horizon is not None:
        env_kwargs['horizon'] = args.horizon

    if env_key == "parkour":
        body_type = BodiesEnum.get_body_type(body_name)
        lidar_map = {BodyTypesEnum.CLIMBER: 'up', BodyTypesEnum.SWIMMER: 'full', BodyTypesEnum.WALKER: 'full'}
        env_kwargs['lidars_type'] = lidar_map.get(body_type, 'down') # 'down' will be fallback
        print(f"Automatically set Lidar to '{env_kwargs['lidars_type']}' for body '{body_name}'.")

        if body_type in [BodyTypesEnum.SWIMMER, BodyTypesEnum.AMPHIBIAN]:
            from TeachMyAgent.environments.envs.parametric_continuous_parkour import WATER_DENSITY
            env_kwargs['density'] = WATER_DENSITY
            print(f"Automatically set density for body '{body_name}'.")
    
    # Now, all collected kwargs are passed to gym.make
    env = gym.make(env_id, **env_kwargs)
    
    if env_key == "stump":
        default_params = {"roughness": 0.0, "stump_height": (0.1, 0.05)}
        default_params.update(user_params)
        env.unwrapped.set_environment(**default_params)
        print("STUMP environment configured with:", default_params)

    elif env_key == "parkour":
        default_params = {
            "input_vector": np.array([-0.25, 0.8, 0.0]),
            "water_level": 0.1,
            "creepers_width": 0.25,
            "creepers_height": 2.0,
            "creepers_spacing": 1.5
        }
        if 'creepers_height' in user_params and user_params['creepers_height'] == 0:
            user_params['creepers_width'] = None

        default_params.update(user_params)
        env.unwrapped.set_environment(**default_params)
        print("PARKOUR environment configured with:", default_params)
        
    return env
# END CHANGE

def setup_render_window(env, args):
    """Compute and apply render window resolution."""
    if hasattr(env.unwrapped, "_SET_RENDERING_VIEWPORT_SIZE"):
        screen_w, screen_h = get_screen_resolution()
        if args.fullscreen:
            render_width, render_height = screen_w, screen_h
        elif args.width and args.height:
            render_width, render_height = args.width, args.height
        else:
            render_height = int(screen_h * 0.8)
            render_width = int(render_height * 16 / 9)
        
        print(f"Render resolution set to: {render_width}x{render_height}")
        env.unwrapped._SET_RENDERING_VIEWPORT_SIZE(render_width, render_height, keep_ratio=False)