import math
import numpy as np
import gymnasium as gym
import Box2D
from Box2D.b2 import circleShape
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from .parametric_continuous_parkour import (
    ParametricContinuousParkour, LidarCallback,
    SCALE, FPS, VIEWPORT_W, VIEWPORT_H, NB_LIDAR, LIDAR_RANGE,
    TERRAIN_STEP, TERRAIN_LENGTH, TERRAIN_END, INITIAL_RANDOM,
    TERRAIN_HEIGHT, WATER_DENSITY
)
from .utils.custom_user_data import CustomUserDataObjectTypes, CustomBodyUserData
from TeachMyAgent.environments.envs.bodies.BodiesEnum import BodiesEnum
from TeachMyAgent.environments.envs.bodies.BodyTypesEnum import BodyTypesEnum

class InteractiveMultiAgentParkour(ParametricContinuousParkour, MultiAgentEnv):
    """
    Interactive multi-agent Parkour environment.
    All agents coexist and act in ONE shared physics world.
    """
    def __init__(self, config: dict):
        self.n_agents = config.get("n_agents", 2)
        self.agent_body_type = config.get("agent_body_type", "classic_bipedal")
        
        parent_config = config.copy()
        parent_config.pop("n_agents", None) 
        
        body_type = BodiesEnum.get_body_type(self.agent_body_type)
        if body_type in [BodyTypesEnum.SWIMMER, BodyTypesEnum.AMPHIBIAN]:
            parent_config['density'] = WATER_DENSITY

        super().__init__(**parent_config)

        self.agent_body = None
        self.agent_bodies = {}
        self.prev_shapings = {}

        self._agent_ids = {f"agent_{i}" for i in range(self.n_agents)}
        self.possible_agents = list(self._agent_ids)
        
        self.agents = set()
        
        # Create a temporary single-agent env to correctly get spaces
        temp_env_config = parent_config.copy()
        temp_env_config.pop("render_mode", None)
        temp_env = ParametricContinuousParkour(**temp_env_config)
        
        single_action_space = temp_env.action_space
        single_observation_space_original = temp_env.observation_space
        temp_env.close()

        self.action_space = gym.spaces.Dict({i: single_action_space for i in self.possible_agents})

        original_obs_size = single_observation_space_original.shape[0]
        other_agents_obs_size = (self.n_agents - 1) * 2 # Relative (x,y) of other agents
        single_obs_shape = (original_obs_size + other_agents_obs_size,)
        
        high = np.array([np.inf] * single_obs_shape[0], dtype=np.float32)
        single_observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        
        self.observation_space = gym.spaces.Dict({i: single_observation_space for i in self.possible_agents})
    
    def _destroy(self):
        if not self.world:
            return
            
        self.world.contactListener = None
        
        if self.terrain:
            for t in self.terrain:
                if t: self.world.DestroyBody(t)
        self.terrain = []
        
        if self.agent_body:
            self.agent_body.destroy(self.world)
            self.agent_body = None

        self._destroy_agents()

    def _generate_agent(self):
        # Not used in this multi-agent env, _generate_agents is used instead
        pass

    def _destroy_agents(self):
        if not self.agent_bodies:
            return
        for agent_id, body in self.agent_bodies.items():
            if body:
                body.destroy(self.world)
        self.agent_bodies = {}

    def _generate_agents(self):
        self._destroy_agents()
        self.agent_bodies = {}
        
        body_params = {'scale': SCALE}
        body_type = BodiesEnum.get_body_type(self.agent_body_type)
        if body_type in [BodyTypesEnum.SWIMMER, BodyTypesEnum.AMPHIBIAN]:
            body_params['density'] = WATER_DENSITY
            
        for i in range(self.n_agents):
            agent_id = f"agent_{i}"
            body = BodiesEnum[self.agent_body_type].value(**body_params)
            self.agent_bodies[agent_id] = body
            
            init_x = (TERRAIN_STEP * self.TERRAIN_STARTPAD / 2) + i * 2.5
            init_y = self.terrain_ground_y[int(self.TERRAIN_STARTPAD / 2)] + body.AGENT_CENTER_HEIGHT
            
            body.draw(self.world, init_x, init_y, self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM))

    def reset(self, *, seed=None, options=None):
        super(ParametricContinuousParkour, self).reset(seed=seed)
        self._destroy()
        
        self.world = Box2D.b2World(contactListener=self.contact_listener)

        self.world.contactListener = self.contact_listener
        if self.contact_listener:
            self.contact_listener.Reset()

        self.critical_contact = False
        self.prev_shapings = {agent_id: None for agent_id in self._agent_ids}
        self.scroll = [0.0, 0.0]
        
        self.water_y = self.GROUND_LIMIT
        self.lidar_render = 0
        self.nb_steps_outside_water = 0
        self.nb_steps_under_water = 0
        
        self.lidar = [LidarCallback(None) for _ in range(NB_LIDAR)] 

        self._generate_terrain()
        self._generate_agents()

        # Add this line to correctly populate the active agents for RLlib
        self.agents = self._agent_ids

        self.drawlist = self.terrain.copy()
        for body in self.agent_bodies.values():
            self.drawlist += body.get_elements_to_render()

        self.terminateds = {agent_id: False for agent_id in self._agent_ids}
        self.truncateds = {agent_id: False for agent_id in self._agent_ids}
        self.terminateds["__all__"] = False
        self.truncateds["__all__"] = False

        return self._get_obs(), {}

    def step(self, action_dict):
        for agent_id, action in action_dict.items():
            if agent_id in self.agent_bodies and action is not None and not self.terminateds.get(agent_id, False):
                self.agent_bodies[agent_id].activate_motors(action)

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        rewards = {}
        
        positions = {aid: body.reference_head_object.position for aid, body in self.agent_bodies.items() if not (self.terminateds.get(aid, False) or self.truncateds.get(aid, False))}
        if positions:
            leading_agent_pos_x = max(p[0] for p in positions.values())
            avg_agent_pos_y = sum(p[1] for p in positions.values()) / len(positions)
            self.scroll = [
                leading_agent_pos_x - self.rendering_viewer_w / SCALE / 5,
                avg_agent_pos_y - self.rendering_viewer_h / SCALE / 2.5
            ]
            
        for agent_id in self.possible_agents:
            if self.terminateds.get(agent_id, False) or self.truncateds.get(agent_id, False):
                rewards[agent_id] = 0
                continue
            
            body = self.agent_bodies[agent_id]
            pos = body.reference_head_object.position
            
            shaping = 130 * pos[0] / SCALE
            reward = 0
            if self.prev_shapings.get(agent_id) is not None:
                reward = shaping - self.prev_shapings[agent_id]
            self.prev_shapings[agent_id] = shaping

            if agent_id in action_dict and action_dict[agent_id] is not None:
                for a in action_dict[agent_id]:
                    reward -= body.TORQUE_PENALTY * 80 * np.clip(np.abs(a), 0, 1)

            agent_critical_contact = False
            for part in body.body_parts:
                 if isinstance(part.userData, CustomBodyUserData) and part.userData.is_contact_critical and part.userData.has_contact:
                     agent_critical_contact = True
                     break

            if agent_critical_contact or pos[0] < 0:
                reward = -100
                self.terminateds[agent_id] = True
            if pos[0] > (TERRAIN_LENGTH + self.TERRAIN_STARTPAD - TERRAIN_END) * TERRAIN_STEP:
                self.terminateds[agent_id] = True

            rewards[agent_id] = reward
            
        all_done = all(self.terminateds.get(aid, True) for aid in self._agent_ids)
        self.terminateds["__all__"] = all_done
        self.truncateds["__all__"] = all_done

        obs = self._get_obs()
        
        return obs, rewards, self.terminateds, self.truncateds, {}

    def _get_obs(self):
        all_obs = {}
        agent_positions = {aid: body.reference_head_object.position for aid, body in self.agent_bodies.items()}

        for agent_id in self.possible_agents:
            # If agent is done, return a placeholder observation (zero array)
            if self.terminateds.get(agent_id, False) or self.truncateds.get(agent_id, False):
                all_obs[agent_id] = np.zeros(self.observation_space[agent_id].shape, dtype=np.float32)
                continue

            # If agent is active, compute its observation
            body = self.agent_bodies[agent_id]
            head = body.reference_head_object
            vel = head.linearVelocity
            
            for lidar_callback in self.lidar:
                lidar_callback.agent_mask_filter = head.fixtures[0].filterData.maskBits

            for i in range(NB_LIDAR):
                self.lidar[i].fraction = 1.0
                self.lidar[i].p1 = head.position
                self.lidar[i].p2 = (
                    head.position[0] + math.sin((self.lidar_angle * i / NB_LIDAR) + self.lidar_y_offset) * LIDAR_RANGE,
                    head.position[1] - math.cos((self.lidar_angle * i / NB_LIDAR) + self.lidar_y_offset) * LIDAR_RANGE
                )
                self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

            is_under_water = head.position.y <= self.water_y
            
            state = [
                head.angle, 2.0 * head.angularVelocity / FPS,
                0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,
                0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
                1.0 if is_under_water else 0.0, 0.0 
            ]
            state.extend(body.get_motors_state())

            if body.body_type == BodyTypesEnum.CLIMBER:
                state.extend(body.get_sensors_state())

            for lidar in self.lidar: state.append(lidar.fraction)
            for lidar in self.lidar:
                if lidar.is_water_detected: state.append(-1)
                elif lidar.is_creeper_detected: state.append(1)
                else: state.append(0)
            
            original_obs = np.array(state, dtype=np.float32)

            other_agents_pos = []
            my_pos = agent_positions[agent_id]
            for other_id, other_pos in agent_positions.items():
                if other_id != agent_id:
                    relative_pos = (np.array(other_pos) - np.array(my_pos)) / (LIDAR_RANGE * 2)
                    other_agents_pos.extend(relative_pos.tolist())
            
            final_obs = np.concatenate([original_obs, np.array(other_agents_pos, dtype=np.float32)])
            all_obs[agent_id] = final_obs
            
        return all_obs

    def render(self, mode='human', draw_lidars=True):
        from TeachMyAgent.environments.envs.utils import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.rendering_viewer_w, self.rendering_viewer_h)
        
        self.viewer.set_bounds(self.scroll[0], self.rendering_viewer_w/SCALE + self.scroll[0],
                               self.scroll[1], self.rendering_viewer_h/SCALE + self.scroll[1])

        self.viewer.draw_polygon( [
            (self.scroll[0], self.scroll[1]),
            (self.scroll[0]+self.rendering_viewer_w/SCALE, self.scroll[1]),
            (self.scroll[0]+self.rendering_viewer_w/SCALE, self.scroll[1]+self.rendering_viewer_h/SCALE),
            (self.scroll[0], self.scroll[1]+self.rendering_viewer_h/SCALE),
            ], color=(0.9, 0.9, 1.0) )

        for poly,x1,x2 in self.cloud_poly:
            if x2 < self.scroll[0]/2: continue
            if x1 > self.scroll[0]/2 + self.rendering_viewer_w/SCALE: continue
            self.viewer.draw_polygon( [(p[0]+self.scroll[0]/2, p[1]) for p in poly], color=(1,1,1))

        for obj in self.drawlist:
            color1 = obj.color1
            color2 = obj.color2
            if hasattr(obj.userData, 'object_type') and obj.userData.object_type == CustomUserDataObjectTypes.BODY_SENSOR and obj.userData.has_joint:
                color1 = (1.0, 1.0, 0.0)
                color2 = (1.0, 1.0, 0.0)
            
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 30, color=color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 30, color=color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        for poly, color in self.terrain_poly:
            if len(poly) < 2 or poly[1][0] < self.scroll[0]: continue
            if poly[0][0] > self.scroll[0] + self.rendering_viewer_w / SCALE: continue
            self.viewer.draw_polygon(poly, color=color)

        if draw_lidars and hasattr(self, 'lidar'):
            for i in range(len(self.lidar)):
                l = self.lidar[i]
                self.viewer.draw_polyline([l.p1, l.p2], color=(1, 0, 0), linewidth=1)

        flagy1 = TERRAIN_HEIGHT
        flagy2 = flagy1 + 50/SCALE
        x = TERRAIN_STEP*3
        self.viewer.draw_polyline( [(x, flagy1), (x, flagy2)], color=(0,0,0), linewidth=2 )
        f = [(x, flagy2), (x, flagy2-10/SCALE), (x+25/SCALE, flagy2-5/SCALE)]
        self.viewer.draw_polygon(f, color=(0.9,0.2,0) )
        self.viewer.draw_polyline(f + [f[0]], color=(0,0,0), linewidth=2 )

        return self.viewer.render(return_rgb_array = mode=='rgb_array')