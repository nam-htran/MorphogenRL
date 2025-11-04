import math
import numpy as np
import gymnasium as gym
import Box2D
from Box2D.b2 import circleShape, polygonShape, fixtureDef, revoluteJointDef, prismaticJointDef
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from .parametric_continuous_parkour import (
    ParametricContinuousParkour, LidarCallback,
    SCALE, FPS, VIEWPORT_W, VIEWPORT_H, NB_LIDAR, LIDAR_RANGE,
    TERRAIN_STEP, TERRAIN_LENGTH, TERRAIN_END, INITIAL_RANDOM,
    TERRAIN_HEIGHT, WATER_DENSITY
)
from .utils.custom_user_data import CustomUserDataObjectTypes, CustomBodyUserData, CustomUserData

from TeachMyAgent.environments.envs.bodies.BodiesEnum import BodiesEnum
from TeachMyAgent.environments.envs.bodies.BodyTypesEnum import BodyTypesEnum

DOOR_BUTTON_X_POS = TERRAIN_STEP * (TERRAIN_LENGTH / 4)
DOOR_WIDTH = TERRAIN_STEP * 2
DOOR_HEIGHT = TERRAIN_HEIGHT / 2
BUTTON_WIDTH = TERRAIN_STEP
BUTTON_HEIGHT = 0.5

class InteractiveMultiAgentParkour(ParametricContinuousParkour, MultiAgentEnv):
    """Interactive multi-agent parkour environment."""
    def __init__(self, config: dict):
        self.n_agents = config.get("n_agents", 2)
        self.agent_body_type = config.get("agent_body_type", "classic_bipedal")
        self.horizon = config.get("horizon", 1500)
        self.reward_type = config.get("reward_type", "individual")
        self.ts = 0

        parent_config = config.copy()
        parent_config.pop("n_agents", None)
        
        temp_env_config = parent_config.copy()
        temp_env_config.pop("render_mode", None)
        temp_env_config.pop("reward_type", None) 
        
        temp_env = ParametricContinuousParkour(**temp_env_config)
        single_action_space = temp_env.action_space
        single_observation_space_original = temp_env.observation_space
        temp_env.close()

        parent_config.pop("reward_type", None)

        ParametricContinuousParkour.__init__(self, **parent_config)
        MultiAgentEnv.__init__(self)
        
        self.agent_bodies = {}
        self.prev_shapings = {}
        self.possible_agents = [f"agent_{i}" for i in range(self.n_agents)]
        self._agent_ids = set(self.possible_agents)
        self.agents = []
        
        self.button = None; self.door = None; self.door_joint = None
        self.button_pressed = False

        self.action_space = gym.spaces.Dict({i: single_action_space for i in self.possible_agents})

        original_obs_size = single_observation_space_original.shape[0]
        other_agents_obs_size = (self.n_agents - 1) * 2 + 4
        single_obs_shape = (original_obs_size + other_agents_obs_size,)
        high = np.array([np.inf] * single_obs_shape[0], dtype=np.float32)
        single_observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.observation_space = gym.spaces.Dict({i: single_observation_space for i in self.possible_agents})
        
        self._obs_space_in_preferred_format = True
        self._action_space_in_preferred_format = True

    def _destroy(self):
        if not self.world: return
        self.world.contactListener = None
        if self.terrain:
            for t in self.terrain:
                if t: self.world.DestroyBody(t)
        self.terrain = []
        if self.button: self.world.DestroyBody(self.button); self.button = None
        if self.door: self.world.DestroyBody(self.door); self.door = None
        self.door_joint = None
        self._destroy_agents()
    
    def _generate_coop_mechanism(self):
        button_y = self.terrain_ground_y[int(DOOR_BUTTON_X_POS / TERRAIN_STEP)] + BUTTON_HEIGHT/2
        self.button = self.world.CreateStaticBody(position=(DOOR_BUTTON_X_POS - 5, button_y),
                                                  shapes=polygonShape(box=(BUTTON_WIDTH/2, BUTTON_HEIGHT/2)),
                                                  userData=CustomUserData("button", CustomUserDataObjectTypes.TERRAIN))
        self.button.color1, self.button.color2 = (0.9, 0.1, 0.1), (0.7, 0.1, 0.1)

        door_anchor = self.world.CreateStaticBody(position=(DOOR_BUTTON_X_POS, TERRAIN_HEIGHT + DOOR_HEIGHT))
        self.door = self.world.CreateDynamicBody(position=(DOOR_BUTTON_X_POS, TERRAIN_HEIGHT + DOOR_HEIGHT / 2),
                                                 fixtures=fixtureDef(shape=polygonShape(box=(DOOR_WIDTH/2, DOOR_HEIGHT/2)), density=5.0, friction=0.5),
                                                 userData=CustomUserData("door", CustomUserDataObjectTypes.TERRAIN))
        self.door.color1, self.door.color2 = (0.2, 0.2, 0.8), (0.1, 0.1, 0.6)

        self.door_joint = self.world.CreatePrismaticJoint(bodyA=door_anchor, bodyB=self.door, anchor=(DOOR_BUTTON_X_POS, TERRAIN_HEIGHT + DOOR_HEIGHT),
                                                          axis=(0, 1), lowerTranslation=0, upperTranslation=DOOR_HEIGHT * 1.5,
                                                          enableLimit=True, maxMotorForce=2000.0, motorSpeed=0.0, enableMotor=True)

    def _generate_agent(self): pass

    def _destroy_agents(self):
        if not self.agent_bodies: return
        for agent_id, body in self.agent_bodies.items():
            if body: body.destroy(self.world)
        self.agent_bodies = {}

    # START FIX: Add a safe method to destroy specific agent bodies during simulation
    def _destroy_agents_in_simulation(self, agent_ids_to_destroy):
        """Safely destroys the bodies of terminated agents."""
        if not agent_ids_to_destroy:
            return
        for agent_id in agent_ids_to_destroy:
            if agent_id in self.agent_bodies and self.agent_bodies[agent_id]:
                self.agent_bodies[agent_id].destroy(self.world)
                self.agent_bodies[agent_id] = None # Mark as destroyed
    # END FIX

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
        
        self.ts = 0; self.world = Box2D.b2World(contactListener=self.contact_listener); self.world.contactListener = self.contact_listener
        if self.contact_listener: self.contact_listener.Reset()
        self.critical_contact = False; self.prev_shapings = {agent_id: None for agent_id in self.possible_agents}
        self.scroll = [0.0, 0.0]; self.water_y = self.GROUND_LIMIT; self.lidar = [LidarCallback(None) for _ in range(NB_LIDAR)] 

        self._generate_terrain(); self._generate_coop_mechanism(); self._generate_agents()
        
        self.agents = list(self.possible_agents); self.drawlist = self.terrain.copy()
        if self.button: self.drawlist.append(self.button)
        if self.door: self.drawlist.append(self.door)
        for body in self.agent_bodies.values(): self.drawlist += body.get_elements_to_render()
        self.terminateds = {agent_id: False for agent_id in self.possible_agents}; self.truncateds = {agent_id: False for agent_id in self.possible_agents}
        self.terminateds["__all__"] = False; self.truncateds["__all__"] = False
        
        return self._get_obs(), {}

    def step(self, action_dict):
        self.ts += 1

        # Handle button/door mechanism
        self.button_pressed = False
        if self.button:
            for agent_id in self.agents:
                body = self.agent_bodies[agent_id]
                if body: # Check if body exists
                    for part in body.body_parts:
                        for contact in part.contacts:
                            if contact.other == self.button: self.button_pressed = True; break
                        if self.button_pressed: break
                if self.button_pressed: break
        
        if self.door_joint: self.door_joint.motorSpeed = 5.0 if self.button_pressed else -5.0

        # Apply actions
        for agent_id, action in action_dict.items():
            if agent_id in self.agents and self.agent_bodies.get(agent_id) and action is not None and len(action) > 0:
                self.agent_bodies[agent_id].activate_motors(action)

        # Step physics world
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        
        rewards = {}
        positions = {aid: self.agent_bodies[aid].reference_head_object.position for aid in self.agents if self.agent_bodies.get(aid)}
        if positions:
            leading_agent_pos_x = max(p[0] for p in positions.values()); avg_agent_pos_y = sum(p[1] for p in positions.values()) / len(positions)
            self.scroll = [leading_agent_pos_x - self.rendering_viewer_w / SCALE / 5, avg_agent_pos_y - self.rendering_viewer_h / SCALE / 2.5]
            
        # START FIX: Improved termination logic for stability
        agents_terminated_this_step = []
        active_agents_before_step = list(self.agents)

        # Calculate rewards and check for termination conditions
        for agent_id in active_agents_before_step:
            body = self.agent_bodies.get(agent_id)
            if not body:
                continue

            # Calculate rewards (individual or shared)
            if self.reward_type == "individual":
                pos = body.reference_head_object.position
                shaping = 130 * pos[0] / SCALE
                reward = 0
                if self.prev_shapings.get(agent_id) is not None:
                    reward = shaping - self.prev_shapings[agent_id]
                self.prev_shapings[agent_id] = shaping
                reward += 0.01
                if agent_id in action_dict and action_dict[agent_id] is not None:
                    for a in action_dict[agent_id]:
                        reward -= body.TORQUE_PENALTY * 20 * np.clip(np.abs(a), 0, 1)
                rewards[agent_id] = reward
            
            # Check for individual agent termination
            pos = body.reference_head_object.position
            agent_critical_contact = False
            for part in body.body_parts:
                 if isinstance(part.userData, CustomBodyUserData) and part.userData.is_contact_critical and part.userData.has_contact:
                     is_door_contact = False
                     if self.door:
                        for contact in part.contacts:
                            if contact.other == self.door: is_door_contact = True; break
                     if not is_door_contact: agent_critical_contact = True; break
            
            if agent_critical_contact or pos[0] < 0:
                rewards[agent_id] = -100
                self.terminateds[agent_id] = True
            if pos[0] > (TERRAIN_LENGTH + self.TERRAIN_STARTPAD - TERRAIN_END) * TERRAIN_STEP:
                self.terminateds[agent_id] = True
            
            if self.terminateds.get(agent_id, False):
                agents_terminated_this_step.append(agent_id)

        if self.reward_type == "shared":
            agent_progress = []
            for agent_id in active_agents_before_step:
                body = self.agent_bodies.get(agent_id)
                if body:
                    pos = body.reference_head_object.position
                    shaping = 130 * pos[0] / SCALE
                    progress = 0
                    if self.prev_shapings.get(agent_id) is not None:
                        progress = shaping - self.prev_shapings[agent_id]
                    self.prev_shapings[agent_id] = shaping
                    agent_progress.append(progress)
            
            shared_reward = min(agent_progress) if agent_progress else 0
            total_torque_penalty = 0
            for agent_id in active_agents_before_step:
                 body = self.agent_bodies.get(agent_id)
                 if body and agent_id in action_dict and action_dict[agent_id] is not None:
                    for a in action_dict[agent_id]:
                        total_torque_penalty += body.TORQUE_PENALTY * 20 * np.clip(np.abs(a), 0, 1)

            num_active = len(active_agents_before_step)
            avg_torque_penalty = total_torque_penalty / num_active if num_active > 0 else 0
            final_reward = shared_reward - avg_torque_penalty + 0.01
            for agent_id in active_agents_before_step:
                rewards[agent_id] = final_reward
        
        # Safely remove terminated agents from simulation and active list
        if agents_terminated_this_step:
            self._destroy_agents_in_simulation(agents_terminated_this_step)
            for agent_id in agents_terminated_this_step:
                if agent_id in self.agents:
                    self.agents.remove(agent_id)
        
        is_truncated = self.ts >= self.horizon
        all_done = not self.agents or is_truncated
        self.terminateds["__all__"] = all_done and not is_truncated
        self.truncateds["__all__"] = is_truncated
        # END FIX

        obs = self._get_obs()
        return obs, rewards, self.terminateds, self.truncateds, {}

    def _get_obs(self):
        all_obs = {}
        agent_positions = {aid: body.reference_head_object.position for aid, body in self.agent_bodies.items() if body}
        
        for agent_id in self.agents:
            body = self.agent_bodies.get(agent_id)
            if not body: continue # Skip terminated agents

            head = body.reference_head_object
            vel = head.linearVelocity
            my_pos = agent_positions[agent_id]

            for lidar_callback in self.lidar:
                lidar_callback.agent_mask_filter = head.fixtures[0].filterData.maskBits

            for i in range(NB_LIDAR):
                self.lidar[i].fraction = 1.0
                self.lidar[i].p1 = head.position
                self.lidar[i].p2 = (head.position[0] + math.sin((self.lidar_angle * i / NB_LIDAR) + self.lidar_y_offset) * LIDAR_RANGE,
                                   head.position[1] - math.cos((self.lidar_angle * i / NB_LIDAR) + self.lidar_y_offset) * LIDAR_RANGE)
                self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

            is_under_water = head.position.y <= self.water_y
            state = [head.angle, 2.0 * head.angularVelocity / FPS, 0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,
                     0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS, 1.0 if is_under_water else 0.0, 0.0]
            state.extend(body.get_motors_state())
            if body.body_type == BodyTypesEnum.CLIMBER: state.extend(body.get_sensors_state())
            
            for lidar in self.lidar: state.append(lidar.fraction)
            for lidar in self.lidar:
                if lidar.is_water_detected: state.append(-1)
                elif lidar.is_creeper_detected: state.append(1)
                else: state.append(0)
            
            original_obs = np.array(state)
            
            other_agents_pos = []
            for other_id in self.possible_agents:
                if other_id != agent_id:
                    if other_id in agent_positions: # Check if other agent is still active
                        other_pos = agent_positions[other_id]
                        relative_pos = (np.array(other_pos) - np.array(my_pos)) / (LIDAR_RANGE * 2)
                        other_agents_pos.extend(relative_pos.tolist())
                    else:
                        other_agents_pos.extend([0.0, 0.0])

            coop_mechanism_obs = []
            if self.door: coop_mechanism_obs.extend(((np.array(self.door.position) - np.array(my_pos)) / (LIDAR_RANGE*2)).tolist())
            else: coop_mechanism_obs.extend([0.0, 0.0])
            if self.button: coop_mechanism_obs.extend(((np.array(self.button.position) - np.array(my_pos)) / (LIDAR_RANGE*2)).tolist())
            else: coop_mechanism_obs.extend([0.0, 0.0])

            final_obs = np.concatenate([original_obs, np.array(other_agents_pos), np.array(coop_mechanism_obs)])
            all_obs[agent_id] = final_obs.astype(np.float32)
            
        return all_obs
    
    def render(self, draw_lidars=True):
        from TeachMyAgent.environments.envs.utils import rendering
        if self.viewer is None and self.render_mode is not None:
            self.viewer = rendering.Viewer(self.rendering_viewer_w, self.rendering_viewer_h, visible=(self.render_mode == 'human'))
        
        if self.viewer is None or self.viewer.window is None or self.viewer.window.has_exit:
            if self.render_mode == 'rgb_array':
                return np.zeros((self.rendering_viewer_h, self.rendering_viewer_w, 3), dtype=np.uint8)
            return None

        self.viewer.set_bounds(self.scroll[0], self.rendering_viewer_w / SCALE + self.scroll[0],
                               self.scroll[1], self.rendering_viewer_h / SCALE + self.scroll[1])
        
        self.viewer.draw_polygon([
            (self.scroll[0], self.scroll[1]),
            (self.scroll[0] + self.rendering_viewer_w / SCALE, self.scroll[1]),
            (self.scroll[0] + self.rendering_viewer_w / SCALE, self.scroll[1] + self.rendering_viewer_h / SCALE),
            (self.scroll[0], self.scroll[1] + self.rendering_viewer_h / SCALE),
        ], color=(0.9, 0.9, 1.0))

        for poly, x1, x2 in self.cloud_poly:
            if x2 < self.scroll[0] / 2 or x1 > self.scroll[0] / 2 + self.rendering_viewer_w / SCALE:
                continue
            self.viewer.draw_polygon([(p[0] + self.scroll[0] / 2, p[1]) for p in poly], color=(1, 1, 1))

        for obj in self.drawlist:
            # Check if object still exists in the physics world before rendering
            if not obj:
                continue

            is_an_agent_head = False
            for agent_id, body in self.agent_bodies.items():
                if body and obj == body.reference_head_object:
                    is_an_agent_head = True
                    break
            
            color1, color2 = obj.color1, obj.color2
            if hasattr(obj.userData, 'object_type') and obj.userData.object_type == CustomUserDataObjectTypes.BODY_SENSOR and hasattr(obj.userData, 'has_joint') and obj.userData.has_joint:
                color1, color2 = (1.0, 1.0, 0.0), (1.0, 1.0, 0.0)
            elif is_an_agent_head:
                pass

            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 30, color=color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 30, color=color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=color2, linewidth=2)

        for poly, color in self.terrain_poly:
            if len(poly) < 2 or poly[1][0] < self.scroll[0] or poly[0][0] > self.scroll[0] + self.rendering_viewer_w / SCALE:
                continue
            self.viewer.draw_polygon(poly, color=color)

        if draw_lidars and hasattr(self, 'lidar'):
            for i in range(len(self.lidar)):
                l = self.lidar[i]
                self.viewer.draw_polyline([l.p1, l.p2], color=(1, 0, 0), linewidth=1)

        flagy1 = TERRAIN_HEIGHT
        flagy2 = flagy1 + 50 / SCALE
        x = TERRAIN_STEP * 3
        self.viewer.draw_polyline([(x, flagy1), (x, flagy2)], color=(0, 0, 0), linewidth=2)
        f = [(x, flagy2), (x, flagy2 - 10 / SCALE), (x + 25 / SCALE, flagy2 - 5 / SCALE)]
        self.viewer.draw_polygon(f, color=(0.9, 0.2, 0))
        self.viewer.draw_polyline(f + [f[0]], color=(0, 0, 0), linewidth=2)

        return self.viewer.render(return_rgb_array=self.render_mode == 'rgb_array')