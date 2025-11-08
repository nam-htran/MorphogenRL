# TeachMyAgent/environments/envs/parametric_continuous_parkour.py
import math
import os

import Box2D
import gymnasium as gym
import numpy as np
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef)
from gymnasium import spaces
from gymnasium.utils import seeding, EzPickle

from TeachMyAgent.environments.envs.Box2D_dynamics.water_dynamics import WaterDynamics, WaterContactDetector
from TeachMyAgent.environments.envs.Box2D_dynamics.climbing_dynamics import ClimbingDynamics, ClimbingContactDetector
from TeachMyAgent.environments.envs.PCGAgents.CPPN.cppn_pytorch import CPPN_Pytorch
from TeachMyAgent.environments.envs.bodies.BodiesEnum import BodiesEnum
from TeachMyAgent.environments.envs.bodies.BodyTypesEnum import BodyTypesEnum
from TeachMyAgent.environments.envs.utils.custom_user_data import (
    CustomUserDataObjectTypes, 
    CustomUserData, 
    CustomBodyUserData, 
    CustomMotorUserData, 
    CustomBodySensorUserData
)

class ContactDetector(WaterContactDetector, ClimbingContactDetector):
    def __init__(self, env):
        # Gọi tất cả các hàm __init__ của lớp cha một cách tường minh
        WaterContactDetector.__init__(self)
        ClimbingContactDetector.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        fA, fB = contact.fixtureA, contact.fixtureB
        
        # Kiểm tra an toàn toàn diện: đảm bảo body và userData tồn tại và đúng loại
        if not all(hasattr(f, 'body') and f.body and hasattr(f.body, 'userData') and 
                   isinstance(f.body.userData, CustomUserData) for f in [fA, fB]):
            return

        bodies = [fA.body, fB.body]
        if any(body.userData.object_type == CustomUserDataObjectTypes.WATER for body in bodies):
            WaterContactDetector.BeginContact(self, contact)

    def EndContact(self, contact):
        fA, fB = contact.fixtureA, contact.fixtureB
        
        # Kiểm tra an toàn toàn diện: đảm bảo body và userData tồn tại và đúng loại
        if not all(hasattr(f, 'body') and f.body and hasattr(f.body, 'userData') and 
                   isinstance(f.body.userData, CustomUserData) for f in [fA, fB]):
            return
        
        bodies = [fA.body, fB.body]
        
        is_water_contact = any(
            isinstance(body.userData, CustomUserData) and body.userData.object_type == CustomUserDataObjectTypes.WATER 
            for body in bodies
        )
        is_sensor_contact = any(
            isinstance(body.userData, CustomUserData) and body.userData.object_type == CustomUserDataObjectTypes.BODY_SENSOR 
            for body in bodies
        )

        if is_water_contact:
            WaterContactDetector.EndContact(self, contact)
        elif is_sensor_contact:
            ClimbingContactDetector.EndContact(self, contact)
        else:
            for body in bodies:
                if (isinstance(body.userData, CustomBodyUserData) and 
                    body.userData.object_type == CustomUserDataObjectTypes.BODY_OBJECT and 
                    body.userData.check_contact):
                    body.userData.has_contact = False

    def Reset(self):
        WaterContactDetector.Reset(self)
        ClimbingContactDetector.Reset(self)

class LidarCallback(Box2D.b2.rayCastCallback):
    def __init__(self, agent_mask_filter):
        Box2D.b2.rayCastCallback.__init__(self)
        self.agent_mask_filter = agent_mask_filter
        self.fixture = None
        self.is_water_detected = False
        self.is_creeper_detected = False
    def ReportFixture(self, fixture, point, normal, fraction):
        if (fixture.filterData.categoryBits & self.agent_mask_filter) == 0: return -1
        self.p2 = point
        self.fraction = fraction
        if hasattr(fixture, 'body') and fixture.body and hasattr(fixture.body, 'userData') and isinstance(fixture.body.userData, CustomUserData):
            self.is_water_detected = fixture.body.userData.object_type == CustomUserDataObjectTypes.WATER
            self.is_creeper_detected = fixture.body.userData.object_type == CustomUserDataObjectTypes.SENSOR_GRIP_TERRAIN
        return fraction
        
FPS=50; SCALE=30.0; VIEWPORT_W=600; VIEWPORT_H=400; NB_LIDAR=10; LIDAR_RANGE=160/SCALE; INITIAL_RANDOM=5; TERRAIN_STEP=14/SCALE; TERRAIN_LENGTH=200; TERRAIN_HEIGHT=VIEWPORT_H/SCALE/4; TERRAIN_END=5; INITIAL_TERRAIN_STARTPAD=20; FRICTION=2.5; WATER_DENSITY=1.0; HULL_CONTACT_PENALTY=0.1

class ParametricContinuousParkour(gym.Env, EzPickle):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': FPS}

    def __init__(self, agent_body_type: str, CPPN_weights_path: str = None, input_CPPN_dim: int = 3,
                 terrain_cppn_scale: int = 10, ceiling_offset: int = 200, ceiling_clip_offset: int = 0,
                 lidars_type: str = 'full', water_clip: int = 20, movable_creepers: bool = False,
                 render_mode: str = None, horizon: int = 3000, flip_termination_steps: int = 50, **kwargs):

        super().__init__()
        self.rendering_viewer_w = VIEWPORT_W
        self.rendering_viewer_h = VIEWPORT_H
        self.render_mode = render_mode
        self.horizon = horizon
        self.ts = 0
        self.flip_termination_steps = flip_termination_steps
        
        self.reward_params = kwargs.copy()
        
        default_rewards = {
            'progress_multiplier': 130.0, 'velocity_multiplier': 0.3, 
            'torque_penalty_multiplier': 40.0, 'alive_bonus': 0.0,
            'stagnation_penalty': 0.1, 'grasp_bonus': 0.02,
            'hull_contact_penalty': HULL_CONTACT_PENALTY, 'body_angle_penalty': 0.0,
            'low_hull_penalty': 0.0, 'standing_bonus_multiplier': 0.0,
            'foot_air_time_bonus': 0.0, 'gait_bonus': 0.0
        }
        for key, value in default_rewards.items():
            if key not in self.reward_params:
                self.reward_params[key] = value

        self.np_random = None
        if lidars_type == "down": self.lidar_angle = 1.5; self.lidar_y_offset = 0
        elif lidars_type == "up": self.lidar_angle = 2.3; self.lidar_y_offset = 1.5
        else: self.lidar_angle = np.pi; self.lidar_y_offset = 0

        self.seed()
        self.viewer = None
        self.contact_listener = ContactDetector(self)
        self.world = Box2D.b2World(contactListener=self.contact_listener)
        self.movable_creepers = movable_creepers
        self.climbing_dynamics = ClimbingDynamics()

        body_type = BodiesEnum.get_body_type(agent_body_type)
        
        body_constructor_args = {}
        if body_type in [BodyTypesEnum.SWIMMER, BodyTypesEnum.AMPHIBIAN]:
            body_constructor_args['density'] = WATER_DENSITY
        
        self.agent_body = BodiesEnum[agent_body_type].value(scale=SCALE, **body_constructor_args)

        self.terrain = []
        self.water_dynamics = WaterDynamics(self.world.gravity, max_push=water_clip)
        self.prev_shaping = None
        self.episodic_reward = 0

        self.TERRAIN_STARTPAD = max(INITIAL_TERRAIN_STARTPAD, self.agent_body.AGENT_WIDTH / TERRAIN_STEP + 5)
        self._create_terrain_fixtures()

        self.input_CPPN_dim = input_CPPN_dim
        self.terrain_CPPN = CPPN_Pytorch(x_dim=TERRAIN_LENGTH, input_dim=input_CPPN_dim, output_dim=2)
        weights_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "PCGAgents/CPPN/weights/same_ground_ceiling_cppn_pytorch.pt")
        self.terrain_CPPN.load_tf_weights(weights_path)

        self.set_terrain_cppn_scale(terrain_cppn_scale, ceiling_offset, ceiling_clip_offset)

        default_params = {"input_vector": np.array([-0.25, 0.8, 0.0]), "water_level": 0.1,
                          "creepers_width": 0.25, "creepers_height": 2.0, "creepers_spacing": 1.5}
        self.set_environment(**default_params)

        self._generate_agent()

        agent_action_size = self.agent_body.get_action_size()
        self.action_space = spaces.Box(np.array([-1]*agent_action_size), np.array([1]*agent_action_size), dtype=np.float32)

        total_obs_size = 6 + NB_LIDAR*2 + len(self.agent_body.get_motors_state())
        if self.agent_body.body_type == BodyTypesEnum.CLIMBER:
            total_obs_size += len(self.agent_body.get_sensors_state())
        high = np.array([np.inf]*total_obs_size)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
    
    def _create_terrain_fixtures(self):
        self.fd_polygon = fixtureDef(shape=polygonShape(vertices=[(0,0),(1,0),(1,-1),(0,-1)]), friction=FRICTION)
        self.fd_edge = fixtureDef(shape=edgeShape(vertices=[(0,0),(1,1)]), friction=FRICTION)
        self.fd_water = fixtureDef(shape=polygonShape(vertices=[(0,0),(1,0),(1,-1),(0,-1)]), density=WATER_DENSITY, isSensor=True)
        self.fd_creeper = fixtureDef(shape=polygonShape(vertices=[(0,0),(1,0),(1,-1),(0,-1)]), density=5.0, isSensor=True)

    def _generate_agent(self):
        init_x = TERRAIN_STEP*self.TERRAIN_STARTPAD/2
        if self.agent_body.body_type == BodyTypesEnum.CLIMBER:
            spawn_margin = 0.5
            init_y = TERRAIN_HEIGHT + self.ceiling_offset - (self.agent_body.AGENT_HEIGHT / 2) - spawn_margin
        else:
            spawn_buffer = 2.0
            if hasattr(self, 'terrain_ground_y') and len(self.terrain_ground_y) > int(self.TERRAIN_STARTPAD / 2):
                ground_y = self.terrain_ground_y[int(self.TERRAIN_STARTPAD / 2)]
            else:
                ground_y = TERRAIN_HEIGHT
            init_y = ground_y + self.agent_body.AGENT_CENTER_HEIGHT + spawn_buffer
        self.agent_body.draw(self.world, init_x, init_y, self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed); return [seed]

    def set_terrain_cppn_scale(self, terrain_cppn_scale, ceiling_offset, ceiling_clip_offset):
        assert terrain_cppn_scale > 1; self.TERRAIN_CPPN_SCALE = terrain_cppn_scale
        self.CEILING_LIMIT = 1000 / self.TERRAIN_CPPN_SCALE; self.GROUND_LIMIT = -1000 / self.TERRAIN_CPPN_SCALE
        self.ceiling_offset = ceiling_offset / self.TERRAIN_CPPN_SCALE; self.ceiling_clip_offset = ceiling_clip_offset / self.TERRAIN_CPPN_SCALE

    def set_environment(self, input_vector, water_level, creepers_width=None, creepers_height=None, creepers_spacing=0.1, terrain_cppn_scale=10):
        self.CPPN_input_vector = input_vector; self.water_level = water_level.item() if isinstance(water_level, np.float32) else water_level
        self.water_level = max(0.01, self.water_level); self.creepers_width = creepers_width if creepers_width is not None else creepers_width
        self.creepers_height = creepers_height if creepers_height is not None else creepers_height; self.creepers_spacing = max(0.01, creepers_spacing)
        self.set_terrain_cppn_scale(terrain_cppn_scale, self.ceiling_offset*self.TERRAIN_CPPN_SCALE, self.ceiling_clip_offset*self.TERRAIN_CPPN_SCALE)

    def _destroy(self):
        if not self.world: return
        self.world.contactListener = None
        for t in self.terrain:
            if t: self.world.DestroyBody(t)
        self.terrain = [];
        if self.agent_body: self.agent_body.destroy(self.world)

    def _get_state(self):
        head = self.agent_body.reference_head_object; vel = head.linearVelocity; pos = head.position
        for i in range(NB_LIDAR):
            self.lidar[i].fraction = 1.0; self.lidar[i].p1 = pos
            self.lidar[i].p2 = (pos[0] + math.sin((self.lidar_angle * i / NB_LIDAR + self.lidar_y_offset)) * LIDAR_RANGE,
                               pos[1] - math.cos((self.lidar_angle * i / NB_LIDAR) + self.lidar_y_offset) * LIDAR_RANGE)
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)
        is_under_water = pos.y <= self.water_y; is_agent_dead = False
        state = [head.angle, 2.0 * head.angularVelocity / FPS, 0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS, 0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
                 1.0 if is_under_water else 0.0, 1.0 if is_agent_dead else 0.0]
        state.extend(self.agent_body.get_motors_state())
        if self.agent_body.body_type == BodyTypesEnum.CLIMBER: state.extend(self.agent_body.get_sensors_state())
        surface_dectected = []
        for lidar in self.lidar:
            state.append(lidar.fraction)
            if lidar.is_water_detected: surface_dectected.append(-1)
            elif lidar.is_creeper_detected: surface_dectected.append(1)
            else: surface_dectected.append(0)
        state.extend(surface_dectected)
        return np.array(state, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self._destroy()
        self.world = Box2D.b2World(contactListener=self.contact_listener); self.world.contactListener = self.contact_listener
        if self.contact_listener: self.contact_listener.Reset()
        self.ts = 0; self.critical_contact = False; self.prev_shaping = None; self.episodic_reward = 0; self.scroll = [0.0, 0.0]; self.lidar_render = 0
        self.water_y = self.GROUND_LIMIT; self.nb_steps_outside_water = 0; self.nb_steps_under_water = 0; self.flipped_counter = 0; self.stagnation_counter = 0; self.last_progress_x = 0
        self._generate_terrain(); self._generate_agent()
        self.drawlist = self.terrain + self.agent_body.get_elements_to_render()
        self.lidar = [LidarCallback(self.agent_body.reference_head_object.fixtures[0].filterData.maskBits) for _ in range(NB_LIDAR)]
        if self.agent_body.body_type == BodyTypesEnum.CLIMBER:
            grasp_action = np.ones(self.action_space.shape[0])
            for _ in range(50):
                self.agent_body.activate_motors(grasp_action); self.climbing_dynamics.before_step_climbing_dynamics(grasp_action, self.agent_body, self.world)
                self.world.Step(1.0 / FPS, 6 * 30, 2 * 30); self.climbing_dynamics.after_step_climbing_dynamics(self.world.contactListener, self.world)
        else:
            for _ in range(10): self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.critical_contact = False
        if self.contact_listener: self.contact_listener.Reset()
        for part in self.agent_body.body_parts:
            if hasattr(part.userData, 'has_contact'): part.userData.has_contact = False
        self.prev_pos_x = self.agent_body.reference_head_object.position.x; self.last_progress_x = self.prev_pos_x
        return self._get_state(), {}

    def step(self, action):
        self.ts += 1; is_agent_dead = False
        if hasattr(self.agent_body, "nb_steps_can_survive_outside_water") and self.nb_steps_outside_water > self.agent_body.nb_steps_can_survive_outside_water: is_agent_dead = True
        if hasattr(self.agent_body, "nb_steps_can_survive_under_water") and self.nb_steps_under_water > self.agent_body.nb_steps_can_survive_under_water: is_agent_dead = True
        if is_agent_dead: action = np.array([0] * self.action_space.shape[0])
        
        self.agent_body.activate_motors(action)
        if self.agent_body.body_type == BodyTypesEnum.CLIMBER: self.climbing_dynamics.before_step_climbing_dynamics(action, self.agent_body, self.world)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        if self.agent_body.body_type == BodyTypesEnum.CLIMBER: self.climbing_dynamics.after_step_climbing_dynamics(self.world.contactListener, self.world)
        self.water_dynamics.calculate_forces(self.world.contactListener.fixture_pairs)
        
        state = self._get_state()
        pos = self.agent_body.reference_head_object.position
        vel = self.agent_body.reference_head_object.linearVelocity
        is_under_water = pos.y <= self.water_y
        self.scroll = [pos[0] - self.rendering_viewer_w / SCALE / 5, pos[1] - self.rendering_viewer_h / SCALE / 2.5]
        
        reward = 0
        info = {}

        shaping = self.reward_params['progress_multiplier'] * pos[0] / SCALE 
        if self.prev_shaping is not None: reward += shaping - self.prev_shaping
        self.prev_shaping = shaping
        reward += self.reward_params['velocity_multiplier'] * vel.x
        
        if self.agent_body.body_type == BodyTypesEnum.WALKER and len(self.agent_body.motors) == 4:
            right_foot = self.agent_body.motors[1].userData.contact_body
            left_foot = self.agent_body.motors[3].userData.contact_body
            
            air_time_bonus = 0.0
            if not right_foot.userData.has_contact:
                air_time_bonus += self.reward_params.get('foot_air_time_bonus', 0.0)
            if not left_foot.userData.has_contact:
                air_time_bonus += self.reward_params.get('foot_air_time_bonus', 0.0)
            reward += air_time_bonus
            
            right_hip_speed = self.agent_body.motors[0].speed
            left_hip_speed = self.agent_body.motors[2].speed
            if right_hip_speed * left_hip_speed < 0:
                reward += self.reward_params.get('gait_bonus', 0.0)
        
        if self.reward_params.get('standing_bonus_multiplier', 0.0) > 0 and not is_under_water:
            if len(self.agent_body.motors) >= 2:
                knee_joints = self.agent_body.motors[-2:] 
                knee_straight_bonus = 0.0
                for joint in knee_joints:
                    normalized_angle = (joint.angle - joint.lowerLimit) / (joint.upperLimit - joint.lowerLimit)
                    knee_straight_bonus += normalized_angle
                knee_straight_bonus /= len(knee_joints)
                reward += self.reward_params['standing_bonus_multiplier'] * knee_straight_bonus

        reward -= self.reward_params['body_angle_penalty'] * abs(state[0])
        
        ground_y = TERRAIN_HEIGHT
        current_x_idx = int(pos.x / TERRAIN_STEP)
        if 0 <= current_x_idx < len(self.terrain_ground_y):
            ground_y = self.terrain_ground_y[current_x_idx]
        current_height_from_ground = pos.y - ground_y

        if current_height_from_ground < self.agent_body.AGENT_CENTER_HEIGHT * 0.5 and not is_under_water:
            reward -= self.reward_params['low_hull_penalty']
            
        torque_penalty = sum(self.agent_body.TORQUE_PENALTY * self.reward_params['torque_penalty_multiplier'] * np.clip(np.abs(a), 0, 1) for a in action)
        reward -= torque_penalty
        
        if abs(pos.x - self.last_progress_x) < 0.01: self.stagnation_counter += 1
        else: self.stagnation_counter = 0; self.last_progress_x = pos.x
        if self.stagnation_counter > 100: reward -= self.reward_params['stagnation_penalty']
        
        terminated = False
        is_fall = False
        if abs(state[0]) > 1.5: self.flipped_counter += 1
        else: self.flipped_counter = 0
        
        if self.flipped_counter > self.flip_termination_steps: 
            reward = -100; terminated = True; is_fall = True
        if self.critical_contact or pos[0] < 0 or is_agent_dead: 
            reward = -100; terminated = True; is_fall = True
        if pos[0] > (TERRAIN_LENGTH + self.TERRAIN_STARTPAD - TERRAIN_END) * TERRAIN_STEP: 
            terminated = True
            
        self.episodic_reward += reward
        truncated = self.ts >= self.horizon

        info['progress_x'] = pos[0]
        info['body_height'] = pos[1]
        if terminated or truncated:
            info['is_fall'] = 1.0 if is_fall else 0.0
            info['episode'] = { 'r': self.episodic_reward, 'l': self.ts, 'final_progress_x': pos[0] }

        if self.render_mode == "human": self.render()
        
        return state, reward, terminated, truncated, info

    def _generate_terrain(self):
        self.cloud_poly, self.terrain_x, self.terrain_ground_y, self.terrain_ceiling_y, self.terrain_poly, self.terrain = [], [], [], [], [], []
        y = self.terrain_CPPN.generate(self.CPPN_input_vector) / self.TERRAIN_CPPN_SCALE; ground_y, ceiling_y = y[:, 0], y[:, 1]
        offset = TERRAIN_HEIGHT - ground_y[0]; ground_y = np.add(ground_y, offset); offset = TERRAIN_HEIGHT + self.ceiling_offset - ceiling_y[0]; ceiling_y = np.add(ceiling_y, offset)
        terrain_creepers = []; water_body = None; x = 0; max_x = TERRAIN_LENGTH * TERRAIN_STEP + self.TERRAIN_STARTPAD * TERRAIN_STEP; i = 0
        while x < max_x:
            self.terrain_x.append(x)
            if i < self.TERRAIN_STARTPAD: self.terrain_ground_y.append(TERRAIN_HEIGHT); self.terrain_ceiling_y.append(TERRAIN_HEIGHT + self.ceiling_offset)
            else:
                self.terrain_ground_y.append(ground_y[i - self.TERRAIN_STARTPAD].item())
                ceiling_val = ceiling_y[i - self.TERRAIN_STARTPAD] if ceiling_y[i - self.TERRAIN_STARTPAD] >= ground_y[i - self.TERRAIN_STARTPAD] + self.ceiling_clip_offset else ground_y[i - self.TERRAIN_STARTPAD] + self.ceiling_clip_offset
                self.terrain_ceiling_y.append(ceiling_val.item())
            x += TERRAIN_STEP; i += 1
        space_from_precedent_creeper = self.creepers_spacing
        for i in range(len(self.terrain_x) - 1):
            poly = [(self.terrain_x[i], self.terrain_ground_y[i]), (self.terrain_x[i + 1], self.terrain_ground_y[i + 1])]
            self.fd_edge.shape.vertices = poly; t = self.world.CreateStaticBody(fixtures=self.fd_edge, userData=CustomUserData("grass", CustomUserDataObjectTypes.TERRAIN))
            color = (0.3, 1.0 if (i % 2) == 0 else 0.8, 0.3); t.color1 = color; t.color2 = color; self.terrain.append(t); poly += [(poly[1][0], self.GROUND_LIMIT), (poly[0][0], self.GROUND_LIMIT)]; self.terrain_poly.append((poly, (0.4, 0.6, 0.3)))
            poly = [(self.terrain_x[i], self.terrain_ceiling_y[i]), (self.terrain_x[i + 1], self.terrain_ceiling_y[i + 1])]
            self.fd_edge.shape.vertices = poly; t = self.world.CreateStaticBody(fixtures=self.fd_edge, userData=CustomUserData("rock", CustomUserDataObjectTypes.GRIP_TERRAIN))
            color = (0, 0.25, 0.25); t.color1 = color; t.color2 = color; self.terrain.append(t); poly += [(poly[1][0], self.CEILING_LIMIT), (poly[0][0], self.CEILING_LIMIT)]; self.terrain_poly.append((poly, (0.5, 0.5, 0.5)))
            if self.creepers_width is not None and self.creepers_height is not None and self.creepers_height > 0:
                if space_from_precedent_creeper >= self.creepers_spacing:
                    creeper_height = max(0.2, self.np_random.normal(self.creepers_height, 0.1)); creeper_x = self.terrain_x[i] + TERRAIN_STEP / 2
                    creeper_y = self.terrain_ceiling_y[i]; poly = [(creeper_x - self.creepers_width / 2, creeper_y), (creeper_x + self.creepers_width / 2, creeper_y), (creeper_x + self.creepers_width / 2, creeper_y - creeper_height), (creeper_x - self.creepers_width / 2, creeper_y - creeper_height)]
                    self.fd_creeper.shape.vertices = poly
                    t = self.world.CreateDynamicBody(position=(0, 0), fixtures=self.fd_creeper, userData=CustomUserData("creeper", CustomUserDataObjectTypes.SENSOR_GRIP_TERRAIN)) if self.movable_creepers else self.world.CreateStaticBody(fixtures=self.fd_creeper, userData=CustomUserData("creeper", CustomUserDataObjectTypes.SENSOR_GRIP_TERRAIN))
                    t.color1, t.color2 = (1, 1, 0), (0.8, 0.8, 0); terrain_creepers.append(t); space_from_precedent_creeper = 0
                else: space_from_precedent_creeper += TERRAIN_STEP
        air_max_distance = max(self.terrain_ceiling_y) - self.GROUND_LIMIT; self.water_y = self.GROUND_LIMIT + self.water_level * air_max_distance
        water_poly = [(self.terrain_x[0], self.GROUND_LIMIT), (self.terrain_x[0], self.water_y), (self.terrain_x[len(self.terrain_x) - 1], self.water_y), (self.terrain_x[len(self.terrain_x) - 1], self.GROUND_LIMIT)]
        self.fd_water.shape.vertices = water_poly; t = self.world.CreateStaticBody(fixtures=self.fd_water, userData=CustomUserData("water", CustomUserDataObjectTypes.WATER))
        c = (0.465, 0.676, 0.898); t.color1 = c; t.color2 = c; water_body = t
        self.terrain.extend(terrain_creepers); self.terrain.append(water_body); self.terrain.reverse()
        
    def _SET_RENDERING_VIEWPORT_SIZE(self, width, height, keep_ratio=True):
        self.rendering_viewer_w = width
        if keep_ratio or height is None: self.rendering_viewer_h = int(self.rendering_viewer_w / (VIEWPORT_W / VIEWPORT_H))
        else: self.rendering_viewer_h = height
        
    def close(self):
        if self.viewer: self.viewer.close(); self.viewer = None
        self._destroy()

    def color_agent_head(self, c1, c2):
        ratio = 0
        if hasattr(self.agent_body, "nb_steps_can_survive_outside_water"): ratio = self.nb_steps_outside_water / self.agent_body.nb_steps_can_survive_outside_water
        elif hasattr(self.agent_body, "nb_steps_can_survive_under_water"): ratio = self.nb_steps_under_water / self.agent_body.nb_steps_can_survive_under_water
        return (c1[0] + ratio*(1.0 - c1[0]), c1[1] + ratio*(0.0 - c1[1]), c1[2] + ratio*(0.0 - c1[2])), c2

    def render(self, draw_lidars=True):
        from TeachMyAgent.environments.envs.utils import rendering
        if self.viewer is None and self.render_mode is not None: self.viewer = rendering.Viewer(self.rendering_viewer_w, self.rendering_viewer_h, visible=(self.render_mode == 'human'))
        if self.viewer is None or self.viewer.window is None or self.viewer.window.has_exit:
            if self.render_mode == 'rgb_array': return np.zeros((self.rendering_viewer_h, self.rendering_viewer_w, 3), dtype=np.uint8)
            return None
        self.viewer.set_bounds(self.scroll[0], self.rendering_viewer_w/SCALE + self.scroll[0], self.scroll[1], self.rendering_viewer_h/SCALE + self.scroll[1])
        self.viewer.draw_polygon([(self.scroll[0], self.scroll[1]), (self.scroll[0]+self.rendering_viewer_w/SCALE, self.scroll[1]), (self.scroll[0]+self.rendering_viewer_w/SCALE, self.scroll[1]+self.rendering_viewer_h/SCALE), (self.scroll[0], self.scroll[1]+self.rendering_viewer_h/SCALE)], color=(0.9, 0.9, 1.0))
        for poly,x1,x2 in self.cloud_poly:
            if x2 < self.scroll[0]/2 or x1 > self.scroll[0]/2 + self.rendering_viewer_w/SCALE: continue
            self.viewer.draw_polygon([(p[0]+self.scroll[0]/2, p[1]) for p in poly], color=(1,1,1))
        for obj in self.drawlist:
            color1, color2 = obj.color1, obj.color2
            if hasattr(obj.userData, 'object_type') and obj.userData.object_type == CustomUserDataObjectTypes.BODY_SENSOR and hasattr(obj.userData, 'has_joint') and obj.userData.has_joint: color1, color2 = (1.0, 1.0, 0.0), (1.0, 1.0, 0.0)
            elif obj == self.agent_body.reference_head_object: color1, color2 = self.color_agent_head(color1, color2)
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 30, color=color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 30, color=color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=color1); path.append(path[0]); self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)
        for poly, color in self.terrain_poly:
            if len(poly) < 2 or poly[1][0] < self.scroll[0] or poly[0][0] > self.scroll[0] + self.rendering_viewer_w / SCALE: continue
            self.viewer.draw_polygon(poly, color=color)
        if draw_lidars and hasattr(self, 'lidar'):
            for i in range(len(self.lidar)):
                l = self.lidar[i]; self.viewer.draw_polyline([l.p1, l.p2], color=(1, 0, 0), linewidth=1)
        flagy1 = TERRAIN_HEIGHT; flagy2 = flagy1 + 50/SCALE; x = TERRAIN_STEP*3
        self.viewer.draw_polyline([(x, flagy1), (x, flagy2)], color=(0,0,0), linewidth=2)
        f = [(x, flagy2), (x, flagy2-10/SCALE), (x+25/SCALE, flagy2-5/SCALE)]
        self.viewer.draw_polygon(f, color=(0.9,0.2,0)); self.viewer.draw_polyline(f + [f[0]], color=(0,0,0), linewidth=2)
        return self.viewer.render(return_rgb_array = self.render_mode=='rgb_array')