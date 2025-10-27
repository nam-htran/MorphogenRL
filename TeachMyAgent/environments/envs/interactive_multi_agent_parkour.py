# TeachMyAgent/environments/envs/interactive_multi_agent_parkour.py
import math
import numpy as np
import gymnasium as gym
import Box2D
from Box2D.b2 import circleShape, polygonShape,fixtureDef, revoluteJointDef, prismaticJointDef
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

# === START: ĐỊNH NGHĨA CHO CỬA VÀ NÚT BẤM ===
DOOR_BUTTON_X_POS = TERRAIN_STEP * (TERRAIN_LENGTH / 2) # Vị trí của cơ chế
DOOR_WIDTH = TERRAIN_STEP * 2
DOOR_HEIGHT = TERRAIN_HEIGHT / 2
BUTTON_WIDTH = TERRAIN_STEP
BUTTON_HEIGHT = 0.5
# === END: ĐỊNH NGHĨA CHO CỬA VÀ NÚT BẤM ===

class InteractiveMultiAgentParkour(ParametricContinuousParkour, MultiAgentEnv):
    """
    Môi trường Parkour đa tác nhân tương tác.
    """
    def __init__(self, config: dict):
        # 1. Chuẩn bị các tham số
        self.n_agents = config.get("n_agents", 2)
        self.agent_body_type = config.get("agent_body_type", "classic_bipedal")
        self.horizon = config.get("horizon", 1500)
        self.ts = 0

        parent_config = config.copy()
        parent_config.pop("n_agents", None)
        
        # 2. Tạo một môi trường tạm thời chỉ để lấy không gian obs/act chính xác
        temp_env_config = parent_config.copy()
        temp_env_config.pop("render_mode", None) # Không cần render cho env tạm
        temp_env = ParametricContinuousParkour(**temp_env_config)
        single_action_space = temp_env.action_space
        single_observation_space_original = temp_env.observation_space
        temp_env.close() # Hủy môi trường tạm sau khi lấy xong thông tin

        # 3. Khởi tạo lớp cha ParametricContinuousParkour cho 'self'
        ParametricContinuousParkour.__init__(self, **parent_config)
        
        # 4. Khởi tạo lớp cha MultiAgentEnv
        MultiAgentEnv.__init__(self)

        # 5. Thiết lập các thuộc tính cho multi-agent và ghi đè spaces
        self.agent_bodies = {}
        self.prev_shapings = {}
        self.possible_agents = [f"agent_{i}" for i in range(self.n_agents)]
        self._agent_ids = set(self.possible_agents)
        self.agents = []
        
        # === START: THIẾT LẬP CHO CỬA VÀ NÚT BẤM ===
        self.button = None
        self.door = None
        self.door_joint = None
        self.button_pressed = False
        # === END: THIẾT LẬP CHO CỬA VÀ NÚT BẤM ===

        # Ghi đè action_space thành Dict
        self.action_space = gym.spaces.Dict({i: single_action_space for i in self.possible_agents})

        # Ghi đè observation_space thành Dict với kích thước đã mở rộng
        original_obs_size = single_observation_space_original.shape[0]
        # Thêm 2 float cho vị trí tương đối của cửa, 2 float cho vị trí của nút bấm
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
        # === START: HỦY CỬA VÀ NÚT BẤM ===
        if self.button: self.world.DestroyBody(self.button)
        if self.door: self.world.DestroyBody(self.door)
        self.button = None
        self.door = None
        self.door_joint = None
        # === END: HỦY CỬA VÀ NÚT BẤM ===
        self._destroy_agents()
    
    # === START: TẠO CỬA VÀ NÚT BẤM ===
    def _generate_coop_mechanism(self):
        # Tạo nút bấm
        button_y = self.terrain_ground_y[int(DOOR_BUTTON_X_POS / TERRAIN_STEP)] + BUTTON_HEIGHT/2
        self.button = self.world.CreateStaticBody(
            position=(DOOR_BUTTON_X_POS - 5, button_y),
            shapes=polygonShape(box=(BUTTON_WIDTH/2, BUTTON_HEIGHT/2)),
            userData=CustomUserData("button", CustomUserDataObjectTypes.TERRAIN) # Dùng lại userData có sẵn
        )
        self.button.color1 = (0.9, 0.1, 0.1) # Màu đỏ
        self.button.color2 = (0.7, 0.1, 0.1)

        # Tạo cửa
        door_anchor = self.world.CreateStaticBody(position=(DOOR_BUTTON_X_POS, TERRAIN_HEIGHT + DOOR_HEIGHT))
        self.door = self.world.CreateDynamicBody(
            position=(DOOR_BUTTON_X_POS, TERRAIN_HEIGHT + DOOR_HEIGHT / 2),
            fixtures=fixtureDef(
                shape=polygonShape(box=(DOOR_WIDTH/2, DOOR_HEIGHT/2)),
                density=5.0,
                friction=0.5,
            ),
            userData=CustomUserData("door", CustomUserDataObjectTypes.TERRAIN)
        )
        self.door.color1 = (0.2, 0.2, 0.8) # Màu xanh
        self.door.color2 = (0.1, 0.1, 0.6)

        # Tạo khớp trượt cho cửa
        self.door_joint = self.world.CreatePrismaticJoint(
            bodyA=door_anchor,
            bodyB=self.door,
            anchor=(DOOR_BUTTON_X_POS, TERRAIN_HEIGHT + DOOR_HEIGHT),
            axis=(0, 1),
            lowerTranslation=0,
            upperTranslation=DOOR_HEIGHT * 1.5,
            enableLimit=True,
            maxMotorForce=2000.0,
            motorSpeed=0.0,
            enableMotor=True,
        )
    # === END: TẠO CỬA VÀ NÚT BẤM ===

    def _generate_agent(self):
        pass

    def _destroy_agents(self):
        if not self.agent_bodies: return
        for agent_id, body in self.agent_bodies.items():
            if body: body.destroy(self.world)
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
        
        self.ts = 0
        self.world = Box2D.b2World(contactListener=self.contact_listener)
        self.world.contactListener = self.contact_listener
        if self.contact_listener: self.contact_listener.Reset()
        self.critical_contact = False
        self.prev_shapings = {agent_id: None for agent_id in self.possible_agents}
        self.scroll = [0.0, 0.0]
        self.water_y = self.GROUND_LIMIT
        self.lidar = [LidarCallback(None) for _ in range(NB_LIDAR)] 

        self._generate_terrain()
        # === START: TẠO CƠ CHẾ HỢP TÁC ===
        self._generate_coop_mechanism()
        # === END: TẠO CƠ CHẾ HỢP TÁC ===
        self._generate_agents()
        
        self.agents = list(self.possible_agents)
        self.drawlist = self.terrain.copy()
        # === START: THÊM CỬA VÀ NÚT BẤM VÀO DANH SÁCH VẼ ===
        if self.button: self.drawlist.append(self.button)
        if self.door: self.drawlist.append(self.door)
        # === END: THÊM CỬA VÀ NÚT BẤM VÀO DANH SÁCH VẼ ===
        for body in self.agent_bodies.values(): self.drawlist += body.get_elements_to_render()
        self.terminateds = {agent_id: False for agent_id in self.possible_agents}
        self.truncateds = {agent_id: False for agent_id in self.possible_agents}
        self.terminateds["__all__"] = False
        self.truncateds["__all__"] = False
        
        return self._get_obs(), {}

    def step(self, action_dict):
        self.ts += 1

        # === START: LOGIC ĐIỀU KHIỂN CỬA VÀ NÚT BẤM ===
        # Kiểm tra xem có tác nhân nào đang đứng trên nút bấm không
        self.button_pressed = False
        if self.button:
            for agent_id in self.agents:
                body = self.agent_bodies[agent_id]
                for part in body.body_parts:
                    for contact in part.contacts:
                        if contact.other == self.button:
                            self.button_pressed = True
                            break
                    if self.button_pressed: break
                if self.button_pressed: break
        
        # Điều khiển cửa dựa vào trạng thái nút bấm
        if self.door_joint:
            if self.button_pressed:
                self.door_joint.motorSpeed = 5.0 # Mở cửa
            else:
                self.door_joint.motorSpeed = -5.0 # Đóng cửa
        # === END: LOGIC ĐIỀU KHIỂN CỬA VÀ NÚT BẤM ===

        for agent_id, action in action_dict.items():
            if agent_id in self.agents and action is not None and len(action) > 0:
                self.agent_bodies[agent_id].activate_motors(action)

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        rewards = {}
        positions = {aid: self.agent_bodies[aid].reference_head_object.position for aid in self.agents if aid in self.agent_bodies}
        if positions:
            leading_agent_pos_x = max(p[0] for p in positions.values())
            avg_agent_pos_y = sum(p[1] for p in positions.values()) / len(positions)
            self.scroll = [
                leading_agent_pos_x - self.rendering_viewer_w / SCALE / 5,
                avg_agent_pos_y - self.rendering_viewer_h / SCALE / 2.5
            ]
            
        active_agents_before_step = list(self.agents)
        for agent_id in active_agents_before_step:
            body = self.agent_bodies[agent_id]
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

            agent_critical_contact = False
            for part in body.body_parts:
                 if isinstance(part.userData, CustomBodyUserData) and part.userData.is_contact_critical and part.userData.has_contact:
                     # Bỏ qua va chạm với cửa
                     is_door_contact = False
                     if self.door:
                        for contact in part.contacts:
                            if contact.other == self.door:
                                is_door_contact = True
                                break
                     if not is_door_contact:
                        agent_critical_contact = True
                        break


            if agent_critical_contact or pos[0] < 0:
                reward = -100
                self.terminateds[agent_id] = True
            if pos[0] > (TERRAIN_LENGTH + self.TERRAIN_STARTPAD - TERRAIN_END) * TERRAIN_STEP:
                self.terminateds[agent_id] = True

            rewards[agent_id] = reward
            if self.terminateds.get(agent_id, False):
                if agent_id in self.agents:
                    self.agents.remove(agent_id)
        
        is_truncated = self.ts >= self.horizon
        all_done = not self.agents or is_truncated

        self.terminateds["__all__"] = all_done and not is_truncated
        self.truncateds["__all__"] = is_truncated

        obs = self._get_obs()
        return obs, rewards, self.terminateds, self.truncateds, {}

    def _get_obs(self):
        all_obs = {}
        agent_positions = {aid: body.reference_head_object.position for aid, body in self.agent_bodies.items()}

        for agent_id in self.agents:
            body = self.agent_bodies[agent_id]
            head = body.reference_head_object
            vel = head.linearVelocity
            my_pos = agent_positions[agent_id]

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
            if body.body_type == BodyTypesEnum.CLIMBER: state.extend(body.get_sensors_state())
            for lidar in self.lidar: state.append(lidar.fraction)
            for lidar in self.lidar:
                if lidar.is_water_detected: state.append(-1)
                elif lidar.is_creeper_detected: state.append(1)
                else: state.append(0)
            original_obs = np.array(state)
            
            # === START: THÊM THÔNG TIN MÔI TRƯỜNG VÀ TÁC NHÂN KHÁC VÀO QUAN SÁT ===
            # Vị trí tác nhân khác
            other_agents_pos = []
            for other_id in self.possible_agents:
                if other_id != agent_id:
                    if other_id in self.agents and other_id in agent_positions:
                        other_pos = agent_positions[other_id]
                        relative_pos = (np.array(other_pos) - np.array(my_pos)) / (LIDAR_RANGE * 2)
                        other_agents_pos.extend(relative_pos.tolist())
                    else:
                        other_agents_pos.extend([0.0, 0.0])

            # Vị trí cửa và nút bấm
            coop_mechanism_obs = []
            if self.door:
                rel_door_pos = (np.array(self.door.position) - np.array(my_pos)) / (LIDAR_RANGE*2)
                coop_mechanism_obs.extend(rel_door_pos.tolist())
            else:
                coop_mechanism_obs.extend([0.0, 0.0])
            
            if self.button:
                rel_button_pos = (np.array(self.button.position) - np.array(my_pos)) / (LIDAR_RANGE*2)
                coop_mechanism_obs.extend(rel_button_pos.tolist())
            else:
                coop_mechanism_obs.extend([0.0, 0.0])

            final_obs = np.concatenate([original_obs, np.array(other_agents_pos), np.array(coop_mechanism_obs)])
            # === END: THÊM THÔNG TIN MÔI TRƯỜNG VÀ TÁC NHÂN KHÁC VÀO QUAN SÁT ===

            all_obs[agent_id] = final_obs.astype(np.float32)
        return all_obs
    
    # Hàm render không thay đổi
    def render(self, mode='human', draw_lidars=True):
        return super().render(mode, draw_lidars)