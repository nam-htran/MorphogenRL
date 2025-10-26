import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from .parametric_continuous_parkour import ParametricContinuousParkour
from .bodies.BodiesEnum import BodiesEnum
from .bodies.BodyTypesEnum import BodyTypesEnum
from .parametric_continuous_parkour import WATER_DENSITY

class MultiAgentParkour(MultiAgentEnv):
    _single_observation_space: gym.Space
    _single_action_space: gym.Space

    def __init__(self, config: dict):
        super().__init__()
        self.n_agents = config.get("n_agents", 2)
        self.render_mode = config.get("render_mode", None)

        sub_env_config = config.copy()
        sub_env_config.pop("n_agents", None)
        sub_env_config.pop("render_mode", None)

        body_type = BodiesEnum.get_body_type(sub_env_config.get("agent_body_type", "classic_bipedal"))
        if body_type in [BodyTypesEnum.SWIMMER, BodyTypesEnum.AMPHIBIAN]:
            sub_env_config['density'] = WATER_DENSITY

        tmp_env = ParametricContinuousParkour(**sub_env_config)
        self._single_observation_space = tmp_env.observation_space
        self._single_action_space = tmp_env.action_space
        tmp_env.close()

        self._agent_ids = {f"agent_{i}" for i in range(self.n_agents)}
        self.possible_agents = list(self._agent_ids)
        
        self.observation_space = gym.spaces.Dict({i: self._single_observation_space for i in self.possible_agents})
        self.action_space = gym.spaces.Dict({i: self._single_action_space for i in self.possible_agents})
        self.envs = [ParametricContinuousParkour(**sub_env_config) for _ in range(self.n_agents)]

        self._terminateds = set()
        self._truncateds = set()

        if self.render_mode == "human":
            self.envs[0].render_mode = "human"
            for i in range(1, self.n_agents):
                self.envs[i].render_mode = None

    def reset(self, *, seed=None, options=None):
        self._terminateds.clear()
        self._truncateds.clear()
        obs, infos = {}, {}
        for i, agent_id in enumerate(self.possible_agents):
            agent_seed = seed + i if seed is not None else None
            obs[agent_id], infos[agent_id] = self.envs[i].reset(seed=agent_seed, options=options)
        return obs, infos

    def step(self, action_dict):
        obs, rewards, terminateds, truncateds, infos = {}, {}, {}, {}, {}
        for agent_id, action in action_dict.items():
            idx = int(agent_id.split("_")[1])
            o, r, term, trunc, info = self.envs[idx].step(action)
            obs[agent_id] = o
            rewards[agent_id] = r
            terminateds[agent_id] = term
            truncateds[agent_id] = trunc
            infos[agent_id] = info
            if term: self._terminateds.add(agent_id)
            if trunc: self._truncateds.add(agent_id)

        all_done = len(self._terminateds) + len(self._truncateds) == self.n_agents
        terminateds["__all__"] = all_done
        truncateds["__all__"] = all_done
        return obs, rewards, terminateds, truncateds, infos

    def render(self):
        if self.render_mode == "human":
            return self.envs[0].render()

    def close(self):
        for env in self.envs:
            env.close()
