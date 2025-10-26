import gymnasium as gym
import numpy as np

class SB3MultiAgentWrapper(gym.Wrapper):
    """
    Wrapper that converts an RLlib MultiAgentEnv into a Stable Baselines 3–compatible
    environment by flattening observations and actions across all agents.
    """
    def __init__(self, env):
        super().__init__(env)
        # RLlib's MultiAgentEnv doesn’t follow gym.Wrapper’s API,
        # so we access the base environment via self.env.
        self.agents = self.env.possible_agents
        
        agent_obs_space = self.env.observation_space[self.agents[0]]
        agent_act_space = self.env.action_space[self.agents[0]]
        
        self.observation_space = gym.spaces.Box(
            low=np.tile(agent_obs_space.low, len(self.agents)),
            high=np.tile(agent_obs_space.high, len(self.agents)),
            dtype=agent_obs_space.dtype
        )

        if isinstance(agent_act_space, gym.spaces.Box):
            self.action_space = gym.spaces.Box(
                low=np.tile(agent_act_space.low, len(self.agents)),
                high=np.tile(agent_act_space.high, len(self.agents)),
                dtype=agent_act_space.dtype
            )
        else:
            raise NotImplementedError("Only Box action spaces are supported in this wrapper.")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flatten_obs(obs), info

    def step(self, action):
        split_actions = np.split(action, len(self.agents))
        action_dict = {agent: act for agent, act in zip(self.agents, split_actions)}
        
        obs, reward, terminated, truncated, info = self.env.step(action_dict)
        
        flat_obs = self._flatten_obs(obs)
        total_reward = sum(reward.values()) if isinstance(reward, dict) else reward
        all_done = terminated.get('__all__', False)
        all_trunc = truncated.get('__all__', False)
        
        return flat_obs, total_reward, all_done or all_trunc, info

    def _flatten_obs(self, obs_dict):
        """Flatten agent observations from a dict into a single NumPy array."""
        obs_list = []
        for agent in self.agents:
            if agent in obs_dict:
                obs_list.append(obs_dict[agent])
            else:
                # If an agent is done, its observation may be missing from the dict
                obs_list.append(np.zeros_like(self.env.observation_space[agent].sample()))
        return np.concatenate(obs_list)

    @property
    def unwrapped(self):
        """Return the base environment (for attributes like `viewer`)."""
        return self.env
