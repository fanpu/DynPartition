import gym
from gym import spaces
import pygame
import numpy as np


class SchedulerEnv(gym.Env):
    num_batches = 10
    batch_size = 30  # 120
    image_w = 128
    image_h = 128

    """
    Current Assumptions: 2 devices
    Action space is the choice of the split

    Each step is just processing a batch, so this
    could really be a single-epoch environment.
    However let's retain this flexibility for now.
    """

    def __init__(self, render_mode=None, size=5):
        # Do not support render_mode for now

        self.num_layers = 9

        # Blind right now
        self.observation_space = spaces.Dict(
            {}
        )

        # Where to split the network
        self.action_space = spaces.Discrete(self.num_layers - 1)

        assert render_mode is None
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {}

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.current_batch = 0

        observation = self._get_obs()
        info = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

        return observation, info

    def step(self, action):
        assert self.current_batch < self.num_batches

        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(
            self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
