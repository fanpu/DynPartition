import gym
from gym import spaces
import numpy as np
from resnet import PipelineParallelResNet50
import torch
import timeit

batch_size = 30  # 120
image_w = 128
image_h = 128
DEVICE_0 = 'cuda:0'
DEVICE_1 = 'cpu'
DEVICES = [DEVICE_0, DEVICE_1]


class SchedulerEnv(gym.Env):
    num_batches = 24
    num_repeat = 1  # increase for variance reduction when computing rewards

    """
    Current Assumptions: 2 devices
    Action space is the choice of the split

    Each step is just processing a batch, so this
    could really be a single-epoch environment.
    However let's retain this flexibility for now.
    """

    def __init__(self, render_mode=None):
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
        # super().reset()

        self.current_batch = 0
        self.prev_action = None
        self.prev_model = None  # Cache prev model

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        assert self.current_batch < self.num_batches

        partition_layer = action
        self.current_batch += 1

        terminated = self.current_batch == self.num_batches

        setup = f"""\
inputs = torch.randn(batch_size, 3, image_w, image_h)
model = PipelineParallelResNet50(partition_layer={partition_layer})"""
        stmt = f"""\
outputs = model(inputs.to(DEVICE_0))"""
        run_times = timeit.repeat(
            stmt, setup, number=1, repeat=self.num_repeat, globals=globals())
        mean, std = np.mean(run_times), np.std(run_times)

        reward = -mean

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
