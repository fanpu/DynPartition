import timeit

import gym
import ipdb
import numpy as np
import torch
from gym import spaces
from resnet import PipelineParallelResNet50

from dynpartition.dataset.encoding_trees import create_tree_embedding_dataset
from dynpartition.dataset.load import load_tree_lstm
from dynpartition.partitioner.partitioner_utils import \
    device_id_to_device_string

batch_size = 30  # 120

DEVICE_0 = 'cuda:0'
DEVICE_1 = 'cpu'
DEVICES = [DEVICE_0, DEVICE_1]

MAX_NODES = 128


class SchedulerEnv(gym.Env):
    num_batches = 1
    num_repeat = 1  # increase for variance reduction when computing rewards

    def __init__(self, is_train=True, render_mode=None):
        # Do not support render_mode for now

        device = torch.device("cuda" if (
            True and torch.cuda.is_available()) else "cpu")
        _, train_dataset, dev_dataset, test_dataset = load_tree_lstm(device)
        encoded_trees = create_tree_embedding_dataset(
            test_dataset.trees, padding=MAX_NODES, name="test_sst", plot=True)

        if is_train:
            self.dataset = train_dataset
            self.dataset_len = len(train_dataset)
        else:
            self.dataset = test_dataset
            self.dataset_len = len(test_dataset)

        ipdb.set_trace()
        # TODO: state will be the input
        self.observation_shape = encoded_trees[0].numpy().shape
        self.observation_space = spaces.Dict(
            {
                "input_tree": spaces.Box(low=-np.ones(self.observation_shape),
                                         high=np.ones(self.observation_shape), dtype=np.float16)
            }
        )

        # Action space: Which GPU to assign the node to
        self.action_space = spaces.Discrete(torch.cuda.device_count())

        assert render_mode is None
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self):
        # Might want to make this adaptive
        return {"batch_size": batch_size}

    def _get_info(self):
        return None

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        # super().reset()

        # TODO: sample a new input from our dataset
        self.input = None
        self.allocations = {}  # Determined allocations
        self.current_batch = 0
        self.prev_action = None

        observation = self._get_obs()

        return observation

    def step(self, action):
        assert self.current_batch < self.num_batches

        print("Chose action", action)

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

        return observation, reward, terminated, info
