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
        self.model, train_dataset, dev_dataset, test_dataset = load_tree_lstm(
            device)

        if is_train:
            self.dataset = train_dataset
            self.dataset_len = len(train_dataset)
            self.encoded_trees = create_tree_embedding_dataset(
                train_dataset.trees, max_num_nodes=MAX_NODES, name="train_sst",
                set_traversal_index=True, plot=True)
        else:
            self.dataset = test_dataset
            self.dataset_len = len(test_dataset)
            self.encoded_trees = create_tree_embedding_dataset(
                test_dataset.trees, max_num_nodes=MAX_NODES, name="test_sst", plot=True)

        self.obs_id = np.random.randint(low=0, high=self.dataset_len)
        self.observation_shape = self.encoded_trees[0].numpy().shape
        self.observation_space = spaces.Box(
            low=-np.ones(self.observation_shape), high=np.ones(self.observation_shape))

        self.max_nodes = self.observation_shape[0]
        self.num_devices = torch.cuda.device_count() + 1

        # Action space: Which GPU to assign the node to
        self.action_space = spaces.MultiDiscrete(
            [self.num_devices] * self.max_nodes)

        assert render_mode is None
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self):
        return self.encoded_trees[self.obs_id].reshape(-1,)

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
        self.obs_id = np.random.randint(low=0, high=self.dataset_len)

        observation = self._get_obs()

        return observation

    def step(self, action):
        # Action should be 1d

        assert self.current_batch < self.num_batches

        tree_size = self.dataset[self.obs_id][0].size
        device_allocations = {}
        # the tree indices range in [1, tree_size] inclusive
        for idx, device_id in enumerate(action):
            device_allocations[idx] = device_id_to_device_string(device_id)

        print("Allocation:", device_allocations)

        self.current_batch += 1

        terminated = self.current_batch == self.num_batches

        import ipdb
        ipdb.set_trace()
        tree = self.dataset.trees[self.obs_id]
        output = self.model.forward(
            tree, device_allocations=device_allocations)

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
