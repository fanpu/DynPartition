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
    device_id_to_device_string, device_id_to_device, allocation_summary, ALL_DEVICES
from dynpartition.partitioner.async_execution import test_model_with

batch_size = 30  # 120
MAX_NODES = 128


class SchedulerEnv(gym.Env):
    num_batches = 1
    num_repeat = 10  # increase for variance reduction when computing rewards

    def __init__(self, is_train=True, render_mode=None):
        # Do not support render_mode for now
        self.model, train_dataset, dev_dataset, test_dataset = load_tree_lstm()

        if is_train:
            self.dataset = train_dataset
            self.dataset_len = len(train_dataset)
            self.encoded_trees = create_tree_embedding_dataset(
                train_dataset.trees,
                max_num_nodes=MAX_NODES,
                name="train_sst",
                set_traversal_index=True
            )
        else:
            self.dataset = test_dataset
            self.dataset_len = len(test_dataset)
            self.encoded_trees = create_tree_embedding_dataset(
                test_dataset.trees,
                max_num_nodes=MAX_NODES,
                name="test_sst"
            )

        self._gen_new_sample()
        self.observation_shape = self.encoded_trees[0].numpy().shape
        self.observation_space = spaces.Box(
            low=-np.ones(self.observation_shape),
            high=np.ones(self.observation_shape)
        )

        self.max_nodes = self.observation_shape[0]
        self.num_devices = torch.cuda.device_count() + 1

        # Action space: Which GPU to assign the node to
        self.action_space = spaces.MultiDiscrete(
            [self.num_devices] * self.max_nodes
        )

        assert render_mode is None
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _gen_new_sample(self):
        self.obs_id = np.random.randint(low=0, high=self.dataset_len)

    def _get_obs(self):
        return self.encoded_trees[self.obs_id].reshape(-1,).numpy()

    def _get_info(self):
        return None

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        # super().reset()

        self.input = None
        self.allocations = {}  # Determined allocations
        self.current_batch = 0
        self.prev_action = None

        self._gen_new_sample()
        observation = self._get_obs()

        return observation

    def step(self, action):
        # Action should be 1d

        assert self.current_batch < self.num_batches

        tree_size = self.dataset[self.obs_id][0].size
        device_allocations = {}

        for idx, device_id in enumerate(action):
            device_allocations[idx] = device_id_to_device_string(device_id)

        allocation_summary(device_allocations)

        self.current_batch += 1

        terminated = self.current_batch == self.num_batches

        tree = self.dataset.trees[self.obs_id]
        for traversal_idx in range(tree.size()):
            node = tree.traversal_dict[traversal_idx]
            node.device_for_state = device_id_to_device(action[traversal_idx])
            node.device_for_output = device_id_to_device(action[traversal_idx])

        def execute_forward():
            test_model_with(
                self.model,
                dataset=[tree],
                devices=ALL_DEVICES,
                execution_strategy='sync',
                with_tqdm=False
            )

        run_times = timeit.repeat(
            execute_forward,
            number=1,
            repeat=self.num_repeat,
            globals=globals()
        )
        mean, std = np.mean(run_times), np.std(run_times)

        reward = -mean
        self._gen_new_sample()
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, info
