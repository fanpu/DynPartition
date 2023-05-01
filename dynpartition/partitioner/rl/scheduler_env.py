import timeit

import gym
import numpy as np
import torch
from gym import spaces

from dynpartition.dataset.encoding_trees import create_tree_embedding_dataset
from dynpartition.dataset.load import load_tree_lstm
from dynpartition.partitioner.time_measurements import for_time_measurement
from dynpartition.partitioner.utils import \
    device_id_to_device_string, device_id_to_device, ALL_DEVICES

batch_size = 30  # 120
MAX_NODES = 128


class SchedulerEnv(gym.Env):
    num_batches = 1
    num_repeat = 10  # increase for variance reduction when computing rewards

    def __init__(self, is_train=True):
        # Do not support render_mode for now
        self.input = None
        self.allocations = {}  # Determined allocations
        self.current_batch = 0
        self.prev_action = None
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
        self.observation_shape = self.encoded_trees[0].shape
        self.observation_space = spaces.Box(
            low=-np.ones(self.observation_shape),
            high=np.ones(self.observation_shape)
        )

        self.max_nodes = self.observation_shape[0]
        self.num_devices = len(ALL_DEVICES)

        # Action space: Which GPU to assign the node to
        self.action_space = spaces.MultiDiscrete(
            [self.num_devices] * self.max_nodes
        )

        self.render_mode = None
        self.window = None
        self.clock = None

    def _gen_new_sample(self):
        self.obs_id = np.random.randint(low=0, high=self.dataset_len)

    def _get_obs(self):
        return self.encoded_trees[self.obs_id]

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        # super().reset()

        self.input = None
        self.allocations = {}  # Determined allocations
        self.current_batch = 0
        self.prev_action = None

        self._gen_new_sample()
        return self._get_obs()

    def step(self, action) -> (np.ndarray, float, bool, dict):
        assert self.current_batch < self.num_batches
        self.current_batch += 1
        device_allocations = {}

        for idx, device_id in enumerate(action):
            device_allocations[idx] = device_id_to_device_string(device_id)

        tree = self.dataset.trees[self.obs_id]
        for traversal_idx in range(tree.size()):
            node = tree.traversal_dict[traversal_idx]
            node.device_for_state = device_id_to_device(action[traversal_idx])
            node.device_for_output = device_id_to_device(action[traversal_idx])

        execute_forward = for_time_measurement(
            model=self.model,
            tree=tree,
            devices=ALL_DEVICES,
            execution_strategy='async',
        )

        run_times = timeit.repeat(
            execute_forward,
            number=1,
            repeat=self.num_repeat,
            globals=globals()
        )
        run_times = np.array(run_times) * 1000
        mean, std = np.mean(run_times), np.std(run_times)

        # Waits for everything to finish running
        torch.cuda.synchronize()
        self._gen_new_sample()

        observation = self._get_obs()
        reward = -mean
        terminated = self.current_batch == self.num_batches
        return observation, reward, terminated, {}
