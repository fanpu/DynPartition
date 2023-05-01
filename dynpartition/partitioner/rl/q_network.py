import numpy as np
import torch

from dynpartition.get_dir import get_log_path
from dynpartition.partitioner.rl.q_estimator import FullyConnectedModel
from dynpartition.partitioner.rl.scheduler_env import SchedulerEnv


class QNetwork:
    # This class essentially defines the network architecture.
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.

    def __init__(self, env: SchedulerEnv, lr: float):
        self.n_a = env.num_devices
        self.n_nodes = np.prod(env.action_space.shape)
        self.n_s = np.prod(env.observation_space.shape)
        self.n_a_total = self.n_nodes * self.n_a
        self.n_a_shape = (self.n_nodes, self.n_a)
        self.model = FullyConnectedModel(self.n_s, self.n_a_total,
                                         self.n_a_shape)
        self.target = FullyConnectedModel(self.n_s, self.n_a_total,
                                          self.n_a_shape)
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def save_model_weights(self, suffix):
        # Helper function to save your model / weights.
        path = get_log_path().joinpath("model")
        torch.save(self.model.state_dict(), path.joinpath(f"model_{suffix}"))
        return path

    def load_model(self, model_file):
        # Helper function to load an existing model.
        return self.model.load_state_dict(torch.load(model_file))

    def load_model_weights(self, weight_file):
        # Optional Helper function to load model weights.
        pass
