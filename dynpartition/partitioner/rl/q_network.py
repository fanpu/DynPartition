import torch

from dynpartition.get_dir import get_log_path
from dynpartition.partitioner.rl.q_estimator import FullyConnectedModel


class QNetwork:
    # This class essentially defines the network architecture.
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.

    def __init__(self, env, lr):
        nA = env.action_space[0].n
        nNodes = len(env.action_space)
        nS = env.observation_space.shape[0] * env.observation_space.shape[1]
        nA_total = nNodes * nA
        self.nA_shape = (nNodes, nA)
        self.model = FullyConnectedModel(nS, nA_total, self.nA_shape)
        self.target = FullyConnectedModel(nS, nA_total, self.nA_shape)
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
