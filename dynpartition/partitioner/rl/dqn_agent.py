import numpy as np
import torch

from dynpartition.partitioner.rl.q_network import QNetwork
from dynpartition.partitioner.rl.replay_memory import ReplayMemory
from dynpartition.partitioner.rl.scheduler_env import SchedulerEnv


class DqnAgent:
    STRATEGIES = ['static', 'static-cpu', 'random', 'rl']

    def __init__(self, strategy, env=None):
        assert strategy in self.STRATEGIES

        if strategy == 'static' and not torch.cuda.is_available():
            raise ValueError("Cannot use static strategy without GPU,"
                             "use static-cpu instead.")

        # Create an instance of the network itself, as well as the memory.
        lr = 5e-4
        self.epsilon = 0.05
        self.E = 200
        self.N = 32
        self.gamma = 0.99
        self.test_episodes = 20
        self.c = 0
        self.strategy = strategy

        self.env = SchedulerEnv() if env is None else env
        self.q_network = QNetwork(self.env, lr)
        self.replay = ReplayMemory()

        self.n_nodes = self.q_network.n_nodes
        self.n_a = self.q_network.n_a
        self.n_a_shape = self.q_network.n_a_shape
        self.n_s = self.q_network.n_s

    def static_strategy(self):
        return np.ones(self.n_nodes, dtype=int)

    def static_cpu_strategy(self):
        return np.zeros(self.n_nodes, dtype=int)

    def random_strategy(self):
        return np.random.randint(
            low=0,
            high=len(),
            size=self.n_nodes,
            dtype=int
        )

    @torch.no_grad()
    def rl_epsilon_greedy_strategy(self, q_values):
        # Creating epsilon greedy probabilities to sample from.
        policy = np.zeros(self.n_nodes)

        for node in range(self.n_nodes):
            if np.random.rand() < self.epsilon:
                # Random policy
                policy[node] = np.random.choice(self.n_a)
            else:
                policy[node] = np.argmax(q_values[node])

        return policy.astype(int)

    @staticmethod
    def rl_greedy_strategy(q_values):
        # Creating greedy policy for test time.
        return np.argmax(q_values.numpy(), axis=1).astype(int)

    @torch.no_grad()
    def epsilon_greedy_policy(self, q_values):
        if self.strategy == 'static':
            return self.static_strategy()
        elif self.strategy == 'static-cpu':
            return self.static_cpu_strategy()
        elif self.strategy == 'random':
            return self.random_strategy()
        else:
            return self.rl_epsilon_greedy_strategy(q_values)

    @torch.no_grad()
    def greedy_policy(self, q_values):
        if self.strategy == 'rl':
            return self.rl_greedy_strategy(q_values)
        else:
            return self.epsilon_greedy_policy(q_values)

    def compute_loss(self, minibatch) -> torch.Tensor:
        loss = torch.tensor(0.0)
        y = []
        q_predict = []

        for state, action, reward, new_state in minibatch:
            y.append(reward)
            q_state = self.q_network.policy_net(state)
            idx = torch.arange(len(action))
            q_predict.append(q_state[idx, action].sum())

        for yi, qi in zip(y, q_predict):
            yi = torch.tensor(yi, device=qi.device, dtype=qi.dtype)
            loss += torch.nn.functional.mse_loss(yi, qi)

        loss /= len(y)
        return loss

    def train(self):
        # In this function, we will train our network.

        # When use replay memory, you should interact with environment
        # here, and store these transitions to memory, while also
        # updating your model.
        state = self.env.reset()

        self.q_network.policy_net.zero_grad()
        self.q_network.value_net.zero_grad()
        self.q_network.optimizer.zero_grad()

        action = self.epsilon_greedy_policy(
            self.q_network.policy_net(state)
        )
        new_state, reward, done, info = self.env.step(action)
        self.replay.append((state, action, reward, new_state))

        minibatch = self.replay.sample_batch(self.N)
        loss = self.compute_loss(minibatch)
        loss.backward()
        self.q_network.optimizer.step()


    @torch.no_grad()
    def test(self):
        # Evaluate the performance of your agent over 20 episodes,
        # by calculating average cumulative rewards (returns) for
        # the 20 episodes.
        # Here you need to interact with the environment,
        # irrespective of whether you are using replay memory.

        state = self.env.reset()
        cumulative_rewards = 0
        action = self.greedy_policy(self.q_network.policy_net(state))
        state, reward, done, info = self.env.step(action)
        cumulative_rewards += reward
        return cumulative_rewards
