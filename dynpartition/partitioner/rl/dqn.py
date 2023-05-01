#!/usr/bin/env python

import numpy as np
import torch
from torch import tensor

from dynpartition.partitioner.rl.q_network import QNetwork
from dynpartition.partitioner.rl.replay_memory import ReplayMemory


class DqnAgent:
    STRATEGIES = ['static', 'static-cpu', 'random', 'rl']

    def __init__(self, env, strategy, render=False):
        # Create an instance of the network itself, as well as the memory.
        lr = 5e-4
        # self.env = FlattenObservation(env)
        self.env = env
        self.nNodes = len(env.action_space)
        self.nA = self.env.action_space[0].n
        self.nA_shape = (self.nNodes, self.nA)
        self.nS = self.env.observation_space.shape[0] * \
            self.env.observation_space.shape[1]
        self.epsilon = 0.05
        self.q_network = QNetwork(self.env, lr)
        self.replay = ReplayMemory()
        # self.burn_in_memory()
        self.E = 200
        self.N = 32
        self.gamma = 0.99
        self.test_episodes = 20
        self.c = 0
        assert strategy in self.STRATEGIES
        self.strategy = strategy

    def static_strategy(self):
        return np.zeros(self.nNodes, dtype=int)

    def static_cpu_strategy(self):
        return np.ones(self.nNodes, dtype=int)

    def random_strategy(self):
        return np.random.randint(low=0, high=2, size=self.nNodes, dtype=int)

    @torch.no_grad()
    def rl_epsilon_greedy_strategy(self, q_values):
        # Creating epsilon greedy probabilities to sample from.
        policy = np.zeros(self.nNodes)

        for node in range(self.nNodes):
            if np.random.rand() < self.epsilon:
                # Random policy
                policy[node] = np.random.choice(self.nA)
            else:
                policy[node] = np.argmax(q_values.numpy()[node])

        return policy.astype(int)

    @staticmethod
    @torch.no_grad()
    def rl_greedy_strategy(q_values):
        # Creating greedy policy for test time.
        return np.argmax(q_values.numpy(), axis=1).astype(int)

    def epsilon_greedy_policy(self, q_values):
        if self.strategy == 'static':
            return self.static_strategy()
        elif self.strategy == 'static-cpu':
            return self.static_cpu_strategy()
        elif self.strategy == 'random':
            return self.random_strategy()
        else:
            return self.rl_epsilon_greedy_strategy(q_values)

    def greedy_policy(self, q_values):
        if self.strategy == 'static':
            return self.static_strategy()
        elif self.strategy == 'static-cpu':
            return self.static_cpu_strategy()
        elif self.strategy == 'random':
            return self.random_strategy()
        else:
            return self.rl_greedy_strategy(q_values)

    def compute_loss(self, minibatch) -> torch.Tensor:
        loss = torch.tensor(0.0)
        y = []
        q_predict = []

        for (state, action, reward, new_state, sample_done) in minibatch:
            y.append(reward)
            q_predict.append(self.q_network.model(tensor(state))[
                                 torch.arange(len(action)), action].sum())

        for (yi, qi) in zip(y, q_predict):
            loss += torch.square(yi - qi)

        loss /= len(y)

        return loss

    def train(self):
        # In this function, we will train our network.

        # When use replay memory, you should interact with environment
        # here, and store these transitions to memory, while also
        # updating your model.
        state = self.env.reset()
        done = False

        while not done:
            action = self.epsilon_greedy_policy(
                self.q_network.model(tensor(state))
            )
            new_state, reward, done, info = self.env.step(action)
            self.replay.append((state, action, reward, new_state, done))
            minibatch = self.replay.sample_batch(self.N)
            loss: torch.Tensor = self.compute_loss(minibatch)
            self.q_network.optimizer.zero_grad()
            loss.backward()

            self.q_network.optimizer.step()

            # Update weights only every 50 steps to
            # reduce stability issue of moving targets
            self.c += 1
            if self.c % 20 == 0:
                self.q_network.target.load_state_dict(
                    self.q_network.model.state_dict())
                print("updating network")

            state = new_state

    @torch.no_grad()
    def test(self, model_file=None):
        # Evaluate the performance of your agent over 20 episodes,
        # by calculating average cumulative rewards (returns) for
        # the 20 episodes.
        # Here you need to interact with the environment,
        # irrespective of whether you are using replay memory.

        state = self.env.reset()
        done = False
        cumulative_rewards = 0
        while not done:
            action = self.greedy_policy(
                self.q_network.model(tensor(state)))
            state, reward, done, info = self.env.step(action)
            cumulative_rewards += reward
        return cumulative_rewards

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number
        # of episodes / transitions.

        state = self.env.reset()
        for i in range(10000):
            action = np.random.choice(self.nA)
            new_state, reward, done, info = self.env.step(action)
            self.replay.append((state, action, reward, new_state, done))
            state = new_state
            if done:
                state = self.env.reset()
