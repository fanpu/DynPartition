#!/usr/bin/env python
# TODO fanpu

import argparse
import collections
import os
import sys
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch import tensor
from gym.wrappers import FlattenObservation


class FullyConnectedModel(torch.nn.Module):

    def __init__(self, input_size, output_size, output_shape):
        super().__init__()

        self.linear1 = torch.nn.Linear(input_size, 16)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(16, 16)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(16, 16)
        self.activation3 = torch.nn.ReLU()

        self.output_layer = torch.nn.Linear(16, output_size)
        self.output_shape = output_shape
        # no activation output layer

        # initialization
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linear3.weight)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, inputs):
        x = self.activation1(self.linear1(inputs))
        x = self.activation2(self.linear2(x))
        x = self.activation3(self.linear3(x))
        x = self.output_layer(x)
        return x.reshape(self.output_shape)


class QNetwork():

    # This class essentially defines the network architecture.
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.

    def __init__(self, env, lr, logdir=None):
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
        path = os.path.join(self.logdir, "model")
        torch.save(self.model.state_dict(), model_file)
        return path

    def load_model(self, model_file):
        # Helper function to load an existing model.
        return self.model.load_state_dict(torch.load(model_file))

    def load_model_weights(self, weight_file):
        # Optional Helper function to load model weights.
        pass


class Replay_Memory():

    def __init__(self, memory_size=50000, burn_in=10000):
        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions.
        self.memory = collections.deque(maxlen=memory_size)
        self.burn_in = burn_in

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.
        memory_len = len(self.memory)
        indices = np.random.choice(range(memory_len), size=batch_size)
        samples = []
        for i in indices:
            samples.append(self.memory[i])
        return samples

    def append(self, transition):
        # Appends transition to the memory.
        self.memory.append(transition)


class DQN_Agent():
    def __init__(self, env, render=False):
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
        self.replay = Replay_Memory()
        # self.burn_in_memory()
        self.E = 200
        self.N = 32
        self.gamma = 0.99
        self.test_episodes = 20
        self.c = 0

    def epsilon_greedy_policy(self, q_values):
        # Creating epsilon greedy probabilities to sample from.
        policy = np.zeros(self.nNodes)
        with torch.no_grad():
            for node in range(self.nNodes):
                if np.random.rand() < self.epsilon:
                    # Random policy
                    policy[node] = np.random.choice(self.nA)
                else:
                    policy[node] = np.argmax(q_values.numpy()[node])
        return policy.astype(int)

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time.
        with torch.no_grad():
            return np.argmax(q_values.numpy(), axis=1).astype(int)

    def compute_loss(self, minibatch):
        loss = 0
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

        # When use replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.
        state = self.env.reset()
        done = False

        # TODO: change this to monte carlo updates with no discount on rewards
        while not done:
            action = self.epsilon_greedy_policy(
                self.q_network.model(tensor(state)))
            new_state, reward, done, info = self.env.step(action)
            self.replay.append((state, action, reward, new_state, done))
            minibatch = self.replay.sample_batch(self.N)
            loss = self.compute_loss(minibatch)
            self.q_network.optimizer.zero_grad()
            loss.backward()

            self.q_network.optimizer.step()

            # Update weights only every 50 steps to reduce stability issue of moving targets
            self.c += 1
            if self.c % 50 == 0:
                self.q_network.target.load_state_dict(
                    self.q_network.model.state_dict())

            state = new_state

    def test(self, model_file=None):
        # Evaluate the performance of your agent over 20 episodes, by calculating average cumulative rewards (returns) for the 20 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using replay memory.

        with torch.no_grad():
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
        # Initialize your replay memory with a burn_in number of episodes / transitions.

        state = self.env.reset()
        for i in range(10000):
            action = np.random.choice(self.nA)
            new_state, reward, done, info = self.env.step(action)
            self.replay.append((state, action, reward, new_state, done))
            state = new_state
            if done:
                state = self.env.reset()
