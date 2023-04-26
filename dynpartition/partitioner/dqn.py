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

    def __init__(self, input_size, output_size):
        super().__init__()

        self.linear1 = torch.nn.Linear(input_size, 16)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(16, 16)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(16, 16)
        self.activation3 = torch.nn.ReLU()

        self.output_layer = torch.nn.Linear(16, output_size)
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
        return x


class QNetwork():

    # This class essentially defines the network architecture.
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.

    def __init__(self, env, lr, logdir=None):
        # Define your network architecture here. It is also a good idea to define any training operations
        # and optimizers here, initialize your variables, or alternately compile your model here.
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        nA = env.action_space.n
        nS = env.observation_space.shape[0]
        self.model = FullyConnectedModel(nS, nA)
        self.target = FullyConnectedModel(nS, nA)
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # self.loss = torch.nn.MSELoss()

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
        self.env = FlattenObservation(env)
        self.nA = self.env.action_space.n
        self.nS = self.env.observation_space.shape[0]
        self.epsilon = 0.5
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
        with torch.no_grad():
            print(q_values)
            if np.random.rand() < self.epsilon:
                # Random policy
                return np.random.choice(len(q_values))
            else:
                return np.argmax(q_values.numpy())

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time.
        with torch.no_grad():
            return np.argmax(q_values.numpy())

    def compute_loss(self, minibatch):
        loss = 0
        y = []
        q_predict = []

        for (state, action, reward, new_state, sample_done) in minibatch:
            if sample_done:
                y.append(reward)
            else:
                with torch.no_grad():
                    y.append(
                        reward + self.gamma *
                        np.max(self.q_network.target(tensor(new_state)).detach().numpy()))
            q_predict.append(self.q_network.model(tensor(state))[action])

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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Deep Q Network Argument Parser')
    parser.add_argument('--env', dest='env', type=str, default='CartPole-v0')
    parser.add_argument('--render', dest='render', type=int, default=0)
    parser.add_argument('--train', dest='train', type=int, default=1)
    parser.add_argument('--model', dest='model_file', type=str)
    parser.add_argument('--lr', dest='lr', type=float, default=5e-4)
    return parser.parse_args()


def main(args):
    args = parse_arguments()
    environment_name = args.env

    # You want to create an instance of the DQN_Agent class here, and then train / test it.

    num_seeds = 5
    num_episodes = 20
    l = num_episodes // 10
    res = np.zeros((num_seeds, l))
    num_test_episodes = 20
    for i in tqdm.tqdm(range(num_seeds)):
        reward_means = []
        agent = DQN_Agent(environment_name)

        for m in range(num_episodes):
            agent.train()
            if m % 10 == 0:
                print("Episode: {}".format(m))
                G = np.zeros(20)
                for k in range(num_test_episodes):
                    g = agent.test()
                    G[k] = g

                reward_mean = G.mean()
                reward_sd = G.std()
                print("The test reward for episode {0} is {1} with sd of {2}.".format(
                    m, reward_mean, reward_sd))
                reward_means.append(reward_mean)

                # test_video(agent, agent.env, m)

        res[i] = np.array(reward_means)

    ks = np.arange(l) * 10
    avs = np.mean(res, axis=0)
    maxs = np.max(res, axis=0)
    mins = np.min(res, axis=0)

    plt.fill_between(ks, mins, maxs, alpha=0.1)
    plt.plot(ks, avs, '-o', markersize=1)

    plt.xlabel('Episode', fontsize=15)
    plt.ylabel('Return', fontsize=15)

    plt.title("DQN Learning Curve", fontsize=24)
    plt.savefig(get_plot_path().joinpath("dqn_curve.png"))


if __name__ == '__main__':
    main(sys.argv)
