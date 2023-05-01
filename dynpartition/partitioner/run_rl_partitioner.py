# TODO: Fanpu

import timeit

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from dqn import DQN_Agent
from dynpartition.get_dir import get_plot_path
from scheduler_env import SchedulerEnv


def main():
    num_seeds = 1
    num_episodes = 100
    num_test_episodes = 5
    episodes_between_test = 5
    l = num_episodes // episodes_between_test
    res = np.zeros((num_seeds, l))
    agent = DQN_Agent(SchedulerEnv(), strategy='static-cpu')

    reward_means = []
    for i in tqdm.tqdm(range(num_seeds)):
        for m in range(num_episodes):
            print("Episode", m)
            agent.train()

            # if episodes_between_test != 0:
            #     continue

            print(f"Episode: {m}")
            G = np.zeros(20)
            for k in range(num_test_episodes):
                g = agent.test()
                G[k] = g

            reward_mean = G.mean()
            reward_sd = G.std()
            print(
                f"The test reward for episode {m} is {reward_mean} "
                f"with sd of {reward_sd}."
            )
            reward_means.append(reward_mean)

        print(reward_means)
        res[i] = np.array(reward_means)

    ks = np.arange(l) * episodes_between_test
    avs = np.mean(res, axis=0)
    maxs = np.max(res, axis=0)
    mins = np.min(res, axis=0)

    plt.fill_between(ks, mins, maxs, alpha=0.1)
    plt.plot(ks, avs, '-o', markersize=1)

    plt.xlabel('Episode', fontsize=15)
    plt.ylabel('Return', fontsize=15)

    plt.title(f"DynPartition Learning Curve ({agent.strategy})", fontsize=24)
    plot_path = get_plot_path().joinpath(
        f"dynpartition_learning_curve_{agent.strategy}.png")
    plt.savefig(plot_path)
    print("Plot saved in", plot_path)


if __name__ == '__main__':
    main()
