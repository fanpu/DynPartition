import matplotlib.pyplot as plt
import numpy as np

from dynpartition.get_dir import get_plot_path
from dynpartition.partitioner.rl.dqn_agent import DqnAgent


def main():
    num_seeds = 5
    num_episodes = 1000
    num_test_episodes = 5
    episodes_between_test = 5
    l = num_episodes // episodes_between_test
    res = np.zeros((num_seeds, l))
    agent = DqnAgent(strategy='rl')

    for i in range(num_seeds):
        reward_means = []
        for m in range(num_episodes):
            agent.train()

            if m % episodes_between_test != 0:
                continue

            print(f"Episode: {m}")
            G = np.zeros(episodes_between_test)
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
    print(res)
    print(res.shape)
    avs = np.mean(res, axis=0)
    maxs = np.max(res, axis=0)
    mins = np.min(res, axis=0)

    fig, ax = plt.subplots(layout='constrained')
    ax.fill_between(ks, mins, maxs, alpha=0.1)
    ax.plot(ks, avs, '-o', markersize=1)

    ax.set_xlabel('Episode', fontsize=15)
    ax.set_ylabel('Reward', fontsize=15)

    ax.set_title(
        f"DynPartition Learning Curve ({agent.strategy})", fontsize=20
    )
    # ax.set_ylim(-30, 0)

    plot_path = get_plot_path().joinpath(
        f"dynpartition_learning_curve_{agent.strategy}.png"
    )
    plt.savefig(plot_path)
    print("Plot saved in", plot_path)


if __name__ == '__main__':
    main()
