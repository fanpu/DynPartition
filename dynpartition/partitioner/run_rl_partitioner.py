import matplotlib.pyplot as plt
import numpy as np

from dynpartition.get_dir import get_plot_path
from dynpartition.partitioner.rl.dqn import DqnAgent


def main():
    num_episodes = 10
    num_test_episodes = 1
    strategy = 'rl'

    res = np.zeros(num_episodes)
    agent = DqnAgent(strategy=strategy)
    for m in range(num_episodes):
        agent.train()

        test_rewards = np.zeros(num_test_episodes)
        for k in range(num_test_episodes):
            g = agent.test()
            test_rewards[k] = g

        res[m] = test_rewards.mean()
        print(f"{m} Time Reward: {res[m]:.4f}")

    plt.figure(figsize=(10, 5))
    x = np.arange(num_episodes)
    plt.plot(x, res)
    plt.xlabel('Episode', fontsize=15)
    plt.ylabel('Return', fontsize=15)
    plt.title(f"DynPartition Learning Curve ({agent.strategy})", fontsize=24)

    plot_path = get_plot_path().joinpath(
        f"dynpartition_learning_curve_{agent.strategy}.png"
    )
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    print("Plot saved in", plot_path)


if __name__ == '__main__':
    main()
