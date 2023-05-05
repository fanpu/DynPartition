import json
from datetime import datetime

import numpy as np
import torch

from dynpartition.get_dir import get_log_path
from dynpartition.partitioner.rl.dqn_agent import DqnAgent
from dynpartition.partitioner.utils import STRATEGY


def main():
    num_seeds = 5
    num_episodes = 10000
    num_test_episodes = 5
    episodes_between_test = 5
    l = num_episodes // episodes_between_test
    res = np.zeros((num_seeds, l))
    agent = DqnAgent(strategy=STRATEGY)

    for i in range(num_seeds):
        reward_means = []
        for m in range(1, num_episodes + 1):
            agent.train()

            if m % episodes_between_test != 0:
                continue

            G = np.zeros(episodes_between_test)
            for k in range(num_test_episodes):
                g = agent.test()
                G[k] = g

            reward_mean = G.mean()
            reward_sd = G.std()
            reward_means.append(reward_mean)
            print(f"({i + 1}/{num_seeds}) "
                  f"Episode: {m}/{num_episodes} "
                  f"Reward: {reward_mean} +/- {reward_sd}")

        res[i] = np.array(reward_means)

    timestamp = datetime.now().timestamp()
    file_name = f"{timestamp}_dynpartition_learning_curve_{agent.strategy}"
    with open(get_log_path().joinpath(f"{file_name}.json"), "w") as f:
        json.dump({
            "num_seeds": num_seeds,
            "num_episodes": num_episodes,
            "num_test_episodes": num_test_episodes,
            "episodes_between_test": episodes_between_test,
            "l": l,
            "res": res.tolist(),
        }, f)
    torch.save(agent.get_action(),
               get_log_path().joinpath(f"{file_name}_action.pth"))


if __name__ == '__main__':
    main()
