import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from dynpartition.get_dir import get_path, get_plot_path


def plot_rl_distribution(data_dir):
    data_dict = {}
    for file in Path(data_dir).iterdir():
        if not file.is_file():
            continue
        if f"_dynpartition_learning_curve_" not in file.name:
            continue
        if not file.name.endswith(".json"):
            continue

        strategy = file.name.split("_")[-1].replace(".json", "")

        if strategy not in data_dict:
            with open(file, "r") as f:
                data_dict[strategy] = json.load(f)
        else:
            with open(file, "r") as f:
                data = json.load(f)

            strategy_data = data_dict[strategy]
            assert strategy_data["num_episodes"] == data["num_episodes"]
            assert strategy_data["num_test_episodes"] == data[
                "num_test_episodes"]
            assert strategy_data["episodes_between_test"] == data[
                "episodes_between_test"]
            strategy_data["num_seeds"] += data["num_seeds"]
            strategy_data["res"].extend(data["res"])

    for strategy, data in data_dict.items():
        l = data["l"]
        episodes_between_test = data["episodes_between_test"]
        res = np.array(data["res"])

        ks = np.arange(l) * episodes_between_test
        avs = np.mean(res, axis=0)
        maxs = np.max(res, axis=0)
        mins = np.min(res, axis=0)

        fig, ax = plt.subplots()
        ax.fill_between(ks, mins, maxs, alpha=0.5)
        ax.plot(ks, avs, markersize=0.5, alpha=1)
        # add cumulative running average
        window_size = 100
        ax.plot(
            ks[:window_size],
            (np.cumsum(avs) / np.arange(1, len(avs) + 1))[:window_size],
            markersize=1,
            color='red',
        )

        # add running average with window size 10
        ax.plot(
            ks[window_size - 1:],
            np.convolve(avs, np.ones(window_size), 'valid') / window_size,
            markersize=1,
            color='red',
        )

        ax.set_xlim(0, ks[-1])

        ax.set_xlabel('Episode', fontsize=15)
        ax.set_ylabel('Reward', fontsize=15)

        if strategy == "static":
            title = "Static on GPU"
        elif strategy == "static-cpu":
            title = "Static on CPU"
        elif strategy == "rl":
            title = "RL with Q-Learning"
        elif strategy == "rl-policy-value":
            title = "RL with Policy Gradient"
        else:
            title = strategy.capitalize()

        ax.set_title(title, fontsize=20)
        fig.tight_layout()

        plot_path = get_plot_path().joinpath(
            f"dynpartition_learning_curve_{strategy}.png"
        )
        plt.savefig(plot_path)
        print(f"Avg reward: {np.mean(avs[-1000:]):.5f} ms of {res.shape[0]}"
              f" plots saved in {plot_path}")


if __name__ == '__main__':
    plot_rl_distribution(get_path("_logs_a100"))
