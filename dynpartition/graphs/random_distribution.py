import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dynpartition.get_dir import get_path, get_plot_path


def plot_random_distribution(data_dict):
    default_plot_params = {
        "align": 'center',
        "alpha": 0.5,
        "ecolor": 'red',
        "capsize": 10,
        "width": 0.6,
    }

    for i in data_dict:
        files = []
        for file in Path(data_dict[i]).iterdir():
            if not file.is_file():
                continue
            if "_random_" not in file.name:
                continue
            files.append(file)
        data_dict[i] = files

    new_data_dict = {}
    for i, v in data_dict.items():
        for j, filepath in enumerate(v):
            with open(filepath, "r") as f:
                v[j] = json.load(f)

            new_key = filepath.name.replace(".json", "")
            new_key = new_key.replace("_random", "")
            new_key = new_key.replace("_", " ")
            new_key = new_key.replace("treelstm", "TreeLSTM")
            new_key = new_key.replace("mathfunc", "MathFunc")
            new_key = new_key.replace("cpu", "CPU")
            new_key = new_key.replace("gpu", "GPU")
            new_key = new_key.replace(" sync", " Sync")
            new_key = new_key.replace(" async", " Async")
            new_key = new_key.split(" ")
            if new_key[-2] == "1":
                new_key[-1] = f"{int(new_key[-1]) - 1} GPU"
            else:
                new_key[-1] = f"{new_key[-1]} GPU"
            new_key[-2] = " + CPU" if new_key[-2] == "1" else ""
            new_key[-2] = f"{new_key[-1]}{new_key[-2]}"
            new_key = list(reversed(new_key[:-1]))
            new_key[1], new_key[2] = new_key[2], new_key[1]
            new_key = tuple(new_key)
            if new_key[1] != "3":
                continue
            if new_key[2] != "Async":
                continue

            new_value = []
            [new_value.extend(k) for k in v[j]]
            new_value = np.array(new_value) * 1e3
            new_data_dict[new_key] = {
                "mean": np.mean(new_value),
                "std": np.std(new_value),
                "min": np.min(new_value),
                "max": np.max(new_value),
            }

    # arrange keys Sync > Async
    def key_func(key):
        return key[2] != "Sync", key[0]

    keys = list(new_data_dict.keys())
    keys = sorted(keys, key=key_func)

    for i in keys:
        print(i, new_data_dict[i])

    # Plot separate subplots for MathFunc with CPU and GPU
    # and TreeLSTM with CPU and GPU
    fig, ax = plt.subplots(2, 2, figsize=(10, 7), sharey='row')
    fig.tight_layout(pad=3.0)
    fig.subplots_adjust(top=0.85)

    for i, key in enumerate(keys):
        mean = new_data_dict[key]["mean"]
        std = new_data_dict[key]["std"]
        diff = new_data_dict[key]["max"] - new_data_dict[key]["min"]
        err = min(std, diff / 2, mean - new_data_dict[key]["min"])
        name = f"{key[0]}"
        name = name.replace(" + CPU", "")
        name = name.replace(" GPU", "")
        name = int(name)
        if key[3] == "MathFunc":
            if "CPU" in key[0]:
                ax[0][1].bar(name, mean, yerr=err, **default_plot_params)
            else:
                ax[0][0].bar(name, mean, yerr=err, **default_plot_params)
        else:
            if "CPU" in key[0]:
                ax[1][1].bar(name, mean, yerr=err, **default_plot_params)
            else:
                ax[1][0].bar(name, mean, yerr=err, **default_plot_params)

    ax[0][0].set_title("MathFunc (n GPU)")
    ax[0][1].set_title("MathFunc (CPU + n GPU)")
    ax[1][0].set_title("TreeLSTM (n GPU)")
    ax[1][1].set_title("TreeLSTM (CPU + n GPU)")
    ax[0][0].set_ylabel("Time (ms)")
    ax[1][0].set_ylabel("Time (ms)")
    ax[1][0].set_xlabel("Number of GPUs")
    ax[1][1].set_xlabel("Number of GPUs")

    ax[0][0].set_xticks([1, 2, 3, 4])
    ax[0][1].set_xticks([1, 2, 3, 4])
    ax[1][0].set_xticks([1, 2, 3, 4])
    ax[1][1].set_xticks([1, 2, 3, 4])
    fig.suptitle("Random Distribution of Nodes")
    plt.tight_layout()
    plt.savefig(
        get_plot_path().joinpath("random_distribution.png"),
        bbox_inches='tight',
        dpi=300
    )
    plt.show()


if __name__ == '__main__':
    a100_path = get_path("_logs_a100")

    plot_random_distribution({
        "a100": a100_path,
    })
