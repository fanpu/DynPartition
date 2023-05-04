import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dynpartition.get_dir import get_path, get_plot_path


def plot_single_device(data_dict):
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
            if "_single_" not in file.name:
                continue
            if not file.name.endswith(".json"):
                continue
            files.append(file)
        data_dict[i] = files

    new_data_dict = {}
    for i, v in data_dict.items():
        for j, filepath in enumerate(v):
            with open(filepath, "r") as f:
                v[j] = json.load(f)

            new_key = filepath.name.replace(".json", "")
            new_key = new_key.replace("_single", "")
            new_key = new_key.replace("_", " ")
            new_key = new_key.replace("treelstm", "TreeLSTM")
            new_key = new_key.replace("mathfunc", "MathFunc")
            new_key = new_key.replace("cpu", "CPU")
            new_key = new_key.replace("gpu", "GPU")
            new_key = new_key.replace(" sync", " Sync")
            new_key = new_key.replace(" async", " Async")
            new_value = []
            [new_value.extend(k) for k in v[j]]
            new_value = np.array(new_value) * 1e3
            new_data_dict[new_key] = {
                "mean": np.mean(new_value),
                "std": np.std(new_value),
                "min": np.min(new_value),
                "max": np.max(new_value),
            }

    # arrange keys CPU > GPU, Sync > Async
    def sort_key(x):
        if "Sync" in x and "CPU" in x:
            return 0
        elif "Async" in x and "CPU" in x:
            return 1
        elif "Sync" in x and "GPU" in x:
            return 2
        elif "Async" in x and "GPU" in x:
            return 3

    keys = list(new_data_dict.keys())
    keys = sorted(keys, key=sort_key)

    for i in keys:
        print(i, new_data_dict[i])

    # Plot separate subplots for TreeLSTM and MathFunc
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for i in keys:
        mean = new_data_dict[i]["mean"]
        std = new_data_dict[i]["std"]
        diff = new_data_dict[i]["max"] - new_data_dict[i]["min"]
        err = min(std, diff / 2, mean - new_data_dict[i]["min"])
        if "MathFunc" in i:
            name = i.replace("MathFunc ", "")
            ax[0].bar(name, mean, yerr=err, **default_plot_params)
        if "TreeLSTM" in i:
            name = i.replace("TreeLSTM ", "")
            ax[1].bar(name, mean, yerr=err, **default_plot_params)
    ax[0].set_title("MathFunc")
    ax[1].set_title("TreeLSTM")
    ax[0].set_ylabel("Time (ms)")
    ax[1].set_ylabel("Time (ms)")
    fig.suptitle("Single Device Performance")
    plt.tight_layout()
    plt.savefig(
        get_plot_path().joinpath("single_device.png"),
        bbox_inches='tight',
        dpi=300
    )
    plt.show()


if __name__ == '__main__':
    a100_path = get_path("_logs_a100")
    gtx1080ti_path = get_path("_logs_gtx1080ti")

    plot_single_device({
        "a100": a100_path,
        "gtx1080ti": gtx1080ti_path,
    })
