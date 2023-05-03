import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dynpartition.get_dir import get_path, get_plot_path


def plot_manual_distribution(data_dict, manual_index):
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
            if f"_manual_{manual_index}_" not in file.name:
                continue
            files.append(file)
        data_dict[i] = files

    new_data_dict = {}
    for i, v in data_dict.items():
        for j, filepath in enumerate(v):
            with open(filepath, "r") as f:
                v[j] = json.load(f)

            new_key = filepath.name.replace(".json", "")
            new_key = new_key.replace(f"_manual_{manual_index}", "")
            new_key = new_key.replace("treelstm", "TreeLSTM")
            new_key = new_key.replace("mathfunc", "MathFunc")
            new_key = new_key.replace("cpu", "CPU")
            new_key = new_key.replace("cuda", "GPU:")
            new_key = new_key.replace("_sync", "_Sync")
            new_key = new_key.replace("_async", "_Async")
            new_key = new_key.split("_")

            if new_key[1] != "Async":
                continue
            new_key.pop(1)
            new_key = tuple(new_key)

            new_value = []
            [new_value.extend(k) for k in v[j]]
            new_value = np.array(new_value) * 1e3
            new_data_dict[new_key] = {
                "mean": np.mean(new_value),
                "std": np.std(new_value),
                "min": np.min(new_value),
                "max": np.max(new_value),
            }

    keys = list(new_data_dict.keys())
    keys = sorted(keys)

    for i in keys:
        print(i, new_data_dict[i])

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.tight_layout(pad=3.0)
    fig.subplots_adjust(top=0.85)

    for i, key in enumerate(keys):
        mean = new_data_dict[key]["mean"]
        std = new_data_dict[key]["std"]
        diff = new_data_dict[key]["max"] - new_data_dict[key]["min"]
        err = min(std, diff / 2, mean - new_data_dict[key]["min"])
        name = f"{key[1]} - {key[2]}"
        if key[0] == "MathFunc":
            ax[0].bar(name, mean, yerr=err, **default_plot_params)
        elif key[0] == "TreeLSTM":
            ax[1].bar(name, mean, yerr=err, **default_plot_params)
        else:
            raise ValueError(f"Unknown key: {key}")

    ax[0].set_title("MathFunc")
    ax[1].set_title("TreeLSTM")
    ax[0].set_ylabel("Time [ms]")
    ax[1].set_ylabel("Time [ms]")

    if manual_index == 1:
        fig.suptitle("Left portion of the tree on Device 0 "
                     "and Right portion of the tree on Device 1")
    elif manual_index == 2:
        fig.suptitle("All state calculations on Device 0 "
                     "and All output calculations on Device 1")
    else:
        fig.suptitle(f"Manual {manual_index} distribution")

    plt.tight_layout()
    plt.savefig(
        get_plot_path().joinpath(f"manual_{manual_index}_distribution.png"),
        bbox_inches='tight',
        dpi=300
    )
    plt.show()


if __name__ == '__main__':
    a100_path = get_path("_logs_a100")

    plot_manual_distribution({"a100": a100_path}, manual_index=1)
    plot_manual_distribution({"a100": a100_path}, manual_index=2)
