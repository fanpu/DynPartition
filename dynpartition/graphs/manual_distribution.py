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

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey='row')
    fig.tight_layout(pad=3.0)
    fig.subplots_adjust(top=0.85)

    if manual_index == 1:
        fig.suptitle("All left nodes on Device 0 and right nodes on Device 1")
    if manual_index == 2:
        fig.suptitle("All state calculations on Device 0 "
                     "and all output calculations on Device 1")
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
    logs_path = get_path("_logs")
    a100_path = get_path("_logs_a100")
    gtx1080ti_path = get_path("_logs_gtx1080ti")

    plot_manual_distribution({"a100": logs_path}, manual_index=1)
    # plot_manual_distribution({"a100": a100_path}, manual_index=2)
