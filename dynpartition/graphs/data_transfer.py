import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dynpartition.get_dir import get_path, get_plot_path


def plot_data_transfer_speeds(data_dict):
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
            if not file.name.startswith("data_transfer"):
                continue
            if not file.name.endswith(".json"):
                continue
            files.append(file)
        data_dict[i] = files

    for i, v in data_dict.items():
        for j, filepath in enumerate(v):
            with open(filepath, "r") as f:
                v[j] = json.load(f)['data_transfer_speeds']

        combined_data = {}
        for j in v:
            for key, value in j.items():
                if key not in combined_data:
                    combined_data[key] = []
                combined_data[key].extend(value)
        data_dict[i] = combined_data

    remove_keys = []
    for i in data_dict:
        if len(data_dict[i]) == 0:
            remove_keys.append(i)

    for i in remove_keys:
        del data_dict[i]

    for i in data_dict:
        combined_devices = {
            "cpu_cuda": [],
            "cuda_cuda": [],
        }
        for j in data_dict[i]:
            if "cpu" in j and "cuda" in j:
                combined_devices["cpu_cuda"].extend(data_dict[i][j])
            elif "cuda" in j:
                combined_devices["cuda_cuda"].extend(data_dict[i][j])
            else:
                raise ValueError("Unknown device")
        data_dict[i] = combined_devices

    for i in data_dict:
        for j in data_dict[i]:
            data_dict[i][j] = np.array(data_dict[i][j]) * 1e6
            data_dict[i][j] = {
                "mean": np.mean(data_dict[i][j]),
                "std": np.std(data_dict[i][j]),
                "min": np.min(data_dict[i][j]),
                "max": np.max(data_dict[i][j]),
            }

    for i in data_dict:
        for j in data_dict[i]:
            print(f"{i} {j}: {data_dict[i][j]}")

    # plot as bar chart with error bars
    plt.figure(figsize=(10, 5))
    plt.title("Data Transfer Speeds")
    for i in data_dict:
        for j in data_dict[i]:
            x_label = f"{i}: {j}".replace("_", " - ")
            mean = data_dict[i][j]["mean"]
            std = data_dict[i][j]["std"]
            diff = data_dict[i][j]["max"] - data_dict[i][j]["min"]
            err = min(std, diff / 2, mean - data_dict[i][j]["min"])
            plt.bar(x_label, mean, yerr=err, **default_plot_params)

    plt.ylabel("Transfer Time (µs) for (1000, 1000) Matrix")
    plt.savefig(
        get_plot_path().joinpath("data_transfer.png"),
        bbox_inches='tight',
        dpi=300
    )


if __name__ == '__main__':
    a100_path = get_path("_logs_a100")
    gtx1080ti_path = get_path("_logs_gtx1080ti")

    plot_data_transfer_speeds({
        "a100": a100_path,
        "gtx1080ti": gtx1080ti_path,
    })
