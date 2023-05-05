import math
from typing import List, Union

import numpy as np
import torch

from dynpartition.dataset.load import load_tree_lstm, load_math_model
from dynpartition.dataset.tree import Tree
from dynpartition.get_dir import save_log_json
from dynpartition.partitioner.time_measurements import timeit_dataset
from dynpartition.partitioner.utils import ALL_DEVICES


def tree_to_device(tree, device):
    for i in tree.get_all_nodes():
        if i.is_leaf():
            i.device_for_state = device
        else:
            i.device_for_state = device
        i.device_for_output = device


def run_manual_1(
        trees: List[Tree],
        devices: List[Union[str, torch.device]],
):
    # all left of root node in Device 1 and all right of root node in Device 2
    assert len(devices) >= 2
    for tree in trees:
        if tree.is_leaf():
            tree.device_for_state = devices[1]
        else:
            tree.device_for_state = devices[1]

        tree.device_for_output = devices[1]

        half_children = math.floor(len(tree.children) / 2)

        for i in tree.children[:half_children]:
            tree_to_device(i, devices[0])
        for i in tree.children[half_children:]:
            tree_to_device(i, devices[1])

    return trees


def run_manual_2(
        trees: List[Tree],
        devices: List[Union[str, torch.device]],
):
    # all state calculation in Device 1 and all output calculation in Device 2
    assert len(devices) >= 2
    for tree in trees:
        for i in tree.get_all_nodes():
            if i.is_leaf():
                i.device_for_state = devices[0]
            else:
                i.device_for_state = devices[0]

            i.device_for_output = devices[1]

    return trees


def _main():
    devices = ALL_DEVICES
    n_devices = len(devices)

    print("Testing...")
    math_model, dataset = load_math_model()
    tree_lstm, train_dataset, dev_dataset, test_dataset = load_tree_lstm()
    tree_dataset = dev_dataset.trees

    assert len(devices) >= 2
    assert devices[0] == 'cpu'
    assert devices[1].startswith('cuda')
    print(f"Distributed on {devices}")

    print("MathFunc on Manual Distribution with "
          "all left of root node in cpu and "
          "all right of root node in cuda:0")
    trees = run_manual_1(dataset, devices)
    sync_times = timeit_dataset(math_model, trees[:1000], devices, 'sync')
    async_times = timeit_dataset(math_model, trees[:1000], devices, 'async')
    print(f"Sync: {np.mean(sync_times)} +- {np.std(sync_times)}")
    print(f"Async: {np.mean(async_times)} +- {np.std(async_times)}")
    save_log_json(sync_times, name=f"mathfunc_manual_1_sync_cpu_cuda0")
    save_log_json(async_times, name=f"mathfunc_manual_1_async_cpu_cuda0")

    print("TreeLSTM on Manual Distribution with "
          "all left of root node in cpu and "
          "all right of root node in cuda:0")
    trees = run_manual_1(tree_dataset, devices)
    sync_times = timeit_dataset(tree_lstm, trees[:500], devices, 'sync')
    async_times = timeit_dataset(tree_lstm, trees[:500], devices, 'async')
    print(f"Sync: {np.mean(sync_times)} +- {np.std(sync_times)}")
    print(f"Async: {np.mean(async_times)} +- {np.std(async_times)}")
    save_log_json(sync_times, name=f"treelstm_manual_1_sync_cpu_cuda0")
    save_log_json(async_times, name=f"treelstm_manual_1_async_cpu_cuda0")

    print("MathFunc on Manual Distribution with "
          "all state calculation in cpu and "
          "all output calculation in cuda:0")
    trees = run_manual_2(dataset, devices)
    sync_times = timeit_dataset(math_model, trees[:1000], devices, 'sync')
    async_times = timeit_dataset(math_model, trees[:1000], devices, 'async')
    print(f"Sync: {np.mean(sync_times)} +- {np.std(sync_times)}")
    print(f"Async: {np.mean(async_times)} +- {np.std(async_times)}")
    save_log_json(sync_times, name=f"mathfunc_manual_2_sync_cpu_cuda0")
    save_log_json(async_times, name=f"mathfunc_manual_2_async_cpu_cuda0")

    print("TreeLSTM on Manual Distribution with "
          "all state calculation in cpu and "
          "all output calculation in cuda:0")
    trees = run_manual_2(tree_dataset, devices)
    sync_times = timeit_dataset(tree_lstm, trees[:500], devices, 'sync')
    async_times = timeit_dataset(tree_lstm, trees[:500], devices, 'async')
    print(f"Sync: {np.mean(sync_times)} +- {np.std(sync_times)}")
    print(f"Async: {np.mean(async_times)} +- {np.std(async_times)}")
    save_log_json(sync_times, name=f"treelstm_manual_2_sync_cpu_cuda0")
    save_log_json(async_times, name=f"treelstm_manual_2_async_cpu_cuda0")

    if not len(devices) >= 3:
        return

    devices = devices[1:]
    print(f"Distributed on {devices}")
    assert devices[0].startswith('cuda')
    assert devices[1].startswith('cuda')

    print("MathFunc on Manual Distribution with "
          "all left of root node in cuda:0 and "
          "all right of root node in cuda:1")
    trees = run_manual_1(dataset, devices)
    sync_times = timeit_dataset(math_model, trees[:1000], devices, 'sync')
    async_times = timeit_dataset(math_model, trees[:1000], devices, 'async')
    print(f"Sync: {np.mean(sync_times)} +- {np.std(sync_times)}")
    print(f"Async: {np.mean(async_times)} +- {np.std(async_times)}")
    save_log_json(sync_times, name=f"mathfunc_manual_1_sync_cuda0_cuda1")
    save_log_json(async_times, name=f"mathfunc_manual_1_async_cuda0_cuda1")

    print("TreeLSTM on Manual Distribution with "
          "all left of root node in cuda:0 and "
          "all right of root node in cuda:1")
    trees = run_manual_1(tree_dataset, devices)
    sync_times = timeit_dataset(tree_lstm, trees[:500], devices, 'sync')
    async_times = timeit_dataset(tree_lstm, trees[:500], devices, 'async')
    print(f"Sync: {np.mean(sync_times)} +- {np.std(sync_times)}")
    print(f"Async: {np.mean(async_times)} +- {np.std(async_times)}")
    save_log_json(sync_times, name=f"treelstm_manual_1_sync_cuda0_cuda1")
    save_log_json(async_times, name=f"treelstm_manual_1_async_cuda0_cuda1")

    print("MathFunc on Manual Distribution with "
          "all state calculation in cuda:0 and "
          "all output calculation in cuda:1")
    trees = run_manual_2(dataset, devices)
    sync_times = timeit_dataset(math_model, trees[:1000], devices, 'sync')
    async_times = timeit_dataset(math_model, trees[:1000], devices, 'async')
    print(f"Sync: {np.mean(sync_times)} +- {np.std(sync_times)}")
    print(f"Async: {np.mean(async_times)} +- {np.std(async_times)}")
    save_log_json(sync_times, name=f"mathfunc_manual_2_sync_cuda0_cuda1")
    save_log_json(async_times, name=f"mathfunc_manual_2_async_cuda0_cuda1")

    print("TreeLSTM on Manual Distribution with "
          "all state calculation in cuda:0 and "
          "all output calculation in cuda:1")
    trees = run_manual_2(tree_dataset, devices)
    sync_times = timeit_dataset(tree_lstm, trees[:500], devices, 'sync')
    async_times = timeit_dataset(tree_lstm, trees[:500], devices, 'async')
    print(f"Sync: {np.mean(sync_times)} +- {np.std(sync_times)}")
    print(f"Async: {np.mean(async_times)} +- {np.std(async_times)}")
    save_log_json(sync_times, name=f"treelstm_manual_2_sync_cuda0_cuda1")
    save_log_json(async_times, name=f"treelstm_manual_2_async_cuda0_cuda1")


if __name__ == '__main__':
    _main()
