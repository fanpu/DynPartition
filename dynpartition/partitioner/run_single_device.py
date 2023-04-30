import copy
from typing import List, Union

import numpy as np
import torch
from dynpartition.dataset.load import load_tree_lstm, load_math_model
from dynpartition.partitioner.async_execution import test_model_with

from dynpartition.dataset.tree import Tree
from dynpartition.partitioner.partitioner_utils import ALL_DEVICES
from dynpartition.partitioner.time_measurements import timeit_dataset


def run_single_device(
        trees: List[Tree],
        device: Union[str, torch.device],
) -> List[Tree]:
    device = torch.device(device)
    trees = copy.deepcopy(trees)
    for tree in trees:
        for i in tree.get_all_nodes():
            i.device_for_state = device
            i.device_for_output = device

    return trees


def _main():
    print("Testing...")
    print()
    if "cpu" in ALL_DEVICES:
        device = "cpu"
    else:
        device = ALL_DEVICES[0]

    math_model, dataset = load_math_model()
    tree_lstm, train_dataset, dev_dataset, test_dataset = load_tree_lstm()

    print("MathFunc on Single Device")
    trees = run_single_device(dataset, device)
    sync_times = timeit_dataset(math_model, trees[:1000], [device], 'sync')
    async_times = timeit_dataset(math_model, trees[:1000], [device], 'async')
    print(f"Sync: {np.mean(sync_times)} +- {np.std(sync_times)}")
    print(f"Async: {np.mean(async_times)} +- {np.std(async_times)}")

    print("TreeLSTM on Single Device")
    trees = run_single_device(dev_dataset.trees, device)
    timeit_dataset(tree_lstm, trees[:500], [device], 'sync')
    timeit_dataset(tree_lstm, trees[:500], [device], 'async')


if __name__ == '__main__':
    _main()
