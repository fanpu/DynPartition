import copy
from typing import List, Optional, Union

import numpy as np
import torch

from dynpartition.dataset.load import load_tree_lstm, load_math_model
from dynpartition.dataset.tree import Tree
from dynpartition.partitioner.async_execution import test_model_with
from dynpartition.partitioner.partitioner_utils import ALL_DEVICES


def run_random_distribution(
        trees: List[Tree],
        devices: List[Union[str, torch.device]],
        max_layer_per_device: Optional[List[int]] = None
):
    modules = {"leaf_module", "composer", "output_module"}
    max_layer_per_device = copy.deepcopy(max_layer_per_device)
    if max_layer_per_device is None:
        max_layer_per_device = [9999] * len(devices)

    if len(devices) != len(max_layer_per_device):
        raise ValueError("The length of devices and max_layer_per_device "
                         "must be the same")

    for i in range(len(devices)):
        devices[i] = torch.device(devices[i])
        max_layer_per_device[i] = int(max(max_layer_per_device[i], 0))
        max_layer_per_device[i] = int(min(max_layer_per_device[i], 3))

    if sum(max_layer_per_device) < 3:
        raise ValueError("The sum of max_layer_per_device must be at least 3")

    device_availability = {}
    allocated_module_list = {}
    for i in range(len(devices)):
        if max_layer_per_device[i] <= 0:
            continue

        device_availability[devices[i]] = max_layer_per_device[i]
        allocated_module_list[devices[i]] = set()

    for i in modules:
        choice = np.random.choice(list(device_availability.keys()))
        allocated_module_list[choice].add(i)
        device_availability[choice] -= 1

        if device_availability[choice] <= 0:
            device_availability.pop(choice)

    while len(device_availability) > 0:
        choice = np.random.choice(list(device_availability.keys()))
        module = np.random.choice(list(modules - allocated_module_list[choice]))
        allocated_module_list[choice].add(module)
        device_availability[choice] -= 1

        if device_availability[choice] <= 0:
            device_availability.pop(choice)

    module_to_device = {}
    for device, module_list in allocated_module_list.items():
        for module in module_list:
            if module not in module_to_device:
                module_to_device[module] = []

            module_to_device[module].append(device)

    trees = copy.deepcopy(trees)
    for tree in trees:
        for i in tree.get_all_nodes():
            if i.is_leaf():
                i.device_for_state = np.random.choice(
                    module_to_device["leaf_module"]
                )
            else:
                i.device_for_state = np.random.choice(
                    module_to_device["composer"]
                )

            i.device_for_output = np.random.choice(
                module_to_device["output_module"]
            )

    return trees


def _main():
    print("Testing...")
    print()
    devices = ALL_DEVICES

    math_model, dataset = load_math_model()
    tree_lstm, train_dataset, dev_dataset, test_dataset = load_tree_lstm()
    tree_dataset = dev_dataset.trees

    if len(devices) >= 3:
        print("MathFunc on Random Distribution with 1 module per device")
        trees = run_random_distribution(dataset, devices, [1]*len(devices))
        test_model_with(math_model, trees[:1000], devices, 'sync')
        test_model_with(math_model, trees[:1000], devices, 'async')

        print("TreeLSTM on Random Distribution with 1 module per device")
        trees = run_random_distribution(tree_dataset, devices, [1]*len(devices))
        test_model_with(tree_lstm, trees[:500], devices, 'sync')
        test_model_with(tree_lstm, trees[:500], devices, 'async')

    if len(devices) >= 2:
        print("MathFunc on Random Distribution with 2 module per device")
        trees = run_random_distribution(dataset, devices, [2]*len(devices))
        test_model_with(math_model, trees[:1000], devices, 'sync')
        test_model_with(math_model, trees[:1000], devices, 'async')

        print("TreeLSTM on Random Distribution with 2 module per device")
        trees = run_random_distribution(tree_dataset, devices, [2]*len(devices))
        test_model_with(tree_lstm, trees[:500], devices, 'sync')
        test_model_with(tree_lstm, trees[:500], devices, 'async')

    print("MathFunc on Random Distribution with 3 module per device")
    trees = run_random_distribution(dataset, devices, [3] * len(devices))
    test_model_with(math_model, trees[:1000], devices, 'sync')
    test_model_with(math_model, trees[:1000], devices, 'async')

    print("TreeLSTM on Random Distribution with 3 module per device")
    trees = run_random_distribution(tree_dataset, devices, [3] * len(devices))
    test_model_with(tree_lstm, trees[:500], devices, 'sync')
    test_model_with(tree_lstm, trees[:500], devices, 'async')


if __name__ == '__main__':
    _main()
