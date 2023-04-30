import copy
from typing import List, Union

import torch
from dynpartition.dataset.load import load_tree_lstm, load_math_model
from dynpartition.partitioner.async_execution import test_model_with

from dynpartition.dataset.tree import Tree


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
    # import lovely_tensors
    # lovely_tensors.monkey_patch()
    print("Testing...")
    print()
    device = torch.device(
        "cuda" if (False and torch.cuda.is_available()) else "cpu"
    )

    math_model, dataset = load_math_model(device)
    tree_lstm, train_dataset, dev_dataset, test_dataset = load_tree_lstm(device)

    print("MathFunc on Single Device")
    trees = run_single_device(dataset, device)
    test_model_with(math_model, trees[:1000], [device], 'sync')
    test_model_with(math_model, trees[:1000], [device], 'async')

    print("TreeLSTM on Single Device")
    trees = run_single_device(dev_dataset.trees, device)
    test_model_with(tree_lstm, trees[:500], [device], 'sync')
    test_model_with(tree_lstm, trees[:500], [device], 'async')


if __name__ == '__main__':
    _main()
