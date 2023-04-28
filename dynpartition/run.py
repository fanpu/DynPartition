import torch

from dynpartition.dataset.load import load_tree_lstm, load_math_model
from dynpartition.partitioner.async_execution import test_model_with
from dynpartition.partitioner.partitioner import random_distribution, \
    single_device_run


def main():
    # import lovely_tensors
    # lovely_tensors.monkey_patch()
    print("Testing...")
    print()
    device = torch.device(
        "cuda" if (False and torch.cuda.is_available()) else "cpu"
    )
    devices = ["cpu", "cuda:0"]

    model, dataset = load_math_model(device)

    print("MathFunc on Single Device")
    trees = single_device_run(dataset, device)
    test_model_with(model, trees[:1000], devices, 'sync')
    test_model_with(model, trees[:1000], devices, 'async')

    print("MathFunc on Multiple with Random Distribution")
    trees = random_distribution(dataset, devices)
    test_model_with(model, trees[:1000], devices, 'sync')
    test_model_with(model, trees[:1000], devices, 'async')

    model, train_dataset, dev_dataset, test_dataset = load_tree_lstm(device)

    print("TreeLSTM on Single Device")
    trees = single_device_run(dev_dataset.trees, device)
    test_model_with(model, trees[:500], devices, 'sync')
    test_model_with(model, trees[:500], devices, 'async')

    print("TreeLSTM on Multiple with Random Distribution")
    trees = random_distribution(dev_dataset.trees, devices)
    test_model_with(model, trees[:500], devices, 'sync')
    test_model_with(model, trees[:500], devices, 'async')


if __name__ == '__main__':
    main()
