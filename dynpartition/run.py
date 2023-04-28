import torch

from dynpartition.dataset.load import load_tree_lstm, load_math_model
from dynpartition.partitioner.async_execution import test_model_with
from dynpartition.partitioner.partitioner import random_distribution


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
    random_distribution(dataset, devices)
    test_model_with(model, dataset[:1000], devices, 'sync')
    test_model_with(model, dataset[:1000], devices, 'async')

    model, train_dataset, dev_dataset, test_dataset = load_tree_lstm(device)
    random_distribution(dev_dataset.trees, devices)
    test_model_with(model, dev_dataset.trees[:100], devices, 'sync')
    test_model_with(model, dev_dataset.trees[:100], devices, 'async')


if __name__ == '__main__':
    main()
