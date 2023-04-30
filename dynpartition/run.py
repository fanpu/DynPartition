from dynpartition.dataset.load import load_tree_lstm, load_math_model
from dynpartition.partitioner.async_execution import test_model_with
from dynpartition.partitioner.run_random_distribution import \
    run_random_distribution
from dynpartition.partitioner.run_single_device import run_single_device


def main():
    print("Testing...")
    print()
    device = "cuda:0"
    devices = ["cpu", "cuda:0"]

    model, dataset = load_math_model()

    print("MathFunc on Single Device")
    trees = run_single_device(dataset, device)
    test_model_with(model, trees[:1000], [device], 'sync')
    test_model_with(model, trees[:1000], [device], 'async')

    print("MathFunc on Multiple with Random Distribution")
    trees = run_random_distribution(dataset, devices)
    test_model_with(model, trees[:1000], devices, 'sync')
    test_model_with(model, trees[:1000], devices, 'async')

    model, train_dataset, dev_dataset, test_dataset = load_tree_lstm()

    print("TreeLSTM on Single Device")
    trees = run_single_device(dev_dataset.trees, device)
    test_model_with(model, trees[:500], [device], 'sync')
    test_model_with(model, trees[:500], [device], 'async')

    print("TreeLSTM on Multiple with Random Distribution")
    trees = run_random_distribution(dev_dataset.trees, devices)
    test_model_with(model, trees[:500], devices, 'sync')
    test_model_with(model, trees[:500], devices, 'async')


if __name__ == '__main__':
    main()
