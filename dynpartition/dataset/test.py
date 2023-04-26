import sys
from typing import List

import torch
from tqdm import tqdm

from dynpartition.dataset.accuracy import sentiment_accuracy_score
from dynpartition.dataset.load import load_math_model, load_tree_lstm
from dynpartition.dataset.sst_dataset import SSTDataset
from dynpartition.dataset.tree import Tree
from dynpartition.models.MathFuncSolver import MathFuncSolver
from dynpartition.models.TreeLSTM import TreeLSTMSentiment


@torch.no_grad()
def test_math_model(model: MathFuncSolver, dataset: List[Tree]):
    model.eval()
    predictions = torch.zeros(len(dataset))

    for idx in tqdm(range(len(dataset)), desc=f'Testing ', ascii=True):
        tree = dataset[idx]

        output = model(tree)
        predictions[idx] = output

    labels = torch.tensor([tree.label for tree in dataset])
    labels = labels.to(predictions.device).type(predictions.dtype)
    acc = sentiment_accuracy_score(predictions, labels)

    sys.stdout.flush()
    sys.stderr.flush()
    return acc


@torch.no_grad()
def test_tree_lstm(model: TreeLSTMSentiment, dataset: SSTDataset):
    model.eval()
    predictions = torch.zeros(len(dataset))

    for idx in tqdm(range(len(dataset)), desc=f'Testing ', ascii=True, mininterval=1):
        tree = dataset.trees[idx]

        output = model(tree)  # size(1,5)

        output[:, 1] = -9999  # no need middle (neutral) value
        _, pred = torch.max(output, 1)
        predictions[idx] = pred

    acc = sentiment_accuracy_score(predictions, dataset.labels)
    sys.stdout.flush()
    sys.stderr.flush()
    return acc


if __name__ == '__main__':
    import lovely_tensors

    lovely_tensors.monkey_patch()
    print("Testing...")
    print()
    device = torch.device("cuda" if (True and torch.cuda.is_available()) else "cpu")

    model, dataset = load_math_model(device)
    model.to(device)
    math_acc = test_math_model(model, dataset)
    print(f"Math accuracy: {math_acc * 100:.4f}%")

    print()
    print()
    model, train_dataset, dev_dataset, test_dataset = load_tree_lstm(device)
    model.to(device)
    dev_acc = test_tree_lstm(model, dev_dataset)
    print(f"TreeLSTM Dev accuracy: {dev_acc * 100:.4f}%")
