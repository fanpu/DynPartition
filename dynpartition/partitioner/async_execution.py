import copy
import math
import sys
from typing import Union, List, Dict

import numpy as np
import torch
from tqdm import tqdm

from dynpartition.dataset.accuracy import sentiment_accuracy_score
from dynpartition.dataset.load import load_math_model, load_tree_lstm
from dynpartition.dataset.tree import Tree
from dynpartition.models.MathFuncSolver import MathFuncSolver, \
    MathBinaryTreeLeafModule, MathCheckModule, MathBinaryTreeComposer
from dynpartition.models.TreeLSTM import TreeLSTMSentiment, \
    BinaryTreeLeafModule, SentimentModule, BinaryTreeComposer
from dynpartition.partitioner.partitioner_utils import tensors_to_device


TREE_MODELS = Union[MathFuncSolver, TreeLSTMSentiment]


def execute_leaf(
        node: Tree,
        leaf_module: Union[MathBinaryTreeLeafModule, BinaryTreeLeafModule],
        embedding_model: torch.nn.Embedding = None
):
    if isinstance(leaf_module, MathBinaryTreeLeafModule):
        value = tensors_to_device(node.device, torch.tensor(node.value))
        node.state = leaf_module.forward(value)
    elif isinstance(leaf_module, BinaryTreeLeafModule):
        value = torch.tensor(node.value)
        value = tensors_to_device(node.device, value)
        x = torch.unsqueeze(embedding_model(value), 1).T
        node.state = leaf_module.forward(x)
    else:
        raise NotImplementedError("Unknown module type")


def execute_non_leaf(
        node: Tree,
        composer: Union[MathBinaryTreeComposer, BinaryTreeComposer],
):
    states = sum([child.state for child in node.children], ())
    states = tensors_to_device(node.device, states)

    if isinstance(composer, MathBinaryTreeComposer):
        node.state = composer.forward(node.layer, *states)
    elif isinstance(composer, BinaryTreeComposer):
        node.state = composer.forward(*states)
    else:
        raise NotImplementedError("Unknown module type")


def execute_output(
        node: Tree,
        output_module: Union[MathCheckModule, SentimentModule],
):
    state = tensors_to_device(node.device, node.state)
    node.output = output_module.forward(*state)


@torch.no_grad()
def sync_tree_execution(
        tree: Tree,
        model: Dict[torch.device, TREE_MODELS],
) -> torch.Tensor:
    nodes = tree.topological_sort()
    for node in nodes:
        if node.is_leaf():
            execute_leaf(
                node=node,
                leaf_module=model[node.device].leaf_module,
                embedding_model=model[node.device].embedding_model
            )
        else:
            execute_non_leaf(
                node=node,
                composer=model[node.device].composer,
            )

        execute_output(
            node=node,
            output_module=model[node.device].output_module,
        )

    return tree.output


@torch.no_grad()
def async_tree_execution(
        tree: Tree,
        models_dict: Dict[torch.device, TREE_MODELS],
) -> torch.Tensor:
    pass
    # threads = []
    # for node in tree.nodes:
    #     if node.num_children == 0:
    #         threads.append(threading.Thread(
    #             target=execute_leaf,
    #             args=(model, node)
    #         ))
    #     else:
    #         threads.append(threading.Thread(
    #             target=execute_non_leaf,
    #             args=(model, node)
    #         ))
    # for thread in threads:
    #     thread.start()
    # for thread in threads:
    #     thread.join()
    #
    # threads = []
    # for node in tree.nodes:
    #     threads.append(threading.Thread(
    #         target=execute_output,
    #         args=(model, node)
    #     ))
    # for thread in threads:
    #     thread.start()
    # for thread in threads:
    #     thread.join()


@torch.no_grad()
def test_model_with(
        model: [MathFuncSolver, TreeLSTMSentiment],
        dataset: List[Tree],
        devices: List[Union[str, torch.device]],
        execution_strategy: str = 'async',
):
    if not isinstance(model, (MathFuncSolver, TreeLSTMSentiment)):
        raise TypeError(f'Unknown model type: {type(model)}')

    if execution_strategy not in ['async', 'sync']:
        raise ValueError(f'Unknown execution strategy: {execution_strategy}')

    model.eval()
    predictions = torch.zeros(len(dataset))

    for i in range(len(devices)):
        devices[i] = torch.device(devices[i])

    # replicate model
    models_dict: Dict[torch.device, TREE_MODELS] = {}
    for i in range(len(devices)):
        new_model = copy.deepcopy(model)
        new_model.eval()
        models_dict[devices[i]] = new_model.to(devices[i])

    for idx in tqdm(range(len(dataset)), desc=f'Testing ', ascii=True):
        tree = dataset[idx]

        for i in tree.get_all_nodes():
            i.device = devices[math.floor(np.random.rand() * len(devices))]

        if execution_strategy == 'async':
            output = async_tree_execution(tree, models_dict)
        else:  # sync
            output = sync_tree_execution(tree, models_dict)

        if isinstance(model, MathFuncSolver):
            predictions[idx] = output
        else:  # TreeLSTMSentiment
            output[:, 1] = -9999
            _, pred = torch.max(output, 1)
            predictions[idx] = pred

    labels = torch.tensor([tree.label for tree in dataset])
    labels = labels.to(predictions.device).type(predictions.dtype)
    acc = sentiment_accuracy_score(predictions, labels)

    sys.stdout.flush()
    sys.stderr.flush()
    return acc


if __name__ == '__main__':
    # import lovely_tensors
    # lovely_tensors.monkey_patch()
    print("Testing...")
    print()
    device = torch.device(
        "cuda" if (False and torch.cuda.is_available()) else "cpu"
    )

    model, dataset = load_math_model(device)
    math_acc = test_model_with(
        model,
        dataset[:100],
        ["cpu", "cuda"],
        execution_strategy='async'
    )
    print(f"Math accuracy: {math_acc * 100:.4f}%")

    # model, train_dataset, dev_dataset, test_dataset = load_tree_lstm(device)
    # dev_acc = test_model_with(
    #     model,
    #     dev_dataset.trees[:100],
    #     ["cpu", "cuda"],
    #     execution_strategy='sync'
    # )
    # print(f"TreeLSTM Dev accuracy: {dev_acc * 100:.4f}%")
