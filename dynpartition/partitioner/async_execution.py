import asyncio
import copy
import math
import sys
from functools import partial
from typing import Union, List, Dict

import aiocells
import numpy as np
import torch
from tqdm import tqdm

from dynpartition.dataset.accuracy import sentiment_accuracy_score
from dynpartition.dataset.load import load_math_model
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
        model_dict: Dict[torch.device, TREE_MODELS],
) -> torch.Tensor:
    nodes = tree.topological_sort()
    for node in nodes:
        model = model_dict[node.device]

        if node.is_leaf():
            execute_leaf(
                node=node,
                leaf_module=model.leaf_module,
                embedding_model=model.embedding_model
            )
        else:
            execute_non_leaf(
                node=node,
                composer=model.composer,
            )

        execute_output(
            node=node,
            output_module=model.output_module,
        )

    return tree.output


@torch.no_grad()
def async_tree_execution(
        tree: Tree,
        model_dict: Dict[torch.device, TREE_MODELS],
) -> torch.Tensor:
    graph = aiocells.DependencyGraph()
    state_functions = {}
    for node in tree.topological_sort():
        model = model_dict[node.device]

        if node.is_leaf():
            calculate_state = partial(
                execute_leaf,
                node=node,
                leaf_module=model.leaf_module,
                embedding_model=model.embedding_model
            )
        else:
            calculate_state = partial(
                execute_non_leaf,
                node=node,
                composer=model.composer,
            )

        calculate_output = partial(
            execute_output,
            node=node,
            output_module=model.output_module,
        )

        state_functions[id(node)] = calculate_state
        graph.add_node(calculate_state)
        graph.add_node(calculate_output)
        graph.add_precedence(calculate_state, calculate_output)
        for child in node.children:
            graph.add_precedence(state_functions[id(child)], calculate_state)

    asyncio.run(aiocells.async_compute_concurrent(graph))
    return tree.output


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

    for idx in tqdm(
        range(len(dataset)),
        desc=f'Testing {execution_strategy}',
        ascii=True,
        mininterval=0.5,
    ):
        tree = dataset[idx]

        for i in tree.get_all_nodes():
            assert i.device in devices

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


def main():
    # import lovely_tensors
    # lovely_tensors.monkey_patch()
    print("Testing...")
    print()
    device = torch.device(
        "cuda" if (False and torch.cuda.is_available()) else "cpu"
    )
    devices = ["cpu", "cuda:0"]

    for i in range(len(devices)):
        devices[i] = torch.device(devices[i])

    model, dataset = load_math_model(device)
    for tree in dataset:
        for i in tree.get_all_nodes():
            i.device = devices[math.floor(np.random.rand() * len(devices))]

    test_model_with(model, dataset[:1000], devices, 'sync')
    test_model_with(model, dataset[:1000], devices, 'async')

    # model, train_dataset, dev_dataset, test_dataset = load_tree_lstm(device)
    # dev_acc = test_model_with(
    #     model,
    #     dev_dataset.trees[:100],
    #     ["cpu", "cuda"],
    #     execution_strategy='sync'
    # )
    # print(f"TreeLSTM Dev accuracy: {dev_acc * 100:.4f}%")


if __name__ == '__main__':
    main()
