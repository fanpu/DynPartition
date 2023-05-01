import copy
import sys
import timeit
from typing import List, Union, Dict

import torch
from tqdm import tqdm

from dynpartition.dataset.tree import Tree
from dynpartition.models.MathFuncSolver import MathFuncSolver
from dynpartition.models.TreeLSTM import TreeLSTMSentiment
from dynpartition.partitioner.async_execution import TREE_MODELS, \
    async_tree_execution, sync_tree_execution


@torch.no_grad()
def for_time_measurement(
        model: [MathFuncSolver, TreeLSTMSentiment],
        tree: Tree,
        devices: List[Union[str, torch.device]],
        execution_strategy: str = 'async',
):
    if not isinstance(model, (MathFuncSolver, TreeLSTMSentiment)):
        raise TypeError(f'Unknown model type: {type(model)}')

    if execution_strategy not in ['async', 'sync']:
        raise ValueError(f'Unknown execution strategy: {execution_strategy}')

    model.eval()
    for i in range(len(devices)):
        devices[i] = torch.device(devices[i])

    # replicate model
    models_dict: Dict[torch.device, TREE_MODELS] = {}
    for i in range(len(devices)):
        new_model = copy.deepcopy(model)
        new_model.eval()
        models_dict[devices[i]] = new_model.to(devices[i])

    for i in tree.get_all_nodes():
        if isinstance(i.device_for_state, str):
            i.device_for_state = torch.device(i.device_for_state)
        if isinstance(i.device_for_output, str):
            i.device_for_output = torch.device(i.device_for_output)

        print(devices)
        assert i.device_for_state in devices
        assert i.device_for_output in devices

    def to_measure():
        if execution_strategy == 'async':
            _ = async_tree_execution(tree, models_dict)
        else:  # sync
            _ = sync_tree_execution(tree, models_dict)

    return to_measure


def timeit_dataset(
        model: [MathFuncSolver, TreeLSTMSentiment],
        dataset: List[Tree],
        devices: List[Union[str, torch.device]],
        execution_strategy: str = 'async',
        with_tqdm: bool = True,
        num_repeats: int = 10,
):
    if not isinstance(model, (MathFuncSolver, TreeLSTMSentiment)):
        raise TypeError(f'Unknown model type: {type(model)}')

    if execution_strategy not in ['async', 'sync']:
        raise ValueError(f'Unknown execution strategy: {execution_strategy}')

    model.eval()

    for i in range(len(devices)):
        devices[i] = torch.device(devices[i])

    # replicate model
    models_dict: Dict[torch.device, TREE_MODELS] = {}
    for i in range(len(devices)):
        new_model = copy.deepcopy(model)
        new_model.eval()
        models_dict[devices[i]] = new_model.to(devices[i])

    if with_tqdm:
        iterator = tqdm(
            range(len(dataset)),
            desc=f'Testing {execution_strategy}',
            ascii=True,
            mininterval=0.5,
        )
    else:
        iterator = range(len(dataset))

    for idx in range(len(dataset)):
        for i in dataset[idx].get_all_nodes():
            if isinstance(i.device_for_state, str):
                i.device_for_state = torch.device(i.device_for_state)
            if isinstance(i.device_for_output, str):
                i.device_for_output = torch.device(i.device_for_output)

            assert i.device_for_state in devices
            assert i.device_for_output in devices

    def create_to_measure(index):
        tree = dataset[index]

        def to_measure():
            if execution_strategy == 'async':
                _ = async_tree_execution(tree, models_dict)
            else:  # sync
                _ = sync_tree_execution(tree, models_dict)

        return to_measure

    run_times = []
    for idx in iterator:
        run_time = timeit.repeat(
            create_to_measure(idx),
            repeat=num_repeats,
            number=1,
            globals=globals(),
        )
        run_times.append(run_time)

    sys.stdout.flush()
    sys.stderr.flush()
    return run_times
