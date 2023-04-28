import copy
import math
from typing import List, Optional, Union

import numpy as np
import torch

from dynpartition.dataset.tree import Tree


def single_device_run(
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


def random_distribution(
        trees: List[Tree],
        devices: List[Union[str, torch.device]],
) -> List[Tree]:
    for i in range(len(devices)):
        devices[i] = torch.device(devices[i])

    trees = copy.deepcopy(trees)
    for tree in trees:
        for i in tree.get_all_nodes():
            rand1 = math.floor(np.random.rand() * len(devices))
            rand2 = math.floor(np.random.rand() * len(devices))
            i.device_for_state = devices[rand1]
            i.device_for_output = devices[rand2]

    return trees