import math

import numpy as np
import torch


def random_distribution(dataset, devices):
    for i in range(len(devices)):
        devices[i] = torch.device(devices[i])

    for tree in dataset:
        for i in tree.get_all_nodes():
            rand1 = math.floor(np.random.rand() * len(devices))
            rand2 = math.floor(np.random.rand() * len(devices))
            i.device_for_state = devices[rand1]
            i.device_for_output = devices[rand2]
