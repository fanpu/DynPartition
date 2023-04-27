import os
import sys
import timeit

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm
from scheduler_env import SchedulerEnv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from dynpartition.get_dir import get_plot_path
from dynpartition.dataset.load import load_tree_lstm

def in_order_traverse(root, start_index, global_dict, depth_tracer):
    if len(root.children) == 0:
        local_dict = {}
        local_dict["depth"] = depth_tracer
        local_dict["value"] = root.value
        global_dict[start_index] = local_dict
        start_index += 1
        return start_index
    else:
        local_dict = {}
        local_dict["depth"] = depth_tracer
        local_dict["value"] = root.value
        if (len(root.children) == 1):
            left_child = root.children[0]
            start_index = in_order_traverse(left_child, start_index, global_dict, depth_tracer + 1)
        global_dict[start_index] = local_dict
        start_index += 1
        if (len(root.children) == 2):
            right_child = root.children[1]
            start_index = in_order_traverse(right_child, start_index, global_dict, depth_tracer + 1)
        return start_index


def parent_traverse(global_dict):
    parent_list= []
    prev_depth = None
    for i in range(len(global_dict.keys())):
        depth = global_dict[i]["depth"]
        if (prev_depth == None):
            parent_list.append(i+1)
        elif (depth < prev_depth):
            parent_list.append(i-1)
        else:
            assert(depth > prev_depth)
            parent_list.append(i-1)
    return parent_list




class tree_lstm_dataset(Dataset):
    def __init__(self, tree_dataset, max_length):
        self.data_source = tree_dataset.trees
        self.max_length = max_length

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        tree_local = self.data_source[idx]
        res = torch.zeros([self.max_length, 2])
        global_dict = {}
        start_index = 0
        begin_depth = 0
        node_counter = in_order_traverse(tree_local, start_index, global_dict, begin_depth)
        parent_list = parent_traverse(global_dict)
        depth_list = []
        parent_list = []
        info_list = []
        for i in range(node_counter):
            depth = global_dict["depth"]
            value = global_dict["value"]
            parent_idx = parent_list[i]
            depth_list.append(depth)
            info_list.append(value)
            parent_list.append(parent_idx)
        depth_list = torch.tensor(depth_list)
        parent_list = torch.tensor(parent_list)
        res[0, :node_counter] = depth_list
        res[1, :node_counter] = parent_list
        return res
        
            




def load_tree_lstm_with_depth():
    import ipdb
    device = torch.device("cuda" if (
        False and torch.cuda.is_available()) else "cpu")
    embedding_model, model, train_dataset, dev_dataset, test_dataset = load_tree_lstm(device)
    
    ipdb.set_trace()
    pass

load_tree_lstm_with_depth()

