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
        if (len(root.children) > 0):
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
            prev_depth = depth
        elif (depth < prev_depth):
            tracer_idx = i+1
            while (tracer_idx < len(global_dict.keys()) and global_dict[tracer_idx]["depth"] >= depth):
                tracer_idx += 1
            if tracer_idx >= len(global_dict.keys()):
                parent_list.append(-1)
            else:
                parent_list.append(tracer_idx)
            prev_depth = depth
        else:
            assert(depth > prev_depth)
            parent_list.append(i-1)
            prev_depth = depth
    return parent_list




class tree_lstm_dataset(Dataset):
    def __init__(self, tree_dataset, max_length):
        self.data_source = tree_dataset.trees
        self.max_length = max_length

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        #first row : depth
        #second row: parent index, if root, -1
        tree_local = self.data_source[idx]
        res = torch.zeros([2, self.max_length])
        global_dict = {}
        start_index = 0
        begin_depth = 0
        node_counter = in_order_traverse(tree_local, start_index, global_dict, begin_depth)
        parent_list = parent_traverse(global_dict)
        
        depth_list = []
        info_list = []

        for i in range(node_counter):
            #print("tracer1")
            depth = global_dict[i]["depth"]
            #print("tracer2")
            value = global_dict[i]["value"]
            #print("tracer3")
            parent_idx = parent_list[i]
            #print("tracer4")
            
            depth_list.append(depth)
            info_list.append(value)

        depth_list = torch.tensor(depth_list)
        parent_list = torch.tensor(parent_list)
        print(len(depth_list), len(parent_list))
        res[0, :node_counter] = depth_list
        res[1, :node_counter] = parent_list
        return res
        
            




def load_tree_lstm_with_depth_test():
    import ipdb
    device = torch.device("cuda" if (
        False and torch.cuda.is_available()) else "cpu")
    model, train_dataset, dev_dataset, test_dataset = load_tree_lstm(device)
    tree = train_dataset[0][0]
    global_dict = {}
    start_index = in_order_traverse(tree, 0, global_dict, 0)
    parent_list = parent_traverse(global_dict)
    new_dataset = tree_lstm_dataset(train_dataset, 10000)
    ipdb.set_trace()
    pass

#load_tree_lstm_with_depth_test()

