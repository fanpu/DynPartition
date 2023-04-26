# TODO: Fanpu

import os
import sys
import timeit

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm
from dqn import DQN_Agent
from scheduler_env import SchedulerEnv
from torchvision.models.resnet import Bottleneck, ResNet

from dynpartition.get_dir import get_plot_path
from dynpartition.dataset.load import load_tree_lstm

num_batches = 3
batch_size = 5  # 120
image_w = 128
image_h = 128


def train(model):
    model.train(True)

    for _ in range(num_batches):
        # generate random inputs
        inputs = torch.randn(batch_size, 3, image_w, image_h)

        # run forward pass
        outputs = model(inputs.to(DEVICE_0))


plt.switch_backend('Agg')

num_repeat = 10

stmt = "train(model)"


def plot(means, stds, labels, fig_name):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('ResNet50 Execution Time (Second)')
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)


partition_layer = 3


def single_gpu():
    setup = "import torchvision.models as models;" + \
            "model = models.resnet50(num_classes=num_classes).to(DEVICE_0)"
    sg_run_times = timeit.repeat(
        stmt, setup, number=1, repeat=num_repeat, globals=globals())
    sg_mean, sg_std = np.mean(sg_run_times), np.std(sg_run_times)

    return sg_mean, sg_std


def model_parallelism():
    """Model paralellism without pipelining"""
    setup = f"model = ModelParallelResNet50(partition_layer={partition_layer})"
    mp_run_times = timeit.repeat(
        stmt, setup, number=1, repeat=num_repeat, globals=globals())
    mp_mean, mp_std = np.mean(mp_run_times), np.std(mp_run_times)

    return mp_mean, mp_std


def pipeline_parallelism():
    """Model parallelism with pipelining"""
    setup = f"model = PipelineParallelResNet50(partition_layer={partition_layer})"
    pp_run_times = timeit.repeat(
        stmt, setup, number=1, repeat=num_repeat, globals=globals())
    pp_mean, pp_std = np.mean(pp_run_times), np.std(pp_run_times)

    return pp_mean, pp_std


# sg_mean, sg_std = single_gpu()
# mp_mean, mp_std = model_parallelism()
# pp_mean, pp_std = pipeline_parallelism()

# plot([mp_mean, sg_mean, pp_mean],
#      [mp_std, sg_std, pp_std],
#      ['Model Parallel', 'Single GPU', 'Pipelining Model Parallel'],
#      get_plot_path().joinpath('mp_vs_rn_vs_pp.png'))

def main():
    num_seeds = 1
    num_episodes = 100
    num_test_episodes = 5
    episodes_between_test = 5
    l = num_episodes // episodes_between_test
    res = np.zeros((num_seeds, l))
    agent = DQN_Agent(SchedulerEnv())

    reward_means = []
    for i in tqdm.tqdm(range(num_seeds)):
        for m in range(num_episodes):
            print("m", m)
            agent.train()
            if m % episodes_between_test == 0:
                print("Episode: {}".format(m))
                G = np.zeros(20)
                for k in range(num_test_episodes):
                    g = agent.test()
                    G[k] = g

                reward_mean = G.mean()
                reward_sd = G.std()
                print("The test reward for episode {0} is {1} with sd of {2}.".format(
                    m, reward_mean, reward_sd))
                reward_means.append(reward_mean)

        print(reward_means)
        res[i] = np.array(reward_means)

    ks = np.arange(l) * episodes_between_test
    avs = np.mean(res, axis=0)
    maxs = np.max(res, axis=0)
    mins = np.min(res, axis=0)

    plt.fill_between(ks, mins, maxs, alpha=0.1)
    plt.plot(ks, avs, '-o', markersize=1)

    plt.xlabel('Episode', fontsize=15)
    plt.ylabel('Return', fontsize=15)

    plt.title("DynPartition Learning Curve", fontsize=24)
    plt.savefig(get_plot_path().joinpath("dynpartition_learning_curve.png"))


if __name__ == '__main__':
    device = torch.device("cuda" if (
        False and torch.cuda.is_available()) else "cpu")
    embedding_model, model, train_dataset, dev_dataset, test_dataset = load_tree_lstm(
        device)
    ipdb.set_trace()
    pass
    pass
    pass
    pass

    main()
