# TODO: Fanpu
import torchvision.models as models
import timeit
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.resnet import ResNet, Bottleneck
import torch
import torch.nn as nn
import torch.optim as optim
import ipdb

from torchvision.models.resnet import ResNet, Bottleneck

num_classes = 1000

DEVICE_0 = 'cuda:0'
DEVICE_1 = 'cpu'
DEVICES = [DEVICE_0, DEVICE_1]


class ModelParallelResNet50(ResNet):
    # Assume split between 2 devices
    def __init__(self, partition_layer, *args, **kwargs):
        """
        Args:
            partition_layer (int): The first layer [0-indexed] that 
            the second device will start processing
        """
        super(ModelParallelResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)

        layers = [
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.avgpool,
            self.fc
        ]
        assert partition_layer < len(layers) and partition_layer > 0

        self.seq1 = nn.Sequential(
            *layers[:partition_layer]
        ).to(DEVICE_0)

        self.seq2 = nn.Sequential(
            *layers[partition_layer:]
        ).to(DEVICE_1)

    def forward(self, x):
        x = self.seq2(self.seq1(x).to(DEVICE_1))
        return self.fc(x.view(x.size(0), -1))


class PipelineParallelResNet50(ModelParallelResNet50):
    def __init__(self, partition_layer, split_size=20, *args, **kwargs):
        super(PipelineParallelResNet50, partition_layer,
              self).__init__(*args, **kwargs)
        self.split_size = split_size

    def forward(self, x):
        splits = iter(x.split(self.split_size, dim=0))
        s_next = next(splits)
        s_prev = self.seq1(s_next).to(DEVICE_1)
        ret = []

        for s_next in splits:
            # A. ``s_prev`` runs on ``cpu``
            s_prev = self.seq2(s_prev)
            ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

            # B. ``s_next`` runs on ``cuda:0``, which can run concurrently with A
            s_prev = self.seq1(s_next).to(DEVICE_1)

        s_prev = self.seq2(s_prev)
        ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

        return torch.cat(ret)


num_batches = 3
batch_size = 5  # 120
image_w = 128
image_h = 128


def train(model):
    model.train(True)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    one_hot_indices = torch.LongTensor(batch_size) \
                           .random_(0, num_classes) \
                           .view(batch_size, 1)

    for _ in range(num_batches):
        # generate random inputs and labels
        inputs = torch.randn(batch_size, 3, image_w, image_h)
        labels = torch.zeros(batch_size, num_classes) \
                      .scatter_(1, one_hot_indices, 1)

        # run forward pass
        optimizer.zero_grad()
        outputs = model(inputs.to(DEVICE_0))

        # run backward pass
        labels = labels.to(outputs.device)
        loss_fn(outputs, labels).backward()
        optimizer.step()


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


sg_mean, sg_std = single_gpu()
mp_mean, mp_std = model_parallelism()
pp_mean, pp_std = pipeline_parallelism()

plot([mp_mean, sg_mean, pp_mean],
     [mp_std, sg_std, pp_std],
     ['Model Parallel', 'Single GPU', 'Pipelining Model Parallel'],
     'mp_vs_rn_vs_pp.png')

print("HI")
