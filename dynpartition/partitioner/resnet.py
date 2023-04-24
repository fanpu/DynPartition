# TODO Fanpu
import torch
import torch.nn as nn

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
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            num_classes=num_classes,
            *args,
            **kwargs
        )

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
        ]
        assert len(layers) > partition_layer > 0

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
        super(PipelineParallelResNet50, self).__init__(partition_layer, *args, **kwargs)
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
