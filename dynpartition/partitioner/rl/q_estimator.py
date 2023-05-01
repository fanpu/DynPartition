import math

import torch


class FullyConnectedModel(torch.nn.Module):
    def __init__(self, input_size, output_shape, num_layers=4):
        super().__init__()
        if isinstance(input_size, tuple):
            input_size = math.prod(input_size)

        self.input_size = input_size
        self.output_shape = output_shape

        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Flatten(start_dim=0))
        self.layers.append(torch.nn.Linear(input_size, 16))
        self.layers.append(torch.nn.ReLU())
        for _ in range(num_layers - 2):
            self.layers.append(torch.nn.Linear(16, 16))
            self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Linear(16, math.prod(output_shape)))
        # no activation output layer

        # initialization
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x.reshape(self.output_shape)
