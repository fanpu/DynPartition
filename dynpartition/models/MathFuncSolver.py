import torch.nn as nn


class MathFuncSolver(nn.Module):
    def __init__(self):
        super(MathFuncSolver, self).__init__()

    def forward(self, tree, x):
        return x
