# https://github.com/Vivswan/TreeLSTMSentiment
# Branch: updates_for_python3_and_pytorch_2.0.0
# Paper: https://arxiv.org/abs/1503.00075
# Survey: https://arxiv.org/abs/2102.04906
# Constituency Parsing vs Dependency Parsing:
# https://www.baeldung.com/cs/constituency-vs-dependency-parsing

import torch
import torch.nn as nn
from typing import Optional

from dynpartition.dataset.tree import Tree
from dynpartition.partitioner.partitioner_utils import tensors_to_device


class BinaryTreeLeafModule(nn.Module):
    def __init__(self, cuda, in_dim, mem_dim):
        super(BinaryTreeLeafModule, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.cx = nn.Linear(self.in_dim, self.mem_dim)
        self.ox = nn.Linear(self.in_dim, self.mem_dim)
        if self.cudaFlag:
            self.cx = self.cx.cuda()
            self.ox = self.ox.cuda()

    def forward(self, x):
        c = self.cx(x)
        o = torch.sigmoid(self.ox(x))
        h = o * torch.tanh(c)
        return c, h


class BinaryTreeComposer(nn.Module):
    def __init__(self, cuda, in_dim, mem_dim):
        super(BinaryTreeComposer, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        def new_gate():
            lh = nn.Linear(self.mem_dim, self.mem_dim)
            rh = nn.Linear(self.mem_dim, self.mem_dim)
            return lh, rh

        self.ilh, self.irh = new_gate()
        self.lflh, self.lfrh = new_gate()
        self.rflh, self.rfrh = new_gate()
        self.ulh, self.urh = new_gate()

        if self.cudaFlag:
            self.ilh = self.ilh.cuda()
            self.irh = self.irh.cuda()
            self.lflh = self.lflh.cuda()
            self.lfrh = self.lfrh.cuda()
            self.rflh = self.rflh.cuda()
            self.rfrh = self.rfrh.cuda()
            self.ulh = self.ulh.cuda()

    def forward(self, lc, lh, rc, rh):
        i = torch.sigmoid(self.ilh(lh) + self.irh(rh))
        lf = torch.sigmoid(self.lflh(lh) + self.lfrh(rh))
        rf = torch.sigmoid(self.rflh(lh) + self.rfrh(rh))
        update = torch.tanh(self.ulh(lh) + self.urh(rh))
        c = i * update + lf * lc + rf * rc
        h = torch.tanh(c)
        return c, h


class SentimentModule(nn.Module):
    def __init__(self, cuda, mem_dim, num_classes):
        super(SentimentModule, self).__init__()
        self.cudaFlag = cuda
        self.mem_dim = mem_dim
        self.num_classes = num_classes
        self.dropout = None
        self.l1 = nn.Linear(self.mem_dim, self.num_classes)
        self.logsoftmax = nn.LogSoftmax()
        if self.cudaFlag:
            self.l1 = self.l1.cuda()

    def forward(self, *vec):
        return self.logsoftmax(self.l1(vec[1]))


class BinaryTreeLSTM(nn.Module):
    def __init__(self, cuda, in_dim, mem_dim, num_classes, embedding_model):
        super(BinaryTreeLSTM, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.embedding_model = embedding_model
        self.leaf_module = BinaryTreeLeafModule(cuda, in_dim, mem_dim)
        self.composer = BinaryTreeComposer(cuda, in_dim, mem_dim)
        self.output_module = SentimentModule(cuda, mem_dim, num_classes)

    def forward(self, tree: Tree, device_allocations: Optional[dict] = None):
        """
        Device allocations: map from node traversal_index to device
        """
        if tree.num_children == 0:
            # Leaf Module
            value = torch.tensor(
                tree.value,
                device=device_allocations[tree.traversal_index] if device_allocations else self.embedding_model.weight.device
            )
            x = torch.unsqueeze(self.embedding_model(value), 1).T
            tree.state = self.leaf_module.forward(x)

        else:
            for child in tree.children:
                self.forward(child, device_allocations=device_allocations)

            # Non-leaf Module
            # TODO: don't want to convert tensor with .to()
            # if already on the same device
            states = sum([child.state for child in tree.children],
                         ())
            states = tensors_to_device(
                device_allocations[tree.traversal_index],
                states
            )
            # .to(device_allocations[tree.idx])
            tree.state = self.composer.forward(*states)

        # Output Module
        tree.output = self.output_module.forward(*tree.state)
        return tree.output


class TreeLSTMSentiment(nn.Module):
    def __init__(
            self,
            cuda,
            vocab_size,
            in_dim,
            mem_dim,
            num_classes,
            model_name,
            embedding_model
    ):
        super(TreeLSTMSentiment, self).__init__()
        self.cudaFlag = cuda
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.tree_module = BinaryTreeLSTM(
            cuda=cuda,
            in_dim=in_dim,
            mem_dim=mem_dim,
            num_classes=num_classes,
            embedding_model=embedding_model
        )

    @property
    def leaf_module(self) -> BinaryTreeLeafModule:
        return self.tree_module.leaf_module

    @property
    def composer(self) -> BinaryTreeComposer:
        return self.tree_module.composer

    @property
    def output_module(self) -> SentimentModule:
        return self.tree_module.output_module

    @property
    def embedding_model(self) -> nn.Embedding:
        return self.tree_module.embedding_model

    def forward(self, tree, device_allocations: Optional[dict] = None):
        return self.tree_module.forward(tree, device_allocations=device_allocations)
