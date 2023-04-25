# TODO: Vivswan
# https://github.com/Vivswan/TreeLSTMSentiment
# Branch: updates_for_python3_and_pytorch_2.0.0
# Data: https://drive.google.com/file/d/1F7Xeb4sBMJ3nsZ5Dj1LJh-XySaD9uMju/view?usp=share_link
# Paper: https://arxiv.org/abs/1503.00075
# Survey: https://arxiv.org/abs/2102.04906
# Constituency Parsing vs Dependency Parsing: https://www.baeldung.com/cs/constituency-vs-dependency-parsing

import torch
import torch.nn as nn

from dynpartition.util.tree import Tree


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
    def __init__(self, cuda, in_dim, mem_dim, num_classes, criterion):
        super(BinaryTreeLSTM, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.criterion = criterion

        self.leaf_module = BinaryTreeLeafModule(cuda, in_dim, mem_dim)
        self.composer = BinaryTreeComposer(cuda, in_dim, mem_dim)
        self.output_module = SentimentModule(cuda, mem_dim, num_classes)

    def forward(self, tree: Tree, inputs: torch.Tensor):
        if tree.num_children == 0:
            tree.state = self.leaf_module.forward(inputs[tree.idx - 1])
        else:
            for child in tree.children:
                self.forward(child, inputs)

            states = sum([child.state for child in tree.children], ())
            tree.state = self.composer.forward(*states)

        tree.output = self.output_module.forward(*tree.state)
        return tree.output


class TreeLSTMSentiment(nn.Module):
    def __init__(self, cuda, vocab_size, in_dim, mem_dim, num_classes, model_name, criterion):
        super(TreeLSTMSentiment, self).__init__()
        self.cudaFlag = cuda
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.tree_module = BinaryTreeLSTM(cuda, in_dim, mem_dim, num_classes, criterion)
        self.output_module = self.tree_module.output_module

    def forward(self, tree, inputs):
        return self.tree_module(tree, inputs)
