import torch
import torch.nn as nn

from dynpartition.models.generate_math_func import get_proper_math_tree, STRING_BINARY_OPS
from dynpartition.util.tree import Tree


class MathBinaryTreeLeafModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.one = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return (x * self.one,)


class MathBinaryTreeComposer(nn.Module):
    def __init__(self):
        super().__init__()
        self.one = nn.Parameter(torch.tensor(1.0))

    def forward(self, layer, *states):
        operation = getattr(torch, layer.lower(), None)

        if operation is None:
            raise ValueError(f"Operation {layer} not found in torch.")

        if layer in STRING_BINARY_OPS:
            if operation is torch.add:
                operation = torch.sum

            if operation is torch.mul:
                operation = torch.prod

            x = operation(torch.tensor(states, device=self.one.device))
        else:
            x = operation(*states)

        return (x * self.one,)


class MathCheckModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.one = nn.Parameter(torch.tensor(1.0))

    def forward(self, *x):
        return x[0] * self.one


class BinaryTreeMath(nn.Module):
    def __init__(self):
        super().__init__()
        self.leaf_module = MathBinaryTreeLeafModule()
        self.composer = MathBinaryTreeComposer()
        self.output_module = MathCheckModule()

    def forward(self, tree: Tree, inputs: torch.Tensor):
        if tree.num_children == 0:
            tree.state = self.leaf_module.forward(inputs[tree.idx - 1])
        else:
            for child in tree.children:
                self.forward(child, inputs)

            states = sum([(tree.layer,)] + [child.state for child in tree.children], ())
            tree.state = self.composer.forward(*states)

        tree.output = self.output_module.forward(*tree.state)
        return tree.output


class MathFuncSolver(nn.Module):
    def __init__(self):
        super().__init__()
        self.tree_module = BinaryTreeMath()
        self.output_module = self.tree_module.output_module

    def forward(self, tree, inputs):
        return self.tree_module(tree, inputs)


if __name__ == '__main__':
    math_func_solver = MathFuncSolver()
    math_func_solver.eval()
    exp_tree = get_proper_math_tree(10)
    inputs_of_tree = torch.tensor(exp_tree.get_leaf_values(set_idx=True, offset=1))
    r = math_func_solver(exp_tree, inputs_of_tree)

    assert torch.isclose(torch.tensor(exp_tree.gold_label, dtype=torch.float), r)
